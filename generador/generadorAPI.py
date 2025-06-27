from google import genai
from dotenv import load_dotenv
from openai import OpenAI
from functools import wraps
import pandas as pd
import os
import re
import time
import socket
import requests

try:
    import openai
except ImportError:
    openai = None

# Decorador para reintentar en caso de error de red o timeout
def retry_on_network_error(max_retries=5, initial_delay=5, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, socket.timeout, TimeoutError) as e:
                    print(f"    ⚠️  Error de red: {e}. Reintentando en {delay} segundos (intento {attempt}/{max_retries})...")
                except Exception as e:
                    # Si es un error de openai de conexión
                    if openai and hasattr(openai.error, 'APIConnectionError') and isinstance(e, openai.error.APIConnectionError):
                        print(f"    ⚠️  Error de conexión OpenAI: {e}. Reintentando en {delay} segundos (intento {attempt}/{max_retries})...")
                    else:
                        raise
                time.sleep(delay)
                delay *= backoff
            print("    ✗ Fallo permanente de red tras varios intentos.")
            return "ERROR"
        return wrapper
    return decorator

# Funciones con retry para Gemini y Deepseek
@retry_on_network_error()
def gemini_generate_content(client, enunciado):
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=enunciado
    )
    return response.text

@retry_on_network_error()
def deepseek_chat_completion(deepseek_client, messages):
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        timeout=30  # segundos
    )
    return response.choices[0].message.content

# Leer un archivo CSV de la carpeta Dataset
csv_path = os.path.join(os.path.dirname(__file__), "Dataset", "variant_1_full.csv")
df = pd.read_csv(csv_path)
print(len(df), "problemas encontrados en el dataset.")

# Automatizar procesamiento de todos los bloques del dataset
BATCH_SIZE = 10
num_problemas = len(df)
num_batches = (num_problemas + BATCH_SIZE - 1) // BATCH_SIZE

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# Inicializar cliente Deepseek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def limpiar_texto(texto):
    if isinstance(texto, str):
        return texto.replace(';', ',')
    return texto

def limpiar_codigo_python(texto):
    if not isinstance(texto, str):
        return texto
    match = re.search(r"```python(.*?)```", texto, re.DOTALL)
    if match:
        return match.group(1).strip()
    return texto.strip()

START_INDEX = 128 
END_INDEX = min(START_INDEX + BATCH_SIZE, num_problemas)
for batch in range(num_batches):
    # Cambiar el cálculo de START_INDEX y END_INDEX dentro del bucle:
    batch_start = START_INDEX + batch * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, num_problemas)
    if batch_start >= num_problemas:
        break
    print(f"\nProcesando problemas del {batch_start+1} al {batch_end} de {num_problemas}...")

    problems = []
    human_codes = []
    gemini_codes = []
    deepseek_codes = []
    gemini_novice_codes = []
    gemini_simple_codes = []
    gemini_novice_spanish_codes = []
    deepseek_novice_codes = []
    deepseek_simple_codes = []
    deepseek_novice_spanish_codes = []

    try:
        for i in range(batch_start, batch_end):
            enunciado = limpiar_texto(df.loc[i, "Problem"])
            codigo_humano = limpiar_texto(df.loc[i, "Python Code"])
            print(f"Procesando problema {i+1}/{batch_end} (Batch {batch+1}/{num_batches})")

            # Inicializar respuestas por defecto
            gemini_code = deepseek_code = gemini_code_novato = gemini_code_simple = gemini_code_novate_es = "ERROR"
            deepseek_code_novato = deepseek_code_simple = deepseek_code_novate_es = "ERROR"

            # Gemini estándar
            try:
                print("  - Solicitando respuesta estándar a Gemini...")
                gemini_raw = gemini_generate_content(client, enunciado)
                print(f"    [DEBUG] Respuesta cruda Gemini Estándar: {repr(gemini_raw)[:500]}")
                gemini_code = limpiar_codigo_python(gemini_raw)
                print("    ✓ Respuesta estándar de Gemini obtenida.")
            except Exception as e:
                print(f"    ✗ Error en Gemini estándar: {e}")
                print(f"    [DEBUG] Prompt enviado: {enunciado}")

            # Deepseek estándar
            try:
                print("  - Solicitando respuesta estándar a Deepseek...")
                deepseek_raw = deepseek_chat_completion(deepseek_client, [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": enunciado},
                ])
                print(f"    [DEBUG] Respuesta cruda Deepseek Estándar: {repr(deepseek_raw)[:500]}")
                deepseek_code = limpiar_codigo_python(deepseek_raw)
                print("    ✓ Respuesta estándar de Deepseek obtenida.")
            except Exception as e:
                print(f"    ✗ Error en Deepseek estándar: {e}")
                print(f"    [DEBUG] Prompt enviado: {enunciado}")

            # Deepseek novato (inglés)
            try:
                print("  - Solicitando respuesta Deepseek novato (inglés)...")
                deepseek_prompt_novato = (
                    enunciado +
                    "\n\nResponde como un programador novato: No uses imports de librerías externas, "
                    "no realices operaciones embebidas en una sola línea como comprensiones de listas, "
                    "ni definas funciones auxiliares. Escribe el código de la forma más simple posible."
                )
                deepseek_raw_novato = deepseek_chat_completion(deepseek_client, [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": deepseek_prompt_novato},
                ])
                print(f"    [DEBUG] Respuesta cruda Deepseek Novato Inglés: {repr(deepseek_raw_novato)[:500]}")
                deepseek_code_novato = limpiar_codigo_python(deepseek_raw_novato)
                print("    ✓ Respuesta Deepseek novato (inglés) obtenida.")
            except Exception as e:
                print(f"    ✗ Error en Deepseek novato (inglés): {e}")
                print(f"    [DEBUG] Prompt enviado: {deepseek_prompt_novato}")

            # Deepseek variables y funciones simples
            try:
                print("  - Solicitando respuesta Deepseek con variables simples...")
                deepseek_prompt_simple = (
                    enunciado +
                    "\n\nResponde generando el código Python con los nombres de variables y funciones lo más simples posible, usando letras individuales como a, b, c. No uses explicaciones."
                )
                deepseek_raw_simple = deepseek_chat_completion(deepseek_client, [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": deepseek_prompt_simple},
                ])
                print(f"    [DEBUG] Respuesta cruda Deepseek Simple Vars: {repr(deepseek_raw_simple)[:500]}")
                deepseek_code_simple = limpiar_codigo_python(deepseek_raw_simple)
                print("    ✓ Respuesta Deepseek con variables simples obtenida.")
            except Exception as e:
                print(f"    ✗ Error en Deepseek con variables simples: {e}")
                print(f"    [DEBUG] Prompt enviado: {deepseek_prompt_simple}")

            # Deepseek novato en español
            try:
                print("  - Solicitando respuesta Deepseek novato (español)...")
                deepseek_prompt_novato_es = (
                    enunciado +
                    "\n\nResponde como un programador novato: No uses imports de librerías externas, "
                    "no realices operaciones embebidas en una sola línea como comprensiones de listas, "
                    "ni definas funciones auxiliares. Escribe el código de la forma más simple posible. Además, usa nombres de variables y funciones en español y responde solo con el código."
                )
                deepseek_raw_novato_es = deepseek_chat_completion(deepseek_client, [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": deepseek_prompt_novato_es},
                ])
                print(f"    [DEBUG] Respuesta cruda Deepseek Novato Español: {repr(deepseek_raw_novato_es)[:500]}")
                deepseek_code_novato_es = limpiar_codigo_python(deepseek_raw_novato_es)
                print("    ✓ Respuesta Deepseek novato (español) obtenida.")
            except Exception as e:
                print(f"    ✗ Error en Deepseek novato (español): {e}")
                print(f"    [DEBUG] Prompt enviado: {deepseek_prompt_novato_es}")

            # Gemini novato (inglés)
            try:
                print("  - Solicitando respuesta Gemini novato (inglés)...")
                prompt_novato = (
                    enunciado +
                    "\n\nResponde como un programador novato: No uses imports de librerías externas, "
                    "no realices operaciones embebidas en una sola línea como comprensiones de listas, "
                    "ni definas funciones auxiliares. Escribe el código de la forma más simple posible."
                )
                gemini_raw_novato = gemini_generate_content(client, prompt_novato)
                print(f"    [DEBUG] Respuesta cruda Gemini Novato Inglés: {repr(gemini_raw_novato)[:500]}")
                gemini_code_novato = limpiar_codigo_python(gemini_raw_novato)
                print("    ✓ Respuesta Gemini novato (inglés) obtenida.")
            except Exception as e:
                print(f"    ✗ Error en Gemini novato (inglés): {e}")
                print(f"    [DEBUG] Prompt enviado: {prompt_novato}")

            # Gemini variables y funciones simples
            try:
                print("  - Solicitando respuesta Gemini con variables simples...")
                prompt_simple = (
                    enunciado +
                    "\n\nResponde generando el código Python con los nombres de variables y funciones lo más simples posible, usando letras individuales como a, b, c. No uses explicaciones."
                )
                gemini_raw_simple = gemini_generate_content(client, prompt_simple)
                print(f"    [DEBUG] Respuesta cruda Gemini Simple Vars: {repr(gemini_raw_simple)[:500]}")
                gemini_code_simple = limpiar_codigo_python(gemini_raw_simple)
                print("    ✓ Respuesta Gemini con variables simples obtenida.")
            except Exception as e:
                print(f"    ✗ Error en Gemini con variables simples: {e}")
                print(f"    [DEBUG] Prompt enviado: {prompt_simple}")

            # Gemini novato en español
            try:
                print("  - Solicitando respuesta Gemini novato (español)...")
                prompt_novato_es = (
                    enunciado +
                    "\n\nResponde como un programador novato: No uses imports de librerías externas, "
                    "no realices operaciones embebidas en una sola línea como comprensiones de listas, "
                    "ni definas funciones auxiliares. Escribe el código de la forma más simple posible. Además, usa nombres de variables y funciones en español y responde solo con el código."
                )
                gemini_raw_novato_es = gemini_generate_content(client, prompt_novato_es)
                print(f"    [DEBUG] Respuesta cruda Gemini Novato Español: {repr(gemini_raw_novato_es)[:500]}")
                gemini_code_novato_es = limpiar_codigo_python(gemini_raw_novato_es)
                print("    ✓ Respuesta Gemini novato (español) obtenida.")
            except Exception as e:
                print(f"    ✗ Error en Gemini novato (español): {e}")
                print(f"    [DEBUG] Prompt enviado: {prompt_novato_es}")

            problems.append(enunciado)
            human_codes.append(codigo_humano)
            gemini_codes.append(limpiar_texto(gemini_code))
            deepseek_codes.append(limpiar_texto(deepseek_code))
            gemini_novice_codes.append(limpiar_texto(gemini_code_novato))
            gemini_simple_codes.append(limpiar_texto(gemini_code_simple))
            gemini_novice_spanish_codes.append(limpiar_texto(gemini_code_novate_es))
            deepseek_novice_codes.append(limpiar_texto(deepseek_code_novato))
            deepseek_simple_codes.append(limpiar_texto(deepseek_code_simple))
            deepseek_novice_spanish_codes.append(limpiar_texto(deepseek_code_novate_es))
    except KeyboardInterrupt:
        print("\nInterrupción detectada. Guardando resultados procesados hasta el momento...")
        # Asegurar que todas las listas tengan la misma longitud (hasta la última fila completa)
        min_len = min(len(problems), len(human_codes), len(gemini_codes), len(deepseek_codes),
                      len(gemini_novice_codes), len(gemini_simple_codes), len(gemini_novice_spanish_codes),
                      len(deepseek_novice_codes), len(deepseek_simple_codes), len(deepseek_novice_spanish_codes))
        problems = problems[:min_len]
        human_codes = human_codes[:min_len]
        gemini_codes = gemini_codes[:min_len]
        deepseek_codes = deepseek_codes[:min_len]
        gemini_novice_codes = gemini_novice_codes[:min_len]
        gemini_simple_codes = gemini_simple_codes[:min_len]
        gemini_novice_spanish_codes = gemini_novice_spanish_codes[:min_len]
        deepseek_novice_codes = deepseek_novice_codes[:min_len]
        deepseek_simple_codes = deepseek_simple_codes[:min_len]
        deepseek_novice_spanish_codes = deepseek_novice_spanish_codes[:min_len]
        # Guardar resultados parciales
        output_df = pd.DataFrame({
            'Problem': problems,
            'Human_Code': human_codes,
            'Gemini_Answer': gemini_codes,
            'Deepseek_Answer': deepseek_codes,
            'Gemini_Novice_Answer': gemini_novice_codes,
            'Gemini_SimpleVars_Answer': gemini_simple_codes,
            'Gemini_NoviceSpanish_Answer': gemini_novice_spanish_codes,
            'Deepseek_Novice_Answer': deepseek_novice_codes,
            'Deepseek_SimpleVars_Answer': deepseek_simple_codes,
            'Deepseek_NoviceSpanish_Answer': deepseek_novice_spanish_codes
        })
        output_dir = os.path.join(os.path.dirname(__file__), "resultados")
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, f"output_{batch_start+1}_{batch_start+len(problems)}_INTERRUPTED.csv")
        output_df.to_csv(output_csv_path, index=False, sep=';')
        print(f"Archivo CSV parcial guardado con los problemas del {batch_start+1} al {batch_start+len(problems)} en: {output_csv_path}")
        break

    # Guardar resultados en un DataFrame y exportar a CSV por lote
    output_df = pd.DataFrame({
        'Problem': problems,
        'Human_Code': human_codes,
        'Gemini_Answer': gemini_codes,
        'Deepseek_Answer': deepseek_codes,
        'Gemini_Novice_Answer': gemini_novice_codes,
        'Gemini_SimpleVars_Answer': gemini_simple_codes,
        'Gemini_NoviceSpanish_Answer': gemini_novice_spanish_codes,
        'Deepseek_Novice_Answer': deepseek_novice_codes,
        'Deepseek_SimpleVars_Answer': deepseek_simple_codes,
        'Deepseek_NoviceSpanish_Answer': deepseek_novice_spanish_codes
    })
    print(f"\nGuardando resultados del batch {batch+1}/{num_batches}...")
    output_dir = os.path.join(os.path.dirname(__file__), "resultados")
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"output_{batch_start+1}_{batch_end}.csv")
    output_df.to_csv(output_csv_path, index=False, sep=';')
    print(f"Archivo CSV generado con los problemas del {batch_start+1} al {batch_end} en: {output_csv_path}")
import os
import pandas as pd
import re
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import time
import requests
from requests.exceptions import ConnectionError, Timeout

# --- Utilidades de limpieza ---
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

# --- Inicialización de APIs ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# --- Funciones de consulta ---
def gemini_generate_content(prompt):
    try:
        print(f"    [DEBUG] Prompt Gemini: {repr(prompt)[:300]}")
        response = client.models.generate_content(model="gemini-2.5-pro", contents=prompt)
        print(f"    [DEBUG] Respuesta cruda Gemini: {repr(response.text)[:500]}")
        return limpiar_codigo_python(response.text)
    except Exception as e:
        print(f"[Gemini ERROR] {e}")
        return "ERROR"

def deepseek_chat_completion(messages):
    try:
        print(f"    [DEBUG] Prompt Deepseek: {repr(messages[1]['content'])[:300]}")
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
            timeout=30
        )
        print(f"    [DEBUG] Respuesta cruda Deepseek: {repr(response.choices[0].message.content)[:500]}")
        return limpiar_codigo_python(response.choices[0].message.content)
    except Exception as e:
        print(f"[Deepseek ERROR] {e}")
        return "ERROR"

# --- Prompts por rol ---
def get_prompt(role, enunciado):
    if role == 'Gemini_Answer':
        return enunciado
    if role == 'Gemini_Novice_Answer':
        return enunciado + "\n\nResponde como un programador novato: No uses imports de librerías externas, no realices operaciones embebidas en una sola línea como comprensiones de listas, ni definas funciones auxiliares. Escribe el código de la forma más simple posible."
    if role == 'Gemini_SimpleVars_Answer':
        return enunciado + "\n\nResponde generando el código Python con los nombres de variables y funciones lo más simples posible, usando letras individuales como a, b, c. No uses explicaciones."
    if role == 'Gemini_NoviceSpanish_Answer':
        return enunciado + "\n\nResponde como un programador novato: No uses imports de librerías externas, no realices operaciones embebidas en una sola línea como comprensiones de listas, ni definas funciones auxiliares. Escribe el código de la forma más simple posible. Además, usa nombres de variables y funciones en español y responde solo con el código."
    if role == 'Deepseek_Answer':
        return enunciado
    if role == 'Deepseek_Novice_Answer':
        return enunciado + "\n\nResponde como un programador novato: No uses imports de librerías externas, no realices operaciones embebidas en una sola línea como comprensiones de listas, ni definas funciones auxiliares. Escribe el código de la forma más simple posible."
    if role == 'Deepseek_SimpleVars_Answer':
        return enunciado + "\n\nResponde generando el código Python con los nombres de variables y funciones lo más simples posible, usando letras individuales como a, b, c. No uses explicaciones."
    if role == 'Deepseek_NoviceSpanish_Answer':
        return enunciado + "\n\nResponde como un programador novato: No uses imports de librerías externas, no realices operaciones embebidas en una sola línea como comprensiones de listas, ni definas funciones auxiliares. Escribe el código de la forma más simple posible. Además, usa nombres de variables y funciones en español y responde solo con el código."
    return enunciado

# --- Saneador principal ---
def sanar_csv(path_csv):
    print(f"\nProcesando archivo: {os.path.basename(path_csv)}")
    df = pd.read_csv(path_csv, sep=';')
    roles = [col for col in df.columns if 'Answer' in col]
    cambios = 0
    try:
        for idx, row in df.iterrows():
            enunciado = limpiar_texto(row['Problem'])
            for role in roles:
                if str(row[role]).strip() == 'ERROR':
                    prompt = get_prompt(role, enunciado)
                    if role.startswith('Gemini'):
                        nuevo = gemini_generate_content(prompt)
                    else:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt},
                        ]
                        nuevo = deepseek_chat_completion(messages)
                    if nuevo != 'ERROR':
                        print(f"  ✓ Reparado {role} en problema {idx+1}")
                    else:
                        print(f"  ✗ Sigue fallando {role} en problema {idx+1}")
                    df.at[idx, role] = limpiar_texto(nuevo)
                    cambios += 1
                    time.sleep(1)  # Para evitar rate limit
    except (requests.exceptions.RequestException, ConnectionError, Timeout, OSError, KeyboardInterrupt) as e:
        print(f"\n[CRÍTICO] Conexión perdida o interrupción detectada: {e}")
        # Guardar avance parcial
        out_path = path_csv.replace('.csv', '_sanitized_PARTIAL.csv')
        df.to_csv(out_path, index=False, sep=';')
        print(f"Progreso parcial guardado en: {out_path}")
        print("Puedes reanudar el saneamiento ejecutando de nuevo el script.")
        return
    if cambios:
        out_path = path_csv.replace('.csv', '_sanitized.csv')
        df.to_csv(out_path, index=False, sep=';')
        print(f"Guardado archivo saneado: {out_path}")
    else:
        print("No se detectaron errores a reparar.")

if __name__ == "__main__":
    resultados_dir = os.path.join(os.path.dirname(__file__), "resultados")
    archivos = [f for f in os.listdir(resultados_dir) if f.endswith('.csv')]
    for archivo in archivos:
        path = os.path.join(resultados_dir, archivo)
        sanar_csv(path)

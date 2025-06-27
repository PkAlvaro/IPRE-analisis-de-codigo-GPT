# Human-AI Code Stylometry Classifier: IPRE Analisis de código generado por GPT

Este proyecto permite evaluar, analizar y visualizar la capacidad de un modelo basado en T5 para distinguir entre código generado por humanos y por IA. Incluye scripts para inferencia, análisis detallado, visualización y manejo robusto de datasets.

## 1. Instalación del entorno

### a) Crear y activar un entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate
```

### b) Instalar los requerimientos

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Generación y unificación del dataset

### a) Configura tus credenciales de API

Crea un archivo `.env` en la carpeta `generador/` con el siguiente formato:

```
DEEPSEEK_API_KEY=tu_api_key_deepseek
gemini_api_key=tu_api_key_gemini
```

### b) Genera los resultados de IA

Desde la carpeta raíz, ejecuta:

```bash
cd generador
python generadorAPI.py
```

Esto generará varios archivos `output_*.csv` en `generador/resultados/`.

### c) Unifica y verifica el dataset

```bash
python analiza_resultados.py
```

- Este script verifica si el dataset está completo.
- Si hay problemas de integridad, ejecuta `sanea_resultados.py` para limpiar y reintentar la generación de outputs faltantes.
- Cuando esté completo, `analiza_resultados.py` generará el archivo unificado `resultados_unificados.csv` en `generador/resultados/`.

## 3. Inferencia y análisis offline

### a) Descarga el modelo y el checkpoint manualmente

- Descarga el modelo base y los archivos de tokenizer desde HuggingFace y colócalos en la carpeta `codet5p-770m-local/`.
- Descarga el archivo `checkpoint.bin` desde el sitio oficial del estudio:
  https://huggingface.co/spaces/isThisYouLLM/Human-Ai/tree/main
- Coloca `checkpoint.bin` en la **raíz del proyecto**.

### b) Ejecuta la inferencia y el análisis

```bash
python test_model.py
python analisis_detallado.py
```

Esto generará el archivo `analisis_resultados/reporte_resultados.csv` y visualizaciones avanzadas.

## 4. Interpretación de resultados

- Revisa los archivos CSV y las imágenes en `analisis_resultados/`.
- El archivo `problemas_conflictivos.csv` incluye los enunciados de los problemas donde el modelo más falla.

## 5. Notas

- El dataset de entrada debe estar en `generador/resultados/resultados_unificados.csv`.
- El modelo se descarga y cachea automáticamente la primera vez, pero para uso offline debes descargar los archivos manualmente como se indica.
- Para reproducibilidad, los scripts fijan la semilla aleatoria donde es relevante.

---

Para dudas o mejoras, revisa los scripts y los comentarios en el código.

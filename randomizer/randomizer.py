from test_model import stylometer_classifier, split_code_into_chunks, UMBRAL
from transformers import AutoTokenizer, T5EncoderModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import traceback
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



csv_path = os.path.join(os.path.dirname(__file__), '../tests/resultados_unificados.csv')
df = pd.read_csv(csv_path, sep=';')

code_columns = [col for col in df.columns if 'code' in col.lower() or 'answer' in col.lower()]

human_col = None
for col in code_columns:
    if col.lower() == 'human_code':
        human_col = col
        break
ia_cols = [col for col in code_columns if col != human_col]

# Filtrar solo filas con código humano válido
mask_human = df[human_col].notna() & (df[human_col].astype(str).str.lower() != 'nan')
human_rows = df[mask_human].reset_index(drop=True)

# Para cada humano, seleccionar un IA aleatorio de cualquier columna IA y fila
np.random.seed(42)
random_ia_codes = []
for _ in range(len(human_rows)):
    ia_col = np.random.choice(ia_cols)
    # Filas válidas en esa columna
    valid_ia = df[ia_col].dropna()
    valid_ia = valid_ia[valid_ia.astype(str).str.lower() != 'nan']
    if len(valid_ia) == 0:
        random_ia_codes.append('')
    else:
        ia_code = valid_ia.sample(1).values[0]
        random_ia_codes.append(ia_code)

# Crear nuevo DataFrame pareado
df_random = pd.DataFrame({
    'Human_Code': human_rows[human_col],
    'IA_Code': random_ia_codes
})

# Guardar resultado
output_dir = os.path.join(os.path.dirname(__file__), 'random_pairs')
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'parejas_Human_vs_IA_random.csv')
df_random.to_csv(out_path, index=False, sep=';')
print(f"Parejas aleatorias Human vs IA guardadas en {out_path} ({len(df_random)} filas)")

# --- INFERENCIA Y ANÁLISIS SOBRE EL SUBSET ALEATORIO ---
model_name = "Salesforce/codet5p-770m"
local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../codet5p-770m-local'))
checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), '../checkpoint.bin'))

if not os.path.exists(local_dir):
    print("Descargando modelo y tokenizer localmente...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_dir)
    model = T5EncoderModel.from_pretrained(model_name)
    model.save_pretrained(local_dir)
else:
    print("Usando modelo y tokenizer local guardados.")

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_dir)
print("Cargando modelo base...")
model = T5EncoderModel.from_pretrained(local_dir)

model = stylometer_classifier(model, dimensionality=model.shared.embedding_dim)
if os.path.exists(checkpoint):
    print("Cargando checkpoint de pesos...")
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
else:
    print("Advertencia: No se encontró el checkpoint. El modelo no está entrenado.")
model = model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Inferencia y análisis
max_len = tokenizer.model_max_length
report = []
for idx, row in df_random.iterrows():
    for col in ['Human_Code', 'IA_Code']:
        try:
            code = str(row[col])
            if not code or code.lower() == 'nan':
                continue
            chunks = split_code_into_chunks(code, tokenizer, max_len)
            chunk_probs = []
            for i, chunk in enumerate(chunks):
                try:
                    inputs = tokenizer([chunk], return_tensors="pt", max_length=max_len, truncation=True)
                    input_ids = inputs.input_ids.to(device)
                    attention_mask = inputs.attention_mask.to(device)
                    with torch.no_grad():
                        out = model(input_ids, attention_mask)
                    prob_ia = out['prob']
                    chunk_probs.append(prob_ia)
                except Exception as e:
                    print(f"[ERROR] Fallo al procesar fragmento {i+1} de la columna {col} en la fila {idx+1}: {e}")
                    continue
            if not chunk_probs:
                print(f"[ADVERTENCIA] No se pudo procesar ningún fragmento para columna {col} en fila {idx+1}.")
                continue
            avg_prob = sum(chunk_probs) / len(chunk_probs)
            veredicto = "Humano" if avg_prob >= UMBRAL else "IA"
            ground_truth = "Humano" if col == 'Human_Code' else "IA"
            acierto = "✔️" if veredicto == ground_truth else "❌"
            report.append({
                "Problema": idx+1,
                "Columna": col,
                "Predicción": veredicto,
                "Confianza_IA": avg_prob,
                "GroundTruth": ground_truth,
                "Acierto": acierto
            })
        except Exception as e:
            print(f"[ERROR] Fallo al procesar columna {col} en fila {idx+1}: {e}")
            traceback.print_exc()
            continue

# Guardar y visualizar resultados
out_dir = os.path.join(output_dir, 'analisis_random')
os.makedirs(out_dir, exist_ok=True)
df_report = pd.DataFrame(report)
df_report.to_csv(os.path.join(out_dir, 'reporte_resultados.csv'), index=False)
print(f"\nReporte guardado en {os.path.join(out_dir, 'reporte_resultados.csv')}")

if not df_report.empty:
    gt = df_report['GroundTruth']
    pred = df_report['Predicción']
    labels = ['Humano', 'IA']
    acc = accuracy_score(gt, pred)
    prec = precision_score(gt, pred, average='macro', zero_division=0)
    rec = recall_score(gt, pred, average='macro', zero_division=0)
    f1 = f1_score(gt, pred, average='macro', zero_division=0)
    cm = confusion_matrix(gt, pred, labels=labels)
    # Guardar métricas
    metrics_txt = os.path.join(out_dir, 'metricas.txt')
    with open(metrics_txt, 'w') as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision (macro): {prec:.4f}\n")
        f.write(f"Recall (macro):    {rec:.4f}\n")
        f.write(f"F1 (macro):        {f1:.4f}\n")
        f.write(f"Matriz de confusión (filas=esperado, columnas=predicho):\n")
        f.write(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    metrics_csv = os.path.join(out_dir, 'metricas.csv')
    pd.DataFrame({
        'Accuracy': [acc],
        'Precision_macro': [prec],
        'Recall_macro': [rec],
        'F1_macro': [f1]
    }).to_csv(metrics_csv, index=False)
    print(f"Métricas guardadas en {metrics_txt} y {metrics_csv}")
    # Visualizaciones
    plt.figure(figsize=(7,4))
    sns.histplot(df_report['Confianza_IA'].astype(float), bins=30, kde=True, color='purple')
    plt.xlabel('Confianza IA')
    plt.title('Distribución de la confianza IA')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'distribucion_confianza_ia.png'))
    plt.close()
    df_report['Acierto_plot'] = df_report['Acierto'].replace({'✔️': 'Correcto', '❌': 'Incorrecto'})
    sns.countplot(x='Acierto_plot', data=df_report, palette='Set2', hue='Acierto_plot', legend=False)
    plt.title('Distribución de aciertos')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'aciertos.png'))
    plt.close()
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicho')
    plt.ylabel('Esperado')
    plt.title('Matriz de confusión')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'matriz_confusion.png'))
    plt.close()
    print(f"Visualizaciones guardadas en {out_dir}/")

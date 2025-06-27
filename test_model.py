import os
import sys
import torch
from transformers import AutoTokenizer, T5EncoderModel
import pandas as pd
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model_name = "Salesforce/codet5p-770m"
local_dir = "./codet5p-770m-local"
checkpoint = "checkpoint.bin"

# Descargar y guardar localmente si no existe
if not os.path.exists(local_dir):
    print("Descargando modelo y tokenizer localmente...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_dir)
    model = T5EncoderModel.from_pretrained(model_name)
    model.save_pretrained(local_dir)
else:
    print("Usando modelo y tokenizer local guardados.")

# Cargar desde local
print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_dir)
print("Cargando modelo base...")
model = T5EncoderModel.from_pretrained(local_dir)

UMBRAL = 0.07

class stylometer_classifier(torch.nn.Module):
    def __init__(self,pretrained_encoder,dimensionality):
        super(stylometer_classifier, self).__init__()
        self.modelBase = pretrained_encoder
        self.pre_classifier = torch.nn.Linear(dimensionality, 768)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, padding_mask):
        output_1 = self.modelBase(input_ids=input_ids, attention_mask=padding_mask)
        hidden_state = output_1[0]
        cls_output = hidden_state[:, 0]
        pooler = self.pre_classifier(cls_output)
        afterActivation = self.activation(pooler)
        pooler_after_act = self.dropout(afterActivation)
        output = torch.sigmoid(self.classifier(pooler_after_act))
        if output >= UMBRAL:
            return {"my_class":"Es Humano!","prob":output.item()}
        else:
            return {"my_class":"Es IA!","prob":output.item()}

# Adaptar el modelo
model = stylometer_classifier(model, dimensionality=model.shared.embedding_dim)

# Cargar checkpoint
if os.path.exists(checkpoint):
    print("Cargando checkpoint de pesos...")
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
else:
    print("Advertencia: No se encontró el checkpoint. El modelo no está entrenado.")

model = model.eval()

# Usar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Tokenizar el código y dividirlo en fragmentos
def split_code_into_chunks(code, tokenizer, max_len):
    tokens = tokenizer(code, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_len):
        chunk_tokens = tokens[i:i+max_len]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificador de código Humano/IA por fila de CSV.")
    parser.add_argument("--index", type=int, default=None, help="Índice de la fila a procesar (0 para la primera, etc). Si no se especifica, procesa todo el dataset.")
    args = parser.parse_args()
    csv_path = os.path.join('generador', 'resultados', 'resultados_unificados.csv')
    row_index = args.index
    if not os.path.exists(csv_path):
        print(f"No se encontró el archivo: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path, sep=None, engine='python')
    max_len = tokenizer.model_max_length
    print(f"(Límite de tokens del modelo: {max_len})")
    code_columns = [col for col in df.columns if 'code' in col.lower() or 'answer' in col.lower()]
    if not code_columns:
        print("No se encontraron columnas de código en el CSV.")
        sys.exit(1)
    if row_index is not None:
        if row_index < 0 or row_index >= len(df):
            print(f"Índice fuera de rango. El dataset tiene {len(df)} filas.")
            sys.exit(1)
        rows = [(row_index, df.iloc[row_index])]
    else:
        rows = list(df.iterrows())
    for idx, row in rows:
        print(f"\n--- Ejemplo {idx+1} ---")
        for col in code_columns:
            code = str(row[col])
            if not code or code.lower() == 'nan':
                continue
            print(f"\nColumna: {col}")
            chunks = split_code_into_chunks(code, tokenizer, max_len)
            chunk_probs = []
            for i, chunk in enumerate(chunks):
                inputs = tokenizer([chunk], return_tensors="pt", max_length=max_len, truncation=True)
                num_tokens = inputs.input_ids.shape[1]
                if num_tokens >= max_len:
                    print(f"[ADVERTENCIA] Fragmento {i+1} truncado a {max_len} tokens.")
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                with torch.no_grad():
                    out = model(input_ids, attention_mask)
                prob_ia = out['prob']
                chunk_probs.append(prob_ia)
            avg_prob = sum(chunk_probs) / len(chunk_probs)
            print(f"Probabilidad promedio de IA: {avg_prob:.2%} (basado en {len(chunks)} fragmentos)")
            print(f"(Si la probabilidad es menor a {UMBRAL*100:.2f}%, se clasifica como IA. Si es mayor o igual, como Humano.)")
            veredicto = "Es Humano!" if avg_prob >= UMBRAL else "Es IA!"
            print(f"Veredicto final: {veredicto}")

    report = []
    for idx, row in rows:
        for col in code_columns:
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
                if "human" in col.lower():
                    ground_truth = "Humano"
                else:
                    ground_truth = "IA"
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
    
#Presentacion y guardado de resultados
    print("\n--- REPORTE DE RESULTADOS ---")
    print(f"{'#Prob':<6}{'Columna':<30}{'Predicción':<12}{'Conf_IA':<10}{'GT':<10}{'Acierto':<8}")
    for r in report:
        print(f"{r['Problema']:<6}{r['Columna']:<30}{r['Predicción']:<12}{r['Confianza_IA']:.2%}   {r['GroundTruth']:<10}{r['Acierto']:<8}")
    df_report = pd.DataFrame(report)
    output_dir = 'analisis_resultados'
    os.makedirs(output_dir, exist_ok=True)
    df_report.to_csv(os.path.join(output_dir, 'reporte_resultados.csv'), index=False)
    print(f"\nReporte guardado en {output_dir}/reporte_resultados.csv")

    # Plots básicos de distribución de confianza y aciertos
    if not df_report.empty:
        plt.figure(figsize=(7,4))
        sns.histplot(df_report['Confianza_IA'].astype(float), bins=30, kde=True, color='purple')
        plt.xlabel('Confianza IA')
        plt.title('Distribución de la confianza IA')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribucion_confianza_ia.png'))
        plt.close()
        # Aciertos por clase (sin FutureWarning y sin símbolos unicode)
        df_report['Acierto_plot'] = df_report['Acierto'].replace({'✔️': 'Correcto', '❌': 'Incorrecto'})
        sns.countplot(x='Acierto_plot', data=df_report, palette='Set2', hue='Acierto_plot', legend=False)
        plt.title('Distribución de aciertos')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aciertos.png'))
        plt.close()
        print(f"Visualizaciones guardadas en {output_dir}/")

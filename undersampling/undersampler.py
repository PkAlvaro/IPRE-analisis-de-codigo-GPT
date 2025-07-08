from transformers import AutoTokenizer, T5EncoderModel
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import traceback

csv_path = 'tests/resultados_unificados.csv'
df = pd.read_csv(csv_path, sep=';')
code_columns = [col for col in df.columns if 'code' in col.lower() or 'answer' in col.lower()]
output_dir = os.path.join(os.path.dirname(__file__), 'minidatasets_balanceados')
os.makedirs(output_dir, exist_ok=True)


# --- Generar CSVs de parejas: Human_Code vs cada IA_code ---
human_col = None
for col in code_columns:
    if col.lower() == 'human_code':
        human_col = col
        break
pareados = []
if human_col is not None:
    for ia_col in code_columns:
        if ia_col == human_col:
            continue
        mask = df[human_col].notna() & (df[human_col].astype(str).str.lower() != 'nan') \
               & df[ia_col].notna() & (df[ia_col].astype(str).str.lower() != 'nan')
        df_pairs = df.loc[mask, [human_col, ia_col]].reset_index(drop=True)
        df_pairs = df_pairs.rename(columns={human_col: 'Human_Code', ia_col: 'IA_Code'})
        out_path = os.path.join(output_dir, f'parejas_Human_vs_{ia_col}.csv')
        df_pairs.to_csv(out_path, index=False, sep=';')
        print(f"Parejas Human_Code vs {ia_col}: {len(df_pairs)} filas -> Guardado en {out_path}")
        pareados.append((ia_col, out_path))

model_name = "Salesforce/codet5p-770m"
local_dir = "./codet5p-770m-local"
checkpoint = "checkpoint.bin"

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

model = stylometer_classifier(model, dimensionality=model.shared.embedding_dim)
if os.path.exists(checkpoint):
    print("Cargando checkpoint de pesos...")
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
else:
    print("Advertencia: No se encontró el checkpoint. El modelo no está entrenado.")
model = model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def split_code_into_chunks(code, tokenizer, max_len):
    tokens = tokenizer(code, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), max_len):
        chunk_tokens = tokens[i:i+max_len]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def run_inference_on_pareado(csv_path, output_dir, ia_col):
    dfp = pd.read_csv(csv_path, sep=';')
    max_len = tokenizer.model_max_length
    report = []
    for idx, row in dfp.iterrows():
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
    df_report = pd.DataFrame(report)
    out_dir = os.path.join(output_dir, f'analisis_{ia_col}')
    os.makedirs(out_dir, exist_ok=True)
    df_report.to_csv(os.path.join(out_dir, 'reporte_resultados.csv'), index=False)
    print(f"\nReporte guardado en {out_dir}/reporte_resultados.csv")

    # Métricas
    if not df_report.empty:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        gt = df_report['GroundTruth']
        pred = df_report['Predicción']
        labels = ['Humano', 'IA']
        acc = accuracy_score(gt, pred)
        prec = precision_score(gt, pred, average='macro', zero_division=0)
        rec = recall_score(gt, pred, average='macro', zero_division=0)
        f1 = f1_score(gt, pred, average='macro', zero_division=0)
        cm = confusion_matrix(gt, pred, labels=labels)
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

# Ejecutar análisis para cada par
for ia_col, csv_path in pareados:
    print(f"\n--- Analizando par Human_Code vs {ia_col} ---")
    run_inference_on_pareado(csv_path, output_dir, ia_col)



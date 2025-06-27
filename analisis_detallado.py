import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar el reporte de resultados
df = pd.read_csv('reporte_resultados.csv', header=None,
                 names=['Numero', 'Nombre', 'Esperado', 'Confianza_IA', 'Predicho', 'Correcto'])

# Eliminar fila de encabezado si está presente como dato
if (df.iloc[0] == ['Numero', 'Nombre', 'Esperado', 'Confianza_IA', 'Predicho', 'Correcto']).all():
    df = df.iloc[1:].reset_index(drop=True)

# Eliminar cualquier fila que sea encabezado accidental
encabezados = ['Numero', 'Nombre', 'Esperado', 'Confianza_IA', 'Predicho', 'Correcto', 'Columna']
df = df[~df['Nombre'].isin(['Nombre', 'Columna'])]
df = df[~df['Numero'].isin(['Numero'])]
df = df.reset_index(drop=True)

# 1. Métricas globales
esperado = df['Esperado']
predicho = df['Predicho']

accuracy = accuracy_score(esperado, predicho)
precision = precision_score(esperado, predicho, average='macro', zero_division=0)
recall = recall_score(esperado, predicho, average='macro', zero_division=0)
f1 = f1_score(esperado, predicho, average='macro', zero_division=0)
cm = confusion_matrix(esperado, predicho, labels=['Humano', 'IA'])

print('--- MÉTRICAS GLOBALES ---')
print(f'Accuracy:  {accuracy:.4f}')
print(f'Precision (macro): {precision:.4f}')
print(f'Recall (macro):    {recall:.4f}')
print(f'F1 (macro):        {f1:.4f}')
print('Matriz de confusión (filas=esperado, columnas=predicho):')
print(pd.DataFrame(cm, index=['Humano', 'IA'], columns=['Humano', 'IA']))

# Métricas por clase
print('\n--- MÉTRICAS POR CLASE ---')
print(classification_report(esperado, predicho, digits=3, zero_division=0))

# 2. Análisis por tipo de código (Nombre)
print('\n--- DESEMPEÑO POR TIPO DE CÓDIGO (Nombre) ---')
for nombre, subdf in df.groupby('Nombre'):
    acc = accuracy_score(subdf['Esperado'], subdf['Predicho'])
    print(f'{nombre:30s}  Accuracy: {acc:.3f}  Casos: {len(subdf)}')

# 3. Problemas donde más se equivoca el modelo
errores = df[df['Correcto'] == '❌']
errores_por_problema = errores.groupby('Numero').size().sort_values(ascending=False)

print('\n--- PROBLEMAS CON MÁS ERRORES ---')
print(errores_por_problema.head(10))

# 4. Exportar los problemas más conflictivos para inspección manual
N = 10  # Top N problemas conflictivos
problemas_conflictivos = errores_por_problema.head(N).index.tolist()
subset = df[df['Numero'].isin(problemas_conflictivos)]
# Crear carpeta de resultados si no existe
output_dir = 'analisis_resultados'
os.makedirs(output_dir, exist_ok=True)

# Guardar archivos en la carpeta dedicada
subset.to_csv(os.path.join(output_dir, 'problemas_conflictivos.csv'), index=False)
print(f'\nSe exportó el detalle de los {N} problemas con más errores a {output_dir}/problemas_conflictivos.csv')

# --- AGREGAR ENUNCIADO AL CSV DE PROBLEMAS CONFLICTIVOS ---
# Cargar dataset original con enunciados
try:
    df_enun = pd.read_csv('generador/resultados/resultados_unificados.csv', sep=';')
    subset = pd.read_csv(os.path.join(output_dir, 'problemas_conflictivos.csv'))
    # Para cada fila, buscar el enunciado correspondiente a ese número de problema (Numero es 1-indexado, df_enun usa index 0)
    subset['Enunciado'] = subset['Numero'].apply(lambda n: df_enun.loc[int(n)-1, 'Problem'] if 0 <= int(n)-1 < len(df_enun) else '')
    subset.to_csv(os.path.join(output_dir, 'problemas_conflictivos.csv'), index=False)
    print(f"Se agregó la columna 'Enunciado' a {output_dir}/problemas_conflictivos.csv")
except Exception as e:
    print(f"[ERROR] No se pudo agregar el enunciado a problemas_conflictivos.csv: {e}")

# --- VISUALIZACIONES ---

# 1. Matriz de confusión
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Humano', 'IA'], yticklabels=['Humano', 'IA'])
plt.xlabel('Predicho')
plt.ylabel('Esperado')
plt.title('Matriz de confusión')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'matriz_confusion.png'))

# 2. Accuracy por tipo de código (sin FutureWarning)
accs = df.groupby('Nombre').apply(lambda x: accuracy_score(x['Esperado'], x['Predicho']), include_groups=False)
if isinstance(accs.index, pd.MultiIndex):
    accs.index = accs.index.droplevel(-1)
accs.index.name = None
plt.figure(figsize=(10,5))
accs.sort_values().plot(kind='barh', color='skyblue')
plt.xlabel('Accuracy')
plt.title('Accuracy por tipo de código (Nombre)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_por_tipo_codigo.png'))

# 3. Errores por número de problema (top 10)
plt.figure(figsize=(8,4))
errores_por_problema.head(10).sort_values().plot(kind='barh', color='salmon')
plt.xlabel('Cantidad de errores')
plt.title('Top 10 problemas con más errores')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'errores_por_problema.png'))

# 4. Distribución de la confianza IA
plt.figure(figsize=(7,4))
sns.histplot(df['Confianza_IA'].astype(float), bins=30, kde=True, color='purple')
plt.xlabel('Confianza IA')
plt.title('Distribución de la confianza IA')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'distribucion_confianza_ia.png'))

# 5. Resumen de hallazgos
print('\n--- RESUMEN ---')
print(f'Total de casos: {len(df)}')
print(f'Total de errores: {len(errores)}')
print(f'Porcentaje de error: {100*len(errores)/len(df):.2f}%')
print(f'Problemas conflictivos: {problemas_conflictivos}')

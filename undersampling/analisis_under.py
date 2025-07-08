import os
import pandas as pd

# Directorio donde están los análisis de cada par
base_dir = os.path.join(os.path.dirname(__file__), 'minidatasets_balanceados')

# Buscar subcarpetas de análisis
analisis_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                 if d.startswith('analisis_') and os.path.isdir(os.path.join(base_dir, d))]

#  métricas de cada subgrupo
resumen = []
for adir in analisis_dirs:
    nombre = os.path.basename(adir).replace('analisis_', '')
    met_csv = os.path.join(adir, 'metricas.csv')
    if os.path.exists(met_csv):
        df = pd.read_csv(met_csv)
        row = df.iloc[0].to_dict()
        row['Subgrupo'] = nombre
        resumen.append(row)
    else:
        print(f"[ADVERTENCIA] No se encontró metricas.csv en {adir}")

# tabla resumen
if resumen:
    df_resumen = pd.DataFrame(resumen)
    cols = ['Subgrupo', 'Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro']
    df_resumen = df_resumen[cols]
    df_resumen = df_resumen.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    print("\n--- RESUMEN DE DESEMPEÑO POR SUBGRUPO ---")
    print(df_resumen)
    df_resumen.to_csv(os.path.join(base_dir, 'resumen_metricas_subgrupos.csv'), index=False)
    print(f"\nResumen guardado en {os.path.join(base_dir, 'resumen_metricas_subgrupos.csv')}")
else:
    print("No se encontraron métricas para ningún subgrupo.")

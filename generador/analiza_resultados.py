import os
import pandas as pd

def unificar_csvs(resultados_dir, nombre_salida='resultados_unificados.csv'):
    archivos = [f for f in os.listdir(resultados_dir) if f.endswith('.csv') and not f.startswith('.')]
    dataframes = []
    for archivo in archivos:
        path = os.path.join(resultados_dir, archivo)
        try:
            df = pd.read_csv(path, sep=';')
            dataframes.append(df)
            print(f"Archivo agregado: {archivo} ({len(df)} filas)")
        except Exception as e:
            print(f"No se pudo leer {archivo}: {e}")
    if dataframes:
        df_unificado = pd.concat(dataframes, ignore_index=True)
        out_path = os.path.join(resultados_dir, nombre_salida)
        df_unificado.to_csv(out_path, index=False, sep=';')
        print(f"\nArchivo unificado generado: {nombre_salida} ({len(df_unificado)} filas)")
    else:
        print("No se encontraron archivos para unificar.")

if __name__ == "__main__":
    resultados_dir = os.path.join(os.path.dirname(__file__), "resultados")
    unificar_csvs(resultados_dir)

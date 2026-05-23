import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# --- PASO 2: Convertir a formato multi-etiqueta ---
print("Iniciando Paso 2...")
# Lee el archivo original que está dentro de tu carpeta data
df_raw = pd.read_csv('data/train.csv')

# Separar el nombre de la imagen y la etiqueta (Fish, Flower, Gravel, Sugar)
df_raw['image_id'] = df_raw['Image_Label'].apply(lambda x: x.split('_')[0])
df_raw['label'] = df_raw['Image_Label'].apply(lambda x: x.split('_')[1])

# Marcar con 1 si hay máscara (si la columna 'EncodedPixels' tiene datos) y 0 si no
df_raw['has_mask'] = df_raw['EncodedPixels'].notnull().astype(int)

# Pivotear la tabla para que queden las columnas: image_id, Fish, Flower, Gravel, Sugar
df = df_raw.pivot(index='image_id', columns='label', values='has_mask').reset_index()
df.columns.name = None

print("Paso 2 completado. Así quedó el Dataset:")
print(df.head())

# Guardar el dataset completo multi-etiqueta
df.to_csv('data/train_multilabel.csv', index=False)
print("Archivo 'data/train_multilabel.csv' guardado.")

# --- PASO 3: Split estratificado 70-15-15 ---
print("\nIniciando Paso 3 (Splits estratificados)...")
X = df[['image_id']].values
y = df[['Fish', 'Flower', 'Gravel', 'Sugar']].values

# Primer corte: Separar 70% para Train y 30% Temporal (que luego será Val + Test)
msss_train = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in msss_train.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

# Segundo corte: Separar ese 30% temporal en dos mitades iguales (15% Validación y 15% Test)
msss_val_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in msss_val_test.split(X_temp, y_temp):
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

# Función rápida para armar los nuevos CSVs y guardarlos
def guardar_csv(X_arr, y_arr, nombre_archivo):
    df_split = pd.DataFrame(X_arr, columns=['image_id'])
    df_split[['Fish', 'Flower', 'Gravel', 'Sugar']] = y_arr
    df_split.to_csv(f'data/{nombre_archivo}', index=False)

# Guardar los 3 archivos finales en tu carpeta data
guardar_csv(X_train, y_train, 'train_split.csv')
guardar_csv(X_val, y_val, 'val_split.csv')
guardar_csv(X_test, y_test, 'test_split.csv')

print(f"Paso 3 completado. Archivos guardados correctamente en la carpeta 'data'.")
print(f"Cantidades -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
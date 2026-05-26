import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Paso 2 — Convertir las anotaciones RLE a formato multi-etiqueta binario.
print("Iniciando Paso 2...")
df_raw = pd.read_csv('data/train.csv')

df_raw['image_id'] = df_raw['Image_Label'].apply(lambda x: x.split('_')[0])
df_raw['label'] = df_raw['Image_Label'].apply(lambda x: x.split('_')[1])

# 1 si hay máscara, 0 si no.
df_raw['has_mask'] = df_raw['EncodedPixels'].notnull().astype(int)

df = df_raw.pivot(index='image_id', columns='label', values='has_mask').reset_index()
df.columns.name = None

print("Paso 2 completado. Así quedó el Dataset:")
print(df.head())

df.to_csv('data/train_multilabel.csv', index=False)
print("Archivo 'data/train_multilabel.csv' guardado.")

# Paso 3 — Partición estratificada multi-etiqueta 70 / 15 / 15.
print("\nIniciando Paso 3 (Splits estratificados)...")
X = df[['image_id']].values
y = df[['Fish', 'Flower', 'Gravel', 'Sugar']].values

# Primer corte: 70% train vs 30% temporal.
msss_train = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in msss_train.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

# Segundo corte: el 30% temporal se parte 50/50 en validación y test (15% + 15%).
msss_val_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in msss_val_test.split(X_temp, y_temp):
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]


def guardar_csv(X_arr, y_arr, nombre_archivo):
    df_split = pd.DataFrame(X_arr, columns=['image_id'])
    df_split[['Fish', 'Flower', 'Gravel', 'Sugar']] = y_arr
    df_split.to_csv(f'data/{nombre_archivo}', index=False)


guardar_csv(X_train, y_train, 'train_split.csv')
guardar_csv(X_val, y_val, 'val_split.csv')
guardar_csv(X_test, y_test, 'test_split.csv')

print(f"Paso 3 completado. Archivos guardados correctamente en la carpeta 'data'.")
print(f"Cantidades -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

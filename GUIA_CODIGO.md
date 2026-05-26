# Guía rápida del código

> Esta guía explica **qué hace cada archivo** y **dónde ocurre lo importante**.
> Todos los scripts se ejecutan desde la **raíz del proyecto** (`Proyecto_Nubes/`),
> nunca desde dentro de `src/` o `app/`.

---

## Estructura del proyecto

```
Proyecto_Nubes/
├── data/                       ← dataset (imágenes + CSVs)
├── models/                     ← checkpoints .pth entrenados
├── results/                    ← métricas, figuras, predicciones, historial
├── src/                        ← código del experimento (puntos A–E)
└── app/                        ← demo interactiva Streamlit (punto F)
```

---

## `src/` — pipeline experimental

Orden de ejecución (de cero a paper):

```
preprocesamiento  →  train  →  train_final  →  evaluate  →  analyze_errors
                                     │
                                     ↓
                            plot_training_curves
```

### `src/preprocesamiento.py`
**Punto A.** Convierte las anotaciones RLE del dataset original a formato multi-etiqueta y arma la **partición estratificada 70 / 15 / 15**.

- Lee `data/train.csv` (anotaciones RLE originales).
- Pivotea a tabla por imagen con columnas `Fish, Flower, Gravel, Sugar` (1/0).
- **Split 70/15/15** con `MultilabelStratifiedShuffleSplit` (preserva la distribución conjunta de las 4 etiquetas).
- Genera: `data/train_split.csv`, `data/val_split.csv`, `data/test_split.csv`.

### `src/dataset.py`
**Preprocesamiento de imágenes + aumento de datos** (Albumentations).

- `CloudDataset`: clase `torch.utils.data.Dataset`. Lee la imagen, aplica transforms y devuelve `(tensor, label)`.
- `get_train_transforms()`: resize a **320×480** + flips + rotaciones ±15° + brillo/contraste + **normalización ImageNet**.
- `get_valid_transforms()`: resize + normalización (sin augmentations).
- Constantes exportadas: `IMG_HEIGHT=320`, `IMG_WIDTH=480`, `IMAGENET_MEAN`, `IMAGENET_STD`.

### `src/model.py`
**Definición de las arquitecturas** vía `timm`.

- `CloudModel`: wrapper genérico que devuelve **logits** (no aplica sigmoid).
- `get_densenet121(pretrained=True)`: DenseNet-121 con fine-tuning.
- `get_mobilenetv3_large(pretrained=True)`: MobileNetV3-Large con fine-tuning.

### `src/train.py`
**Punto B.** Validación cruzada estratificada de 2 folds para elegir entre las dos arquitecturas.

- Carga `data/train_split.csv` (solo el 70% de train; **val/test no se tocan**).
- 2 folds con `MultilabelStratifiedKFold`.
- Entrena cada arquitectura 5 epochs por fold con AMP.
- **Métrica reportada**: F1 macro = media ± desviación estándar.
- Logs en W&B (offline).
- Guarda los mejores checkpoints por fold en `models/best_model_<arch>_fold<n>.pth`.

### `src/train_final.py`
**Punto C – entrenamiento.** Entrena el modelo ganador (DenseNet-121) sobre todo el conjunto de train, validando contra val.

- 25 epochs máximo + **early stopping** (paciencia 5) sobre F1 macro de val.
- LR scheduler: `CosineAnnealingLR` (1e-4 → 1e-6).
- Pérdida: `BCEWithLogitsLoss`. Optimizador: Adam. AMP activado.
- Guarda: `models/best_model_final_DenseNet121.pth` y `results/training_history_final.csv` (historial epoch a epoch).

### `src/evaluate.py`
**Punto C – evaluación sobre val.** Mide el modelo final en el conjunto de validación.

- Carga el mejor checkpoint y corre inferencia con umbral 0.5.
- **Métricas por clase**: accuracy, precision, recall, F1, matriz de confusión 2×2 (TN/FP/FN/TP) + fila MACRO.
- Salidas en `results/`: `metrics_per_class.csv`, `metrics_summary.md`, `confusion_matrices.png`, `predictions_val.csv`.

### `src/analyze_errors.py`
**Punto D.** Evaluación final sobre **test** + análisis cualitativo de errores.

- Métricas finales sobre `test_split.csv` (idénticas a `evaluate.py` pero sobre la partición de prueba).
- **Selección automática** de 5 aciertos diversos (alta confianza, cubren todas las clases positivas) y 5 errores "interesantes" (alta confianza en la dirección equivocada).
- Salidas en `results/`: `metrics_test_per_class.csv`, `metrics_test_summary.md`, `confusion_matrices_test.png`, `predictions_test.csv`, `aciertos_errores.png` (grid 2×5), `analisis_aciertos_errores.md`.

### `src/plot_training_curves.py`
**Figura del paper.** Genera las curvas de entrenamiento a partir de `results/training_history_final.csv`.

- Subplot (a): pérdida train vs val por epoch.
- Subplot (b): F1 macro de val con punto rojo en la mejor epoch.
- Salidas: `results/training_curves.pdf` (vectorial para LNCS) + `.png` 300 dpi.

### `src/visualizar_filtros.py`
**Material didáctico.** Visualiza qué aprende internamente la red.

- `filtros_primera_capa.png`: los **64 filtros 7×7×3** aprendidos en la primera capa convolucional.
- `feature_maps.png`: 8 mapas de activación en 4 profundidades distintas (`conv0`, `denseblock1`, `denseblock3`, `denseblock4`), mostrando cómo la red pasa de bordes a conceptos abstractos.

### `src/inspeccionar_modelo.py`
**Diagnóstico.** Cuenta capas y parámetros de DenseNet-121: Conv2d, Linear, BatchNorm y total de parámetros. Útil para responder "¿cuántas capas tiene la red?" en la presentación.

---

## `app/` — demo interactiva Streamlit (punto F)

Arquitectura limpia: el **dominio** (`core/`) no conoce Streamlit; la **UI** sí.

```
app/
├── streamlit_app.py    ← entry point (orquesta)
├── config.py           ← constantes (modelo, umbral, dimensiones)
├── core/               ← lógica pura, sin Streamlit
│   ├── model_loader.py     carga el .pth
│   ├── preprocessor.py     PIL Image → tensor
│   └── predictor.py        tensor → Prediction
└── ui/
    └── views.py        ← componentes Streamlit reutilizables
```

### `app/config.py`
Todas las constantes centralizadas: `MODEL_PATH`, `CLASS_NAMES`, `DEFAULT_THRESHOLD=0.5`, dimensiones de imagen, estadísticas de ImageNet, textos de UI. **PROJECT_ROOT** se calcula relativo al archivo → la app es portable.

### `app/core/model_loader.py`
Carga `best_model_final_DenseNet121.pth`. Importa la arquitectura desde `src/model.py` (DRY: misma definición que en entrenamiento).

### `app/core/preprocessor.py`
`ImagePreprocessor`: convierte una PIL Image al tensor que espera el modelo. **Idéntico** al pipeline de validación de entrenamiento.

### `app/core/predictor.py`
- `CloudPredictor`: corre el modelo y aplica sigmoid para devolver probabilidades en `[0, 1]`.
- `Prediction` (dataclass inmutable): guarda solo las probabilidades; el umbral se aplica vía `.labels_at(threshold)`. Esto permite que mover el slider del umbral en la UI **no re-ejecute el modelo**.

### `app/streamlit_app.py`
Entry point. Orquesta el flujo (sidebar → uploader → preprocess → predict → render).

- `@st.cache_resource` → modelo y preprocessor cargan una sola vez.
- `@st.cache_data` por bytes de imagen → la predicción se cachea; mover el umbral es instantáneo.

### `app/ui/views.py`
Componentes Streamlit: header, sidebar (slider del umbral), uploader, render de imagen y de predicción (etiquetas detectadas + barra por clase).

---

## Cómo correr todo

Desde **la raíz del proyecto** (`Proyecto_Nubes/`):

```powershell
# Pipeline experimental
python src/preprocesamiento.py
python src/train.py
python src/train_final.py
python src/evaluate.py
python src/analyze_errors.py
python src/plot_training_curves.py

# Material didáctico
python src/inspeccionar_modelo.py
python src/visualizar_filtros.py

# Demo interactiva
python -m streamlit run app/streamlit_app.py
```

---

## Resumen "¿dónde está X?"

| Pregunta | Archivo |
|---|---|
| ¿Dónde se parte el dataset 70/15/15? | `src/preprocesamiento.py` |
| ¿Dónde se preprocesan las imágenes (resize, normalize)? | `src/dataset.py` (`get_train_transforms`, `get_valid_transforms`) |
| ¿Dónde se hace data augmentation? | `src/dataset.py` (`get_train_transforms`) |
| ¿Dónde se compara DenseNet vs MobileNet (CV)? | `src/train.py` |
| ¿Dónde se entrena el modelo final? | `src/train_final.py` |
| ¿Dónde se calculan las métricas en validación? | `src/evaluate.py` |
| ¿Dónde se calculan las métricas finales en test? | `src/analyze_errors.py` |
| ¿Dónde se eligen los aciertos/errores del paper? | `src/analyze_errors.py` (`seleccionar_ejemplos`) |
| ¿Dónde se generan las curvas de entrenamiento? | `src/plot_training_curves.py` |
| ¿Dónde está la lógica de inferencia de la demo? | `app/core/predictor.py` |
| ¿Dónde está el umbral configurable? | `app/config.py` (default) + `app/ui/views.py` (slider) |

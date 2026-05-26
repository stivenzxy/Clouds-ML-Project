# Notas del Proyecto — Clasificación de Nubes

Documento de referencia con todas las decisiones técnicas, conceptos y justificaciones del proyecto. Pensado para consulta durante el desarrollo y para argumentar en el informe final.

---

## 📋 Estado de avance del proyecto

| Punto | Estado | Observaciones |
|-------|--------|---------------|
| A. Multi-etiqueta + split 70/15/15 | ✅ Completo | OK con `MultilabelStratifiedShuffleSplit` |
| B. CV 2 folds + 2 arquitecturas + W&B | ⚠️ En proceso | Falta corregir CSV usado y optimizar performance |
| C. Modelo final + métricas en val | 🔲 Pendiente | |
| D. 5 aciertos + 5 errores en test | 🔲 Pendiente | |
| E. Artículo LNCS (4-8 páginas) | 🔲 Pendiente | |
| F. Interfaz (web/móvil/escritorio) | 🔲 Pendiente | |

---

## 🧠 Conceptos clave (explicados en simple)

### Multi-etiqueta vs. multi-clase

- **Multi-clase**: cada imagen pertenece a UNA sola categoría (perro O gato).
- **Multi-etiqueta**: cada imagen puede pertenecer a VARIAS categorías al mismo tiempo (Fish + Gravel).

En este proyecto las nubes son **multi-etiqueta**: una imagen puede tener Fish, Flower, Gravel y Sugar simultáneamente.

### ¿Qué hace `preprocesamiento.py`?

El CSV original de Kaggle tiene **cada imagen repetida 4 veces** (una fila por tipo de nube) con la máscara de segmentación (`EncodedPixels`).

El script:
1. **Reorganiza** la tabla a formato `image_id | Fish | Flower | Gravel | Sugar` con 1/0 según si esa nube aparece (mirando si hay máscara o no).
2. **Divide** las imágenes en 70% train / 15% val / 15% test usando split estratificado.

### ¿Por qué `MultilabelStratifiedShuffleSplit`?

**Estratificar** = respetar las proporciones de cada clase al dividir el dataset.

Sin estratificación, podés tener mala suerte y que el test quede con 90% Fish y 5% Flower → las métricas mienten.

La versión "Multilabel" es necesaria porque el `StratifiedKFold` normal de sklearn **no sabe trabajar con varias etiquetas por muestra**. La librería `iterstrat` resuelve el split respetando proporciones de TODAS las clases al mismo tiempo.

### ¿Qué es BCE (BCEWithLogitsLoss)?

**Binary Cross Entropy** = función de pérdida que mide qué tan equivocado está el modelo.

Por cada imagen, el modelo responde 4 preguntas independientes sí/no:
- ¿Hay Fish? ¿Hay Flower? ¿Hay Gravel? ¿Hay Sugar?

BCE mide qué tan bien respondió cada pregunta y promedia. Si dice "95% seguro hay Fish" y SÍ había → pérdida chiquita. Si NO había → pérdida grande.

**`WithLogits`** significa que la función internamente aplica sigmoid + BCE en un solo paso. **Es más estable numéricamente.** Por eso el modelo devuelve **logits crudos** (sin sigmoid).

### ¿Qué es Albumentations?

Librería de **aumento de datos** (data augmentation) para imágenes. Genera versiones modificadas al vuelo (rotadas, espejadas, con cambios de brillo) para que el modelo "vea" más variedad sin tener más datos reales.

**Importante**: el aumento de datos SOLO se aplica en **train**, nunca en validación ni test (en esos solo Resize + Normalize).

### ¿Qué es `num_classes=4`?

Al crear el modelo con `timm.create_model(..., num_classes=4)`, le decimos que la última capa tenga 4 salidas (una por tipo de nube). `timm` reemplaza automáticamente la capa final original (1000 clases de ImageNet) por una nueva de 4 salidas.

### ¿Por qué `pretrained=True`?

**Transfer learning**: los modelos vienen preentrenados en ImageNet (millones de imágenes). Ya saben detectar bordes, texturas y formas generales. Solo hay que enseñarles los patrones específicos de nubes.

Sin `pretrained=True` arrancarían desde cero y necesitarías cientos de miles de imágenes + días de entrenamiento para resultados decentes. **Con `pretrained=True` es casi obligatorio en proyectos así.**

### ¿Qué son los batches?

En vez de mostrarle al modelo las imágenes una por una (lento, ruidoso) o todas juntas (no entra en memoria), se le muestran en **grupitos (batches)**.

Con `batch_size=16` y 3900 imágenes en train:
- Por cada epoch hace 3900/16 ≈ **244 ajustes** del modelo.
- Cada ajuste se basa en el error promedio de 16 imágenes.

**Tradeoff**:
- Batch chico (8, 16): menos VRAM, gradientes ruidosos (a veces ayuda a no estancarse), más lento por epoch.
- Batch grande (32, 64, 128): más VRAM, gradientes estables, más rápido por epoch. Si te pasás → `CUDA out of memory`.

### ¿Por qué se baja la resolución? (1400×2100 → 320×480)

**4 razones técnicas:**

1. **VRAM**: a resolución original, DenseNet121 batch 16 necesita ~40-60 GB. La 5060 tiene 8 GB. Literalmente no entra.
2. **Velocidad**: las convoluciones cuestan proporcional al número de píxeles. 320×480 tiene ~19x menos píxeles que 1400×2100 → entrenamiento ~15-20x más rápido.
3. **Preentrenamiento**: DenseNet, MobileNet, etc. fueron entrenados en ImageNet a **224×224**. Sus filtros están optimizados para ese rango. A resoluciones gigantes el transfer learning funciona PEOR.
4. **Clasificación no necesita detalles finos**: los patrones Fish/Flower/Gravel/Sugar son estructuras grandes (cientos de píxeles). Se reconocen perfectamente a 320×480. Si fuera segmentación sí necesitarías alta resolución.

**¿Por qué exactamente 320×480?**
- Mantiene la relación de aspecto original (2:3) → no distorsiona los patrones.
- Divisible por 32 (las CNN hacen downsampling de a 2, 5 veces).
- Está en el rango usado por los top solutions de la competencia original Kaggle "Understanding Clouds" (256×384 a 384×576).

---

## ⚠️ Errores identificados y correcciones pendientes

### 1. ERROR CRÍTICO: data leakage en la CV

En `train.py`:
```python
full_df = pd.read_csv('data/train_multilabel.csv')  # ❌ MAL
```

Esto usa el 100% del dataset para CV → contamina val y test.

**Corrección:**
```python
full_df = pd.read_csv('data/train_split.csv')  # ✅ solo el 70%
```

> El enunciado dice: "validación cruzada estratificada de 2 particiones **sobre el conjunto de entrenamiento**".

Los `.pth` ya generados están contaminados → hay que reentrenar después de corregir.

### 2. Código zombi en `model.py`

```python
self.activation = nn.Sigmoid()  # ❌ declarado pero NUNCA usado
```

Borrar. Está bien que el modelo devuelva logits para `BCEWithLogitsLoss`.

### 3. Orden de augmentations ineficiente

Actual:
```python
A.HorizontalFlip, A.VerticalFlip, A.Rotate(limit=30),
A.RandomBrightnessContrast, A.Resize(320, 480), ...
```

Están rotando imágenes de 1400×2100 ANTES de redimensionar → trabajan sobre ~3 millones de píxeles cuando podrían trabajar sobre ~150 mil.

**Corrección**: poner `A.Resize` **PRIMERO**.

### 4. Sin semilla global

Sin `torch.manual_seed`, `np.random.seed`, `random.seed` no hay reproducibilidad → comparación DenseNet vs MobileNet no es justa.

---

## 🚀 Optimización de performance (de 5 horas a ~45 min)

### Diagnóstico del cuello de botella

- **Antes**: 27 min/epoch para ~3900 imágenes batch 16 = **6.6 s/batch**
- **Esperado en RTX 5060 DenseNet121 320×480**: 0.15–0.25 s/batch
- **Diferencia**: 25-40x más lento de lo que debería

**No es la GPU**, es cómo se le sirven los datos.

> ⚠️ Que `nvidia-smi` reporte 99% GPU NO significa que esté trabajando todo el tiempo. NVIDIA reporta "% del tiempo con al menos un kernel corriendo", no qué tanto trabajo real hace.

### Los 6 culpables (ordenados por impacto)

#### 1. `num_workers=0` ← EL ASESINO PRINCIPAL

Con 0 workers, el proceso principal lee imagen → decodifica JPEG → augmenta → entrena, todo **secuencial**. La GPU espera al CPU constantemente.

#### 2. `OMP_NUM_THREADS=1`

Limita OpenMP (que usan cv2, NumPy, Albumentations) a un solo hilo. **Sabotaje.** Borrar.

#### 3. Sin `pin_memory=True`

Transferencia CPU→GPU más lenta (memoria pageada vs no pageada).

#### 4. Sin Mixed Precision (AMP)

PyTorch con `torch.cuda.amp.autocast()` usa float16 donde puede. **~1.8x-2x más rápido**, mitad de VRAM, sin pérdida de calidad.

#### 5. Sin `non_blocking=True`

Detalle menor: permite que las transferencias se solapen con cómputo.

#### 6. Orden de augmentations (ya mencionado arriba)

### ¿Cuántos `num_workers`?

**No es "ponele todos los threads que tengas"**.

Cada worker es un **proceso aparte** (no thread) que lee + decodifica + augmenta.

Depende de:
1. **Velocidad GPU vs preparar un batch**: si el batch se prepara más rápido de lo que la GPU consume, ya no necesitás más workers.
2. **RAM**: cada worker carga el dataset en su propia memoria. En Windows con `spawn` esto es caro.
3. **Overhead de Windows**: cada worker se crea con `spawn` (proceso nuevo desde cero), no con `fork`. Tarda segundos en arrancar → `persistent_workers=True` es CLAVE.
4. **Más workers de los necesarios** → context switching → empeora performance.

**Regla práctica Windows**: `num_workers ≈ min(4-6, cores_físicos / 2)`

---

## ⚙️ Configuración recomendada final

### Hardware del equipo
- CPU: Ryzen 7 5800XT (8 cores físicos / 16 threads)
- RAM: 32 GB
- GPU: RTX 5060 (8 GB VRAM)
- SO: Windows

### Configuración DataLoader

```python
train_loader = DataLoader(
    train_ds,
    batch_size=32,              # subir desde 16. Si truena → bajar a 24 o 16.
    shuffle=True,
    num_workers=6,              # 8 cores físicos - 2 para sistema/main = 6
    pin_memory=True,
    persistent_workers=True,    # CRÍTICO en Windows
    prefetch_factor=2,
)

val_loader = DataLoader(
    val_ds,
    batch_size=64,              # val no hace backward → entra más
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
```

**Razones:**
- `num_workers=6`: aprovecha el CPU sin saturarlo, deja 2 cores para el proceso principal + sistema.
- `batch_size=32` train: probable que entre en 8 GB con AMP. Si OOM → bajar.
- `batch_size=64` val: en validación no hay backward → entra más imagen en memoria.
- `persistent_workers=True`: evita recrear los 6 workers entre epochs (cada uno tarda segundos en arrancar con `spawn`).

### Variables de entorno

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# QUITAR: os.environ['OMP_NUM_THREADS'] = '1'   ← BORRAR
```

### Estimación de mejora total

| Cambio | Speedup |
|--------|---------|
| `num_workers=6` + `persistent_workers=True` | 3x – 6x |
| Quitar `OMP_NUM_THREADS=1` | 1.5x – 2x |
| Resize primero en augmentations | 1.3x – 1.5x |
| `pin_memory=True` + `non_blocking=True` | 1.05x – 1.1x |
| Mixed Precision (AMP) | 1.5x – 2x |

**Total CV (2 modelos × 2 folds × 5 epochs):**
- Antes: ~5 horas
- Después: **~45 minutos**

### Aviso importante

Los primeros 10-30 segundos al iniciar van a parecer "colgados" → los 6 workers se están creando, importando librerías. **Normal en Windows con `spawn`.** No matar el proceso.

---

## 📝 Argumentos para el informe (listos para usar)

### Sobre el redimensionado a 320×480

> *"Las imágenes originales del dataset poseen una resolución de 1400×2100 píxeles, lo que resulta computacionalmente prohibitivo para entrenar redes neuronales convolucionales profundas en el hardware disponible (RTX 5060, 8 GB VRAM). Se aplicó un redimensionado a 320×480 píxeles, manteniendo la relación de aspecto original (2:3) para no distorsionar los patrones morfológicos de las nubes. Esta resolución cumple tres requisitos: (i) preserva las estructuras macroscópicas que caracterizan a los cuatro patrones (Fish, Flower, Gravel, Sugar), las cuales se manifiestan a escala de cientos de píxeles; (ii) es compatible con los factores de submuestreo de DenseNet-121 y MobileNetV3 (divisible por 32); y (iii) se aproxima al rango de resoluciones utilizado durante el preentrenamiento de las arquitecturas en ImageNet (224×224), maximizando el beneficio del transfer learning."*

### Sobre el tradeoff del redimensionado

> *"El redimensionado implica una pérdida de información de alta frecuencia (detalles finos de textura). Sin embargo, en clasificación de patrones meteorológicos a gran escala, dichos detalles son secundarios frente a las estructuras espaciales globales, por lo que el compromiso resulta favorable."*

### Sobre el split estratificado multi-etiqueta

> *"Se utilizó `MultilabelStratifiedShuffleSplit` de la librería `iterstrat` en lugar del estratificado tradicional de scikit-learn, ya que este último no contempla escenarios donde una muestra pertenece simultáneamente a múltiples clases. Esto garantiza que la distribución de presencia/ausencia de cada uno de los cuatro patrones de nube se preserve aproximadamente entre los conjuntos de entrenamiento (70%), validación (15%) y prueba (15%), evitando sesgos de evaluación."*

### Sobre BCEWithLogitsLoss

> *"Se utilizó `BCEWithLogitsLoss` (Binary Cross Entropy con logits) como función de pérdida, dado que el problema se modela como cuatro clasificaciones binarias independientes (una por patrón de nube). Esta función combina internamente la activación sigmoide con la entropía cruzada binaria, ofreciendo mayor estabilidad numérica que aplicar ambas operaciones por separado."*

### Sobre transfer learning

> *"Ambas arquitecturas (DenseNet-121 y MobileNetV3-Large) se inicializaron con pesos preentrenados en ImageNet a través de la librería `timm`. Esta estrategia de transfer learning permite aprovechar representaciones visuales generales (bordes, texturas, patrones jerárquicos) aprendidas sobre más de un millón de imágenes, reduciendo significativamente el tiempo de entrenamiento y mejorando el rendimiento en el dominio específico de nubes satelitales, donde el conjunto de datos es limitado."*

---

## 🔜 Pendientes inmediatos

1. ✅ Crear este documento de notas.
2. ✅ Aplicar correcciones al código:
   - ✅ Cambiar `train_multilabel.csv` → `train_split.csv` en CV.
   - ✅ Borrar `self.activation` zombie en `model.py`.
   - ✅ Optimizar DataLoader (workers, pin_memory, persistent_workers).
   - ✅ Agregar Mixed Precision (AMP).
   - ✅ Reordenar augmentations (Resize primero).
   - ✅ Agregar semillas globales.
   - ✅ Quitar `OMP_NUM_THREADS=1`.
3. ✅ Reentrenar CV con código corregido.
4. ✅ Punto B completo (resultados abajo).
5. 🔲 Punto C: modelo final + métricas en val + matriz de confusión.
6. 🔲 Punto D: análisis cualitativo de 5 aciertos + 5 errores.
7. 🔲 Punto E: artículo LNCS.
8. 🔲 Punto F: interfaz.

---

## 📊 Resultados del Punto B

Validación cruzada estratificada multi-etiqueta de 2 folds sobre `train_split.csv` (70% del dataset).

| Arquitectura | F1 Macro (media ± std) | Fold 1 | Fold 2 |
|--------------|------------------------|--------|--------|
| MobileNetV3    | 0.7476 ± 0.0046 | 0.7430 | 0.7521 |
| **DenseNet121** | **0.7542 ± 0.0040** | 0.7582 | 0.7502 |

**Conclusión**: DenseNet121 gana por ~0.7 puntos de F1 macro y con menor desviación estándar entre folds (más estable). Ambas arquitecturas están en el rango competitivo del problema (~0.65–0.75 F1 macro según la competencia original de Kaggle).

**Mejor arquitectura → DenseNet121** será la usada en el modelo final del punto C.

CSV exportado: `wandb_export_2026-05-23T22_11_28.862-05_00.csv`

### Sobre los runs de W&B

Los runs se guardaron en modo **offline** en `wandb/offline-run-*/`. Para subirlos al dashboard de wandb.ai y poder mostrar capturas de las curvas en el paper:

```powershell
wandb sync wandb/offline-run-*
```

Esto sube los logs de cada fold (train_loss, val_loss, f1_macro, f1_weighted, lr, epoch_time) y genera un link público al proyecto.

---

## ⚠️ Aclaración importante: dos cosas que se llaman "val"

Confusión común con la nomenclatura. Son DOS cosas distintas:

### 1. `val_split.csv` (en disco, generado en preprocesamiento)
- Es el **15%** del dataset separado en el punto A.
- **Está intocado.** No se cargó nunca durante la CV del punto B.
- Reservado para el **punto C** (evaluar el modelo final).

### 2. `val_loss` que aparece en W&B durante la CV
- Es OTRA cosa. Es la pérdida sobre la **mitad del propio `train_split.csv`** que se usa como validación dentro de cada fold de la CV.
- En `MultilabelStratifiedKFold(n_splits=2)` el 70% se parte en 2 mitades:
  - Fold 1: mitad A entrena → mitad B valida
  - Fold 2: mitad B entrena → mitad A valida
- Esa "val" cambia en cada fold y es interna a la CV.

**Resumen de los 3 splits:**

| CSV | % dataset | ¿Usado en B? | Reservado para |
|-----|-----------|--------------|----------------|
| `train_split.csv` | 70% | ✅ Sí (CV completa) | Modelo final (C) |
| `val_split.csv` | 15% | ❌ Intocado | Evaluación del modelo final (C) |
| `test_split.csv` | 15% | ❌ Intocado | 5 aciertos + 5 errores (D) |

---

## 🎯 Punto C — Plan y decisiones

### Objetivo
Entrenar UN modelo final de DenseNet121 sobre todo `train_split.csv`, evaluarlo en `val_split.csv` (intocado), y reportar **por cada clase de nube**:

- **Exactitud por etiqueta** (Accuracy binaria por clase)
- **Precisión** (Precision)
- **Exhaustividad** (Recall)
- **F1-score**
- **Matriz de confusión** (una 2×2 por cada clase)

### Decisiones de diseño

#### Arquitectura: DenseNet121
Ganador de la CV con F1 macro = 0.7542 ± 0.0040, mejor que MobileNetV3 (0.7476 ± 0.0046) tanto en media como en estabilidad.

#### Número de epochs: 25 (con early stopping)
En la CV del punto B se usaron 5 epochs y el modelo seguía aprendiendo (train y val loss bajando, F1 subiendo). Para el modelo final tiene sentido entrenar más tiempo, ya que:
- No es un experimento comparativo (no necesitamos repetir 4 veces).
- Buscamos el mejor F1 posible.
- Tenemos cómputo de sobra (con AMP cada epoch tarda ~30s).

**Early stopping** con `patience=5`: si la métrica de validación (F1 macro) no mejora durante 5 epochs seguidas, corta el entrenamiento. Evita overfitting y desperdicio de cómputo.

#### Scheduler: CosineAnnealingLR
Reduce el learning rate de forma suave siguiendo una curva coseno desde 1e-4 hasta ~1e-6 a lo largo de los 25 epochs.

**Por qué `CosineAnnealing` y no `ReduceLROnPlateau`:**
- No tiene hiperparámetros que ajustar (factor, patience). Plug & play.
- La reducción gradual permite "afinar" el modelo al final del entrenamiento (lr bajo = pasos más pequeños = ajustes más finos).
- Empíricamente da mejores resultados que LR fijo en transfer learning sobre datasets medianos.

#### Conjuntos
- **Entrenamiento**: `train_split.csv` completo (70%, ~3856 imágenes).
- **Validación**: `val_split.csv` completo (15%, ~826 imágenes), evaluado al final de cada epoch.

#### Modelo guardado
Se guarda el checkpoint del epoch con mejor **F1 macro en validación**, no del último epoch.

### Estimación de tiempo
- 25 epochs × ~30s = ~12-15 minutos para el entrenamiento.
- Evaluación final + matrices de confusión: ~30 segundos.
- **Total Punto C: ~15-20 minutos**.

### Argumentos para el informe

> *"Tras la validación cruzada del punto B, DenseNet-121 se seleccionó como arquitectura final por presentar el mayor F1-score macro promedio (0.7542 ± 0.0040) y la menor variabilidad entre folds, lo que indica mayor estabilidad. El modelo final se entrenó durante 25 épocas sobre el conjunto de entrenamiento completo, con un planificador de tasa de aprendizaje CosineAnnealingLR que reduce el LR de 1×10⁻⁴ a aproximadamente 1×10⁻⁶ siguiendo una curva coseno. Se aplicó early stopping (paciencia = 5) sobre el F1-score macro de validación para evitar sobreajuste. El checkpoint del modelo correspondiente a la mejor métrica de validación fue retenido para la evaluación final."*

### Métricas a reportar (definiciones para el paper)

Para cada clase k ∈ {Fish, Flower, Gravel, Sugar}:

- **Exactitud (Accuracy)**: (TP + TN) / (TP + TN + FP + FN)
- **Precisión (Precision)**: TP / (TP + FP) — de los que predije positivos, cuántos lo son.
- **Exhaustividad (Recall)**: TP / (TP + FN) — de los positivos reales, cuántos atrapé.
- **F1-score**: 2 · (Precision · Recall) / (Precision + Recall) — promedio armónico.
- **Matriz de confusión 2×2**: [[TN, FP], [FN, TP]].

---

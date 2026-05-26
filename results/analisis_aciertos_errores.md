# Análisis cualitativo — aciertos y errores sobre test_split

## Aciertos seleccionados

| # | image_id | Reales | Predichas | Probabilidades |
|---|----------|--------|-----------|----------------|
| 1 | 25459bd.jpg | Gravel, Sugar | Gravel, Sugar | Fish=0.11 | Flower=0.04 | Gravel=0.99 | Sugar=0.98 |
| 2 | bb09fd9.jpg | Flower, Sugar | Flower, Sugar | Fish=0.12 | Flower=0.99 | Gravel=0.13 | Sugar=0.99 |
| 3 | fa53fe0.jpg | Fish, Flower | Fish, Flower | Fish=0.97 | Flower=0.96 | Gravel=0.10 | Sugar=0.19 |
| 4 | 3faf64c.jpg | Gravel, Sugar | Gravel, Sugar | Fish=0.10 | Flower=0.04 | Gravel=0.96 | Sugar=0.96 |
| 5 | d90e620.jpg | Gravel, Sugar | Gravel, Sugar | Fish=0.13 | Flower=0.05 | Gravel=0.98 | Sugar=0.98 |

## Errores seleccionados

| # | image_id | Reales | Predichas | Probabilidades | Tipo de error |
|---|----------|--------|-----------|----------------|---------------|
| 1 | 5a18d32.jpg | Gravel, Sugar | Fish | Fish=1.00 | Flower=0.32 | Gravel=0.20 | Sugar=0.27 | FP Fish; FN Gravel; FN Sugar |
| 2 | fd5b3ab.jpg | Fish, Flower, Gravel | Flower, Sugar | Fish=0.34 | Flower=0.98 | Gravel=0.13 | Sugar=0.95 | FN Fish; FN Gravel; FP Sugar |
| 3 | 0ad97bb.jpg | Gravel | Fish, Sugar | Fish=0.88 | Flower=0.05 | Gravel=0.35 | Sugar=0.94 | FP Fish; FN Gravel; FP Sugar |
| 4 | 993e07b.jpg | Fish | Flower, Sugar | Fish=0.27 | Flower=0.71 | Gravel=0.24 | Sugar=0.97 | FN Fish; FP Flower; FP Sugar |
| 5 | 6e00609.jpg | Fish, Gravel | Fish, Flower, Sugar | Fish=0.66 | Flower=0.99 | Gravel=0.19 | Sugar=0.61 | FP Flower; FN Gravel; FP Sugar |


## Cómo usar esta tabla para el paper

Para cada imagen de error, abrí la figura `aciertos_errores.png` y mirala. Después redactá 1-2 frases que expliquen por qué creés que el modelo se equivocó, relacionando lo que ves visualmente con las características esperadas del patrón (ej. *Fish* tiene filamentos alargados, *Sugar* es disperso y homogéneo, *Flower* son células redondeadas, *Gravel* son agregaciones granulares).

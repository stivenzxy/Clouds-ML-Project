from pathlib import Path

# Raíz del proyecto, relativa a este archivo (hace la app portable).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modelo
MODEL_PATH = PROJECT_ROOT / 'models' / 'best_model_final_DenseNet121.pth'

# Imagen de ejemplo precargada (multi-etiqueta: Fish + Gravel + Sugar).
DEMO_IMAGE_PATH = PROJECT_ROOT / 'app' / 'assets' / 'demo_example.jpg'
CLASS_NAMES = ('Fish', 'Flower', 'Gravel', 'Sugar')
DEFAULT_THRESHOLD = 0.5

# Preprocesamiento (debe coincidir con el entrenamiento).
IMG_HEIGHT = 320
IMG_WIDTH = 480
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# UI
PAGE_TITLE = 'Clasificador de patrones de nubes'
APP_DESCRIPTION = (
    'Cargá una imagen satelital y el modelo predirá qué patrones de nubes '
    'están presentes. Los cuatro patrones (Rasp et al., 2020) corresponden a '
    'distintas formas de organización mesoescalar de la convección somera '
    'sobre el océano y se asocian a regímenes meteorológicos diferentes.'
)

# Información por patrón: descripción visual + contexto meteorológico (Rasp et al., 2020).
# Cada patrón tiene un color asociado para reforzar la identificación visual.
PATTERN_INFO = {
    'Fish': {
        'emoji': '🐟',
        'color': '#1f77b4',          # azul
        'visual': 'Estructuras alargadas, esqueléticas, con ramificaciones tipo espina de pescado.',
        'meteo': (
            'Frecuente cerca de zonas de convergencia mesoescalar y bordes de '
            'cold pools. Asocia masas de aire relativamente húmedas y vientos '
            'organizados; suele preceder a sistemas convectivos más intensos.'
        ),
    },
    'Flower': {
        'emoji': '🌸',
        'color': '#d62728',          # rojo/rosa
        'visual': 'Células redondeadas, agrupadas como pétalos, con espacios despejados entre ellas.',
        'meteo': (
            'Típico de regímenes de estratocúmulos en transición. Indica '
            'convección organizada con circulaciones de celdas cerradas y '
            'subsidencia entre células; condiciones generalmente estables.'
        ),
    },
    'Gravel': {
        'emoji': '🪨',
        'color': '#7f7f7f',          # gris
        'visual': 'Agregaciones granulares pequeñas, textura tipo guijarros distribuidos.',
        'meteo': (
            'Asocia convección somera fragmentada, frecuentemente sobre líneas '
            'de convergencia superficial. Suele aparecer en aire marino con '
            'humedad intermedia y vientos moderados.'
        ),
    },
    'Sugar': {
        'emoji': '✨',
        'color': '#ff7f0e',          # naranja
        'visual': 'Dispersión fina y homogénea de pequeños cúmulos, sin estructura organizada visible.',
        'meteo': (
            'Cumulus someros muy dispersos. Asocia vientos débiles, estabilidad '
            'fuerte en la capa subnube y cielos en general tranquilos, sin '
            'precipitación significativa.'
        ),
    },
}

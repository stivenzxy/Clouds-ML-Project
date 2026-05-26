import sys
from pathlib import Path

# Permite `from app import ...` cuando Streamlit lanza este archivo directamente.
# Debe ir ANTES de los imports del proyecto.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import io

import streamlit as st
import torch
from PIL import Image

from app import config
from app.core.model_loader import load_densenet121
from app.core.predictor import CloudPredictor, Prediction
from app.core.preprocessor import ImagePreprocessor
from app.ui import views


# cache_resource: el modelo se carga UNA sola vez por sesión de Streamlit.
@st.cache_resource(show_spinner='Cargando modelo...')
def _build_predictor() -> CloudPredictor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_densenet121(config.MODEL_PATH, device)
    return CloudPredictor(
        model=model,
        class_names=config.CLASS_NAMES,
        device=device,
    )


@st.cache_resource
def _build_preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor(
        height=config.IMG_HEIGHT,
        width=config.IMG_WIDTH,
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD,
    )


# cache_data por bytes de imagen: mover el slider del umbral NO re-corre el modelo.
@st.cache_data(show_spinner='Analizando imagen...')
def _predict_from_bytes(image_bytes: bytes) -> Prediction:
    image = Image.open(io.BytesIO(image_bytes))
    tensor = _build_preprocessor().to_tensor(image)
    return _build_predictor().predict(tensor)


def main() -> None:
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        layout='wide',
    )

    threshold = views.render_sidebar(config.DEFAULT_THRESHOLD)
    views.render_header(config.PAGE_TITLE, config.APP_DESCRIPTION)

    image_bytes = views.render_uploader()
    if image_bytes is None:
        st.info('Esperando una imagen… o probá con el ejemplo precargado.')
        return

    image = Image.open(io.BytesIO(image_bytes))

    col_img, col_pred = st.columns(2)
    with col_img:
        views.render_image(image)

    prediction = _predict_from_bytes(image_bytes)

    with col_pred:
        views.render_prediction(prediction, threshold=threshold)


if __name__ == '__main__':
    main()

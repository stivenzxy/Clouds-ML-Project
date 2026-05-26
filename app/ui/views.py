import streamlit as st
from PIL import Image

from app import config
from app.core.predictor import Prediction


def render_header(title: str, description: str) -> None:
    st.title(title)
    st.markdown(description)


def render_sidebar(default_threshold: float) -> float:
    with st.sidebar:
        st.header('Configuración')
        threshold = st.slider(
            'Umbral de decisión',
            min_value=0.0,
            max_value=1.0,
            value=default_threshold,
            step=0.05,
            help=(
                'Probabilidad mínima para considerar un patrón como presente. '
                'Mover este umbral NO re-ejecuta el modelo: las probabilidades '
                'son las mismas, cambia solo qué etiquetas se muestran como '
                'detectadas.'
            ),
        )
        st.markdown('---')
        st.caption(
            'Modelo: DenseNet-121 con *fine-tuning* sobre el dataset '
            '*Understanding Clouds from Satellite Images* (Rasp et al., 2020).'
        )

        with st.expander('ℹ️ Los cuatro patrones'):
            for name, info in config.PATTERN_INFO.items():
                st.markdown(
                    f"<span style='color:{info['color']};font-weight:bold'>"
                    f"{info['emoji']} {name}</span> — {info['visual']}",
                    unsafe_allow_html=True,
                )
    return threshold


def render_uploader() -> bytes | None:
    """Devuelve los bytes de la imagen a procesar.

    Soporta dos fuentes:
    - Archivo subido por el usuario.
    - Imagen de ejemplo precargada (botón en sidebar).

    Usa session_state para que el ejemplo persista entre re-renders
    (mover el slider del umbral no debe perderlo).
    """
    col_upload, col_demo = st.columns([3, 1])

    with col_upload:
        uploaded = st.file_uploader(
            'Sube una imagen satelital',
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=False,
            label_visibility='collapsed',
        )

    with col_demo:
        use_demo = st.button(
            '🎲 Probar con ejemplo',
            help='Carga una imagen del conjunto de test con varios patrones presentes.',
            width='stretch',
        )

    # Si el usuario sube un archivo nuevo, ese gana y limpia el demo.
    if uploaded is not None:
        st.session_state.pop('demo_bytes', None)
        return uploaded.getvalue()

    # Si pidió el demo (o ya estaba cargado), devolverlo.
    if use_demo or 'demo_bytes' in st.session_state:
        if 'demo_bytes' not in st.session_state:
            with open(config.DEMO_IMAGE_PATH, 'rb') as f:
                st.session_state['demo_bytes'] = f.read()
        return st.session_state['demo_bytes']

    return None


def render_image(image: Image.Image) -> None:
    st.image(image, caption='Imagen cargada', width='stretch')


def _render_probability_card(name: str, prob: float, threshold: float) -> None:
    """Tarjeta visual de probabilidad por clase.

    No fija color de texto: hereda del tema (funciona en modo claro y oscuro).
    El énfasis viene del borde, el badge y la barra de progreso de color.
    """
    info = config.PATTERN_INFO[name]
    color = info['color']
    detected = prob >= threshold

    if detected:
        # Fondo translúcido del color (visible en claro y oscuro).
        bg = f'{color}26'              # ~15% opacidad
        border = color
        border_width = '2px'
        badge = (
            f"<span style='background:{color};color:white;padding:2px 10px;"
            f"border-radius:10px;font-size:0.75rem;font-weight:bold;'>"
            f"DETECTADO</span>"
        )
    else:
        bg = 'transparent'
        border = color
        border_width = '1px'
        badge = (
            f"<span style='border:1px solid {color}66;color:{color};opacity:0.7;"
            f"padding:2px 10px;border-radius:10px;font-size:0.75rem;'>"
            f"no detectado</span>"
        )

    pct = int(round(prob * 100))
    # Barra: track translúcido (funciona en cualquier tema) + relleno coloreado.
    bar = (
        f"<div style='background:rgba(128,128,128,0.25);border-radius:6px;"
        f"height:8px;overflow:hidden;margin-top:8px;'>"
        f"<div style='background:{color};width:{pct}%;height:100%;'></div>"
        f"</div>"
    )

    card = f"""
    <div style='
        background:{bg};
        border:{border_width} solid {border};
        border-radius:10px;
        padding:12px 14px;
        margin-bottom:10px;
    '>
        <div style='display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-size:1.1rem;font-weight:bold;color:{color};'>
                {info['emoji']} {name}
            </span>
            {badge}
        </div>
        <div style='font-size:1.5rem;font-weight:bold;margin-top:4px;'>
            {prob:.3f}
        </div>
        {bar}
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)


def _render_meteo_panel(detected: tuple) -> None:
    """Panel descriptivo: contexto meteorológico de los patrones detectados."""
    st.markdown('### Interpretación meteorológica')
    if not detected:
        st.info(
            'No se detectó ningún patrón claro por encima del umbral. '
            'Prueba bajar el umbral en la barra lateral si deseas ver la '
            'inclinación del modelo.'
        )
        return

    for name in detected:
        info = config.PATTERN_INFO[name]
        st.markdown(
            f"""
            <div style='
                border-left:4px solid {info['color']};
                background:{info['color']}1A;
                padding:10px 14px;
                margin-bottom:8px;
                border-radius:0 8px 8px 0;
            '>
                <div style='font-weight:bold;color:{info['color']};font-size:1.05rem;'>
                    {info['emoji']} {name}
                </div>
                <div style='font-size:0.92rem;margin-top:6px;'>
                    <b>Forma visual:</b> {info['visual']}
                </div>
                <div style='font-size:0.92rem;margin-top:4px;'>
                    <b>Contexto meteorológico:</b> {info['meteo']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction(prediction: Prediction, threshold: float) -> None:
    detected = prediction.labels_at(threshold)

    st.subheader('Patrones detectados')
    if detected:
        chips = ' '.join(
            f"<span style='background:{config.PATTERN_INFO[p]['color']};"
            f"color:white;padding:4px 10px;border-radius:12px;"
            f"font-weight:bold;margin-right:6px;'>"
            f"{config.PATTERN_INFO[p]['emoji']} {p}</span>"
            for p in detected
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        top_name, top_prob = prediction.top_class()
        st.warning(
            f'Ningún patrón superó el umbral de {threshold:.2f}. '
            f'La clase más probable es **{top_name}** ({top_prob:.3f}).'
        )

    st.markdown('### Probabilidades por clase')
    items = list(prediction.probabilities.items())
    col_a, col_b = st.columns(2)
    for i, (name, prob) in enumerate(items):
        target = col_a if i % 2 == 0 else col_b
        with target:
            _render_probability_card(name, prob, threshold)

    _render_meteo_panel(detected)

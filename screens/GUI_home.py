import streamlit as st
from screens.aux_functions import load_css, load_image

def home_screen():
    load_css('style.css')

    # Sidebar for navigation
    st.sidebar.title("Navegación")
    screen = st.sidebar.radio("Selecciona una pantalla:", ['Inicio', 'Analizar Texto', 'Analizar Video de YouTube'])

    # Logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    image = load_image('logo.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Introducción
    st.markdown('<p class="medium-font">Bienvenido a NO HATE ZONE, nuestro innovador servicio de detección de discurso de odio. Utilizamos inteligencia artificial de vanguardia para identificar y analizar contenido potencialmente dañino.</p>', unsafe_allow_html=True)

    # Servicios
    st.markdown('## Nuestros Servicios', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="service-box">
        <h3>✍️ Análisis de Texto</h3>
        <p>Utilizando BERT, analizamos texto para detectar discurso de odio y contenido potencialmente dañino.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="service-box">
        <h3>🎥 Análisis de Comentarios de YouTube</h3>
        <p>Analizamos los comentarios de videos de YouTube para identificar patrones de discurso de odio y contenido problemático.</p>
        </div>
        """, unsafe_allow_html=True)

    # Llamada a la acción
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Analizar Texto'):
            st.session_state.screen = 'predict'
    with col2:
        if st.button('Analizar comentarios de YouTube'):
            st.session_state.screen = 'link_prediction'

    # Información adicional
    st.markdown('### Sobre el Discurso de Odio', unsafe_allow_html=True)
    st.markdown("""
    El discurso de odio es una forma de expresión que promueve la discriminación, hostilidad o violencia 
    hacia individuos o grupos basándose en características como raza, etnia, género, orientación sexual, 
    religión u otras características. La detección temprana y la prevención son cruciales para crear 
    espacios en línea más seguros y respetuosos. NO HATE ZONE está diseñado para ayudar a identificar 
    y comprender este tipo de contenido utilizando la más avanzada tecnología de inteligencia artificial.
    """)

    # Características principales
    st.markdown('### Características Principales', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h4>🎯 Análisis Preciso</h4>
        <p>Algoritmos entrenados con grandes conjuntos de datos para máxima precisión.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h4>⚡ Resultados Instantáneos</h4>
        <p>Obtenga análisis en tiempo real de texto y comentarios.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pie de página
    st.markdown('---')
    st.markdown('© 2024 No Hate Zone - Tecnología contra el discurso de odio. Todos los derechos reservados.')

if __name__ == "__main__":
    home_screen()
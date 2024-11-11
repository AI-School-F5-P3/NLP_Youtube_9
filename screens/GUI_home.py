import streamlit as st
from screens.aux_functions import load_css, load_image

def home_screen():
    load_css('style.css')

    # Logo
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    image = load_image('logo.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Introducción
    st.markdown('<p class="medium-font">Welcome to NO HATE ZONE, our innovative hate speech detection service. We use cutting-edge artificial intelligence to identify and analyze potentially harmful content.</p>', unsafe_allow_html=True)

    # Servicios
    st.markdown('## Our Services', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="service-box">
        <h3>✍️ Text Analysis</h3>
        <p>Using BERT, we analyze text to detect hate speech and potentially harmful content.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="service-box">
        <h3>🎥 YouTube Comment Analysis</h3>
        <p>We analyze YouTube video comments to identify patterns of hate speech and problematic content.</p>
        </div>
        """, unsafe_allow_html=True)

    # Llamada a la acción
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Analyze Text'):
            st.session_state.screen = 'predict'
    with col2:
        if st.button('Analyze YouTube Comments'):
            st.session_state.screen = 'link_prediction'

    st.markdown('### About Hate Speech', unsafe_allow_html=True)
    st.markdown("""
    Hate speech is a form of expression that promotes discrimination, hostility, or violence 
    towards individuals or groups based on characteristics such as race, ethnicity, gender, sexual orientation, 
    religion, or other traits. Early detection and prevention are crucial in creating 
    safer and more respectful online spaces. NO HATE ZONE is designed to help identify 
    and understand this type of content using the most advanced artificial intelligence technology.
    """)

    # Key Features
    st.markdown('### Key Features', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h4>🎯 Accurate Analysis</h4>
        <p>Algorithms trained on large datasets for maximum accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h4>⚡ Instant Results</h4>
        <p>Get real-time analysis of text and comments.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('---')
    st.markdown('© 2024 No Hate Zone - Technology against hate speech. All rights reserved.')

if __name__ == "__main__":
    home_screen()
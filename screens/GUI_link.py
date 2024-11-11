import streamlit as st
from screens.aux_functions import load_css, predict_youtube_comments, load_image

def link_screen():
    load_css('style.css')
    
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    image = load_image('logo.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Header
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.markdown('## Predicción de Comentarios de YouTube', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Link input
    st.markdown('### Ingrese el enlace del video de YouTube', unsafe_allow_html=True)
    youtube_link = st.text_input("")
    
    # Prediction button
    if st.button('Analizar Comentarios'):
        if youtube_link:
            try:
                # Get predictions for comments
                predictions = predict_youtube_comments(youtube_link)
                
                # Display results
                st.markdown('### Resultados del Análisis', unsafe_allow_html=True)
                st.markdown("""
                <div class="result-box">
                    <h4>Análisis de Comentarios:</h4>
                """, unsafe_allow_html=True)
                
                for comment, prediction in predictions.items():
                    st.markdown(f"""
                    <div class="comment-box">
                        <p><strong>Comentario:</strong> {comment}</p>
                        <p><strong>Predicción:</strong> {prediction}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error('Error al procesar el enlace. Asegúrese de que sea un enlace válido de YouTube.')
        else:
            st.warning('Por favor, ingrese un enlace de YouTube.')
    
    # Return to home button
    if st.button('Volver al Inicio'):
        st.session_state.screen = 'home'

if __name__ == "__main__":
    link_screen()
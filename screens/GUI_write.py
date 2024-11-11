import streamlit as st
from screens.aux_functions import load_css, predict_text

def write_screen():
    load_css('style.css')
    
    # Header
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.markdown('## Predicción basada en Texto', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Text input area
    st.markdown('### Ingrese su mensaje', unsafe_allow_html=True)
    user_text = st.text_area("", height=150)
    
    # Prediction button
    if st.button('Realizar Predicción'):
        if user_text:
            # Get prediction from model
            prediction = predict_text(user_text)
            
            # Display results
            st.markdown('### Resultados', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-box">
                <h4>Predicción:</h4>
                <p>{prediction}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning('Por favor, ingrese un texto para realizar la predicción.')
    
    # Return to home button
    if st.button('Volver al Inicio'):
        st.session_state.screen = 'home'

if __name__ == "__main__":
    write_screen()
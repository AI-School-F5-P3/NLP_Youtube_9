import streamlit as st
from screens.aux_functions import load_css, predict_youtube_comments, load_image

def link_screen():
    load_css('style.css')
    
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    image = load_image('logo.png')
    st.image(image, width=150)
    st.markdown('</div>', unsafe_allow_html=True)

    # Header
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.markdown('## YouTube Comment Prediction', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Link input
    st.markdown('### Enter the YouTube video link', unsafe_allow_html=True)
    youtube_link = st.text_input("")
    
    # Prediction button
    if st.button('Analyze Comments'):
        if youtube_link:
            try:
                # Get predictions for comments
                predictions = predict_youtube_comments(youtube_link)

            except Exception as e:
                st.error('Error processing the link. Please ensure it is a valid YouTube link.')
        else:
            st.warning('Please enter a YouTube link.')
    
    # Return to home button
    if st.button('Back to Home'):
        st.session_state.screen = 'home'

if __name__ == "__main__":
    link_screen()
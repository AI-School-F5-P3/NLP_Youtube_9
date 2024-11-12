import streamlit as st
from screens.GUI_home import home_screen
from screens.GUI_write import write_screen
from screens.GUI_link import link_screen
from screens.aux_functions import load_css
from firebase_utils import FirebaseManager  # Import the FirebaseManager class
        
st.set_page_config(
    page_title="No Hate Zone",
    page_icon="✋",
    layout="wide"
)

# Initialize Firebase manager in session state if not already present
if 'firebase_manager' not in st.session_state:
    try:
        st.session_state.firebase_manager = FirebaseManager()
        st.success("Firebase connection established!")
    except Exception as e:
        st.error(f"Error connecting to Firebase: {str(e)}")

load_css('style.css')

# Set initial screen state if not already set
if 'screen' not in st.session_state:
    st.session_state.screen = 'home'

# Sidebar for navigation
st.sidebar.markdown('<span class="sidebar-header">Menu</span>', unsafe_allow_html=True)
if st.sidebar.button("Home", key="home_button"):
    st.session_state.screen = 'home'
if st.sidebar.button("Text-based prediction", key="predict_button"):
    st.session_state.screen = 'predict'
if st.sidebar.button("YouTube Comment Prediction", key="link_prediction_button"):
    st.session_state.screen = 'link_prediction'

# Display appropriate screen based on session state
def main():
    if st.session_state.screen == 'home':
        home_screen()
    elif st.session_state.screen == 'predict':
        write_screen()
    elif st.session_state.screen == 'link_prediction':
        link_screen()

if __name__ == "__main__":
    main()
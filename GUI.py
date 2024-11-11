import streamlit as st
from screens.GUI_home import home_screen
from screens.GUI_write import write_screen
from screens.GUI_link import link_screen

st.set_page_config(
    page_title="No Hate Zone",
    page_icon="✋",
    layout="wide"
)

if 'screen' not in st.session_state:
    st.session_state.screen = 'home'
    
def main():
    # Display appropriate screen based on session state
    if st.session_state.screen == 'home':
        home_screen()
    elif st.session_state.screen == 'predict':
        write_screen()
    elif st.session_state.screen == 'link_prediction':
        link_screen()

if __name__ == "__main__":
    main()
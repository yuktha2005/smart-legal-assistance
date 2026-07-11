import streamlit as st

def init_session():
    """Initializes session state variables if they don't exist."""
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None

def login_user(user_id: int, username: str):
    """Sets session state for a logged-in user."""
    st.session_state['logged_in'] = True
    st.session_state['user_id'] = user_id
    st.session_state['username'] = username

def logout_user():
    """Clears session state to log out a user."""
    st.session_state['logged_in'] = False
    st.session_state['user_id'] = None
    st.session_state['username'] = None
    
def is_authenticated() -> bool:
    """Checks if the user is authenticated."""
    return st.session_state.get('logged_in', False)

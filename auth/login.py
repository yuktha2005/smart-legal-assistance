import streamlit as st
from database.connection import get_db
from database.models import User
from auth.security import verify_password
from auth.session_manager import login_user

def render_login():
    """Renders the login form."""
    st.title("Login to Smart Legal Assistance")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not username or not password:
                st.error("Please provide both username and password.")
            else:
                db_generator = get_db()
                db = next(db_generator)
                try:
                    user = db.query(User).filter(User.username == username).first()
                    if user and verify_password(password, user.password_hash):
                        login_user(user.id, user.username)
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                finally:
                    db_generator.close()

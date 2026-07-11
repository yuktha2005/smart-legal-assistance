import streamlit as st
from database.connection import get_db
from database.models import User
from auth.security import get_password_hash

def render_register():
    """Renders the user registration form."""
    st.title("Register New Account")

    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Register")

        if submit_button:
            if not username or not email or not password:
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                db_generator = get_db()
                db = next(db_generator)
                try:
                    # Check if user exists
                    existing_user = db.query(User).filter((User.username == username) | (User.email == email)).first()
                    if existing_user:
                        st.error("Username or email already exists.")
                    else:
                        hashed_password = get_password_hash(password)
                        new_user = User(username=username, email=email, password_hash=hashed_password)
                        db.add(new_user)
                        db.commit()
                        st.success("Registration successful! You can now log in.")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error creating account: {e}")
                finally:
                    db_generator.close()

import streamlit as st
from database.connection import get_db
from database.models import Feedback

def render_feedback():
    st.title("User Feedback")
    st.write("Please rate your experience with the Smart Legal Assistance system.")
    
    with st.form("feedback_form"):
        rating = st.slider("Rating (1-5)", min_value=1, max_value=5, value=5)
        comments = st.text_area("Additional Comments (Optional)")
        
        submit = st.form_submit_button("Submit Feedback")
        
        if submit:
            user_id = st.session_state.get('user_id')
            if not user_id:
                st.error("You must be logged in to submit feedback.")
            else:
                db_generator = get_db()
                db = next(db_generator)
                try:
                    fb = Feedback(user_id=user_id, rating=rating, comments=comments)
                    db.add(fb)
                    db.commit()
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving feedback: {e}")
                finally:
                    db_generator.close()

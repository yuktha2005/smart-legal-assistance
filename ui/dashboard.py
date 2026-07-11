import streamlit as st
import pandas as pd
import altair as alt
from database.connection import get_db
from database.models import User, Document, ChatHistory, Feedback

def render_dashboard():
    st.title("Admin Dashboard")
    
    db_generator = get_db()
    db = next(db_generator)
    
    try:
        # Metrics
        total_users = db.query(User).count()
        total_docs = db.query(Document).count()
        total_queries = db.query(ChatHistory).count()
        
        # Calculate avg feedback if there are any
        feedbacks = db.query(Feedback.rating).all()
        avg_rating = sum(f.rating for f in feedbacks) / len(feedbacks) if feedbacks else 0.0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Users", total_users)
        col2.metric("Documents Uploaded", total_docs)
        col3.metric("Questions Answered", total_queries)
        col4.metric("Avg Feedback Rating", f"{avg_rating:.1f} / 5.0")
        
        st.subheader("Recent Activity")
        # Placeholder for more complex analytics or charts.
        # For example, count of documents uploaded over time.
        # We can construct a simple dataframe here.
        docs = db.query(Document.upload_date).all()
        if docs:
            df_docs = pd.DataFrame([{"date": d.upload_date.date()} for d in docs])
            df_grouped = df_docs.groupby('date').size().reset_index(name='count')
            
            chart = alt.Chart(df_grouped).mark_bar(color='#4c78a8').encode(
                x='date:T',
                y='count:Q',
                tooltip=['date', 'count']
            ).properties(title="Documents Uploaded Over Time")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No document upload data to display yet.")
            
    finally:
        db_generator.close()

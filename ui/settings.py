import streamlit as st
from database.connection import get_db
from database.models import ChatHistory
from src.tts import TTSManager

def render_settings():
    st.title("Settings")
    
    # Initialize TTS Manager in session state if not already done
    if "tts_manager" not in st.session_state:
        st.session_state.tts_manager = TTSManager()
        
    st.subheader("Text-to-Speech (TTS) Settings")
    tts_manager = st.session_state.tts_manager
    cfg = tts_manager.config
    
    enable_tts = st.toggle("Enable Text-to-Speech (TTS)", value=cfg.get("enable_tts", True))
    voice_model = st.text_input("Voice Model ONNX Path", value=cfg.get("voice_model", "piper/models/en_US-lessac-medium.onnx"))
    speaking_rate = st.slider("Speaking Rate (Speed)", min_value=0.5, max_value=2.0, value=float(cfg.get("speaking_rate", 1.0)), step=0.1)
    auto_play = st.toggle("Auto-play newly generated responses", value=cfg.get("auto_play", False))
    read_sources = st.toggle("Read sources & citation details", value=cfg.get("read_sources", False))
    
    if st.button("Save TTS Settings"):
        cfg["enable_tts"] = enable_tts
        cfg["voice_model"] = voice_model
        cfg["speaking_rate"] = speaking_rate
        cfg["auto_play"] = auto_play
        cfg["read_sources"] = read_sources
        
        if tts_manager.save_config():
            st.success("TTS settings updated and saved to config/tts_config.json successfully!")
        else:
            st.error("Failed to save TTS settings to file.")
            
    st.divider()
    
    st.subheader("Appearance")
    st.write("Theme selection is handled automatically by Streamlit based on your system preferences. You can force Light/Dark mode via the top-right settings menu in Streamlit.")
    
    st.subheader("Data Management")
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.success("Chat memory cleared from current session.")
        
    if st.button("Delete All Chat History (Database)"):
        user_id = st.session_state.get('user_id')
        if user_id:
            db_generator = get_db()
            db = next(db_generator)
            try:
                db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()
                db.commit()
                st.session_state.messages = []
                st.success("All chat history permanently deleted.")
            except Exception as e:
                db.rollback()
                st.error(f"Failed to delete history: {e}")
            finally:
                db_generator.close()
        else:
            st.error("You must be logged in.")
            
    st.subheader("Export")
    if st.button("Export Conversations (JSON)"):
        user_id = st.session_state.get('user_id')
        if user_id:
            db_generator = get_db()
            db = next(db_generator)
            try:
                history = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).all()
                export_data = [{"Q": h.question, "A": h.answer, "Time": str(h.timestamp)} for h in history]
                st.json(export_data)
            finally:
                db_generator.close()
        else:
            st.error("You must be logged in.")

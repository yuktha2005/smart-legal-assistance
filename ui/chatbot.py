import streamlit as st
import os
import logging
from database.connection import get_db
from database.models import ChatHistory
from src.tts import TTSManager

def render_chatbot(chatbot=None):
    st.title("Legal Assistant Chat")
    
    if not chatbot:
        st.error("Chatbot pipeline is not initialized.")
        return
        
    # Initialize TTS Manager in session state
    if "tts_manager" not in st.session_state:
        st.session_state.tts_manager = TTSManager()
    
    # Initialize active audio session variables
    if "current_audio_text" not in st.session_state:
        st.session_state.current_audio_text = None
    if "current_audio_idx" not in st.session_state:
        st.session_state.current_audio_idx = None

    # Initialize chat memory in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Split layout into chat column and advocate robot visualizer column on the right
    chat_col, robot_col = st.columns([3, 1])

    with chat_col:
        # Display chat messages from history on app rerun
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    tts_cfg = st.session_state.tts_manager.config
                    if tts_cfg.get("enable_tts", True):
                        if st.button("🔊 Listen", key=f"listen_{idx}"):
                            st.session_state.current_audio_text = message["content"]
                            st.session_state.current_audio_idx = idx
                            st.rerun()

        # React to user input
        if prompt := st.chat_input("Ask a legal question... (e.g., 'What is section 376DA?')"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate response using EnhancedChatbot
            with st.spinner("Analyzing documents..."):
                try:
                    # Reconstruct session history for the chatbot
                    session_history = [{"question": m["content"]} for m in st.session_state.messages if m["role"] == "user"]
                    # Exclude the current prompt from history since we pass it separately
                    if session_history:
                        session_history.pop()
                        
                    response_data = chatbot.chat(prompt, session_history=session_history, last_sources=st.session_state.get("last_sources", []))
                    
                    answer = response_data.get('answer', 'Error generating answer.')
                    sources = response_data.get('sources', [])
                    
                    # Save sources invisibly for follow-up source requests
                    if sources:
                        st.session_state.last_sources = sources
                        
                    response_md = answer
                except Exception as e:
                    import logging
                    logger = logging.getLogger("rag_engine")
                    logger.exception(e)
                    response_md = f"Error: {str(e)}"
                    
                if response_md is None or response_md.strip() == "":
                    response_md = "I couldn't find sufficient information in the uploaded legal documents."

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response_md)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_md})
            
            # If auto_play is enabled, trigger audio player automatically
            new_idx = len(st.session_state.messages) - 1
            if st.session_state.tts_manager.config.get("auto_play", False):
                st.session_state.current_audio_text = response_md
                st.session_state.current_audio_idx = new_idx

            # Save to database
            if st.session_state.get('user_id'):
                db_generator = get_db()
                db = next(db_generator)
                try:
                    chat_log = ChatHistory(
                        user_id=st.session_state['user_id'],
                        question=prompt,
                        answer=response_md
                    )
                    db.add(chat_log)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    st.error(f"Failed to log chat: {e}")
                finally:
                    db_generator.close()

            # Trigger redraw of Streamlit component to show audio player
            st.rerun()

    with robot_col:
        # Load base64 images for HTML embedding
        import base64
        
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        idle_path = os.path.join(repo_root, "assets", "advocate_robot_idle.png")
        speak_path = os.path.join(repo_root, "assets", "advocate_robot_speaking.png")
        
        def get_b64_img(p):
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            return ""
            
        idle_b64 = get_b64_img(idle_path)
        speak_b64 = get_b64_img(speak_path)
        
        audio_b64 = ""
        if st.session_state.current_audio_text:
            try:
                audio_path = st.session_state.tts_manager.generate_speech(st.session_state.current_audio_text)
                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                logging.getLogger("tts_engine").error(f"Offline speech generation failed: {e}")
        
        # Build custom HTML block
        html_code = f"""
        <div id="container" style="text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0e1117; padding: 20px; border-radius: 15px; border: 1px solid #30363d; box-shadow: 0 4px 15px rgba(0,0,0,0.5); max-width: 250px; margin: 0 auto;">
            <div style="margin-bottom: 15px; font-size: 14px; font-weight: bold; color: #58a6ff; letter-spacing: 0.5px;">ADVOCATE CO-PILOT</div>
            <div style="position: relative; display: inline-block;">
                <img id="robot-img" src="data:image/png;base64,{idle_b64}" width="200" style="border-radius: 15px; border: 2px solid #30363d; transition: all 0.3s ease-in-out;" />
                <div id="soundwave" style="display: none; position: absolute; bottom: 15px; left: 50%; transform: translateX(-50%); text-align: center; background: rgba(14, 17, 23, 0.85); padding: 5px 15px; border-radius: 10px; border: 1px solid #30363d;">
                    <span class="bar" style="display: inline-block; width: 4px; height: 10px; background: #58a6ff; margin: 0 2px; animation: bounce 0.6s infinite alternate;"></span>
                    <span class="bar" style="display: inline-block; width: 4px; height: 20px; background: #58a6ff; margin: 0 2px; animation: bounce 0.6s infinite alternate; animation-delay: 0.2s;"></span>
                    <span class="bar" style="display: inline-block; width: 4px; height: 15px; background: #58a6ff; margin: 0 2px; animation: bounce 0.6s infinite alternate; animation-delay: 0.1s;"></span>
                    <span class="bar" style="display: inline-block; width: 4px; height: 25px; background: #58a6ff; margin: 0 2px; animation: bounce 0.6s infinite alternate; animation-delay: 0.3s;"></span>
                </div>
            </div>
            <div id="status" style="margin-top: 15px; font-size: 13px; font-weight: bold; color: #8b949e; transition: color 0.3s;">Status: Idle</div>
            
            <style>
            @keyframes bounce {{
                0% {{ transform: scaleY(0.3); }}
                100% {{ transform: scaleY(1.1); }}
            }}
            .speaking-glow {{
                box-shadow: 0 0 20px rgba(88, 166, 255, 0.6) !important;
                border-color: #58a6ff !important;
            }}
            </style>
        """

        if audio_b64:
            html_code += f"""
            <audio id="audio-player" src="data:audio/wav;base64,{audio_b64}" autoplay controls style="margin-top: 15px; width: 100%; display: block; border-radius: 5px; background-color: #161b22;"></audio>
            <script>
                const audio = document.getElementById("audio-player");
                const img = document.getElementById("robot-img");
                const status = document.getElementById("status");
                const wave = document.getElementById("soundwave");
                
                let idleSrc = "data:image/png;base64,{idle_b64}";
                let speakSrc = "data:image/png;base64,{speak_b64}";
                let speakInterval;
                let speakToggle = false;
                
                audio.onplay = () => {{
                    status.innerText = "Status: Speaking...";
                    status.style.color = "#58a6ff";
                    img.classList.add("speaking-glow");
                    wave.style.display = "block";
                    
                    // Lip-syncing simulation: alternate between idle (closed mouth) and speaking (open mouth) images
                    clearInterval(speakInterval);
                    speakInterval = setInterval(() => {{
                        speakToggle = !speakToggle;
                        img.src = speakToggle ? speakSrc : idleSrc;
                        img.style.transform = speakToggle ? "scale(1.04)" : "scale(1.0)";
                    }}, 150);
                }};
                
                audio.onended = () => {{
                    clearInterval(speakInterval);
                    img.src = idleSrc;
                    img.style.transform = "scale(1.0)";
                    img.classList.remove("speaking-glow");
                    status.innerText = "Status: Idle";
                    status.style.color = "#8b949e";
                    wave.style.display = "none";
                }};
                
                audio.onpause = () => {{
                    clearInterval(speakInterval);
                    img.src = idleSrc;
                    img.style.transform = "scale(1.0)";
                    img.classList.remove("speaking-glow");
                    status.innerText = "Status: Idle";
                    status.style.color = "#8b949e";
                    wave.style.display = "none";
                }};
            </script>
            """
        else:
            html_code += """
            <div style="margin-top: 15px; font-size: 12px; color: #8b949e; font-style: italic;">Select "Listen" on any message to speak.</div>
            """
            
        html_code += "</div>"
        
        import streamlit.components.v1 as components
        components.html(html_code, height=480)

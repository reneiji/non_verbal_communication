import streamlit as st


import os
import plotly.express as px
import pandas as pd
import subprocess
import sys

from face_model.face_model import analyze_video
from body_language_mod.body_model import init_model as init_body_model, analyze_video as analyze_body_video
from speech_model.model import calculate_confidence_score, extract_audio_features, extract_visual_features_from_video, extract_audio_from_video

print(">>> DEBUG: PORT from env =", os.getenv("PORT"))

# Set page config
st.set_page_config(
    page_title="Nonverbal Communication Analyzer",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
    <style>
    .big-title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
    color: #333;
    }
    .sub-title {
        font-size: 22px;
        text-align: center;
        color: #555;
        margin-bottom: 40px;
    }

    /* ===== GLOBAL FONT ===== */
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
    }

    /* ===== FILE UPLOADER ===== */
    div.stFileUploader label div {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #d62828 !important;  /* deep red */
        padding-bottom: 10px;
        max-width: 400px !important;
    }

    /* ===== SECTION TITLES ===== */
    h2 {
        padding-left: 450px !important;
    }
    
    h3 {
        font-size: 26px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* ===== CUSTOM COLORS ===== */
    /* Speech Score */
    .speech-score h3 {
        color: #F25C05;  /* orange */
    }

    /* Face Score */
    .face-score h3 {
        color: #0077b6;  /* blue */
    }

    /* Body Score */
    .body-score h3 {
        color: #2a9d8f;  /* green */
    }
    
    div.stButton > button {
        padding-top: 18px !important;
        padding-bottom: 18px !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        max-width: 250px !important;
        margin: 20px auto !important;
        background-color: #1f77b4 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: background-color 0.2s, transform 0.1s;
    }

    div.stButton > button:hover {
        background-color: #155a8a !important;
        transform: translateY(-2px);
    }
    
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        max-width: 600px;  /* control width */
        margin: 0 auto;  /* center horizontally */
        padding-top: 30px;
        padding-bottom: 30px;
    }

    .button-row {
        display: flex;
        justify-content: center;  /* center horizontally */
        align-items: center;      /* center vertically if needed */
        gap: 40px;                /* space between buttons */
        margin-bottom: 40px;
        margin-top: 20px;
        flex-wrap: wrap;          /* wrap on small screens */
}
    
        div[data-testid="stFileUploader"] {
            width: 100% !important;
            max-width: 750px !important;
            margin: 0 auto !important;
            padding-left: 70px !important;
            padding-right: 10px !important;
        }

        div[data-testid="stFileUploader"] label div {
            font-size: 24px !important;
            font-weight: bold !important;
            color: #d62828 !important;
            margin-bottom: 10px !important;
        }


    </style>

""", unsafe_allow_html=True)

# Streamlit frontend starts here:
st.title("🗣️ Communication Analyzer")
st.caption("Analyze **speech**, **facial expressions**, and **body language** to assess presentation confidence 🚀")
st.markdown("---")

# Centered container
st.markdown('<div class="centered-container">', unsafe_allow_html=True)

# st.markdown("""
# <div class="input-mode-header">
#     <h2>Select Input Mode:</h2>
# </div>
# """, unsafe_allow_html=True)


# Initialize session state for input mode
if "input_mode" not in st.session_state:
    st.session_state.input_mode = None

# Initialize session state for webcam
if "webcam_launched" not in st.session_state:
    st.session_state.webcam_launched = False

# Path to venv (pyenv safe)
venv_activate = os.path.join(os.path.dirname(sys.executable), "activate")

# Project path and face_model.py path
project_path = os.path.abspath(".")
face_model_script = os.path.join(project_path, "face_model/face_model.py")

# Build command
command = f'source {venv_activate} && python {face_model_script}'

# Build AppleScript command
osa_command = f'''osascript -e 'tell application "Terminal" to do script "cd {project_path} && {command}"' '''

# # Two columns for two buttons
# col1, col2 = st.columns(2)

col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])

# with col1:
#     st.empty()  # spacer

with col3:
    if st.button("📂 Upload Video", use_container_width=True):
        st.session_state.input_mode = "upload"

with col4:
    if st.button("🎥 Launch Webcam", use_container_width=True):
        st.info("Launching webcam in new Terminal window... Close webcam window to return to app.")
        subprocess.Popen(osa_command, shell=True)

# with col4:
#     st.empty()  # spacer
    
# --- UPLOAD VIDEO FLOW ---
if st.session_state.input_mode == "upload":
    video_file = st.file_uploader("Choose a video...", type=["mp4"])

    if video_file:
        # Save the uploaded video to disk
        video_path = f"uploads/{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        # st.markdown(
        #     f"""
        #     <video controls width="500" style="border-radius:10px; margin-top:10px;">
        #         <source src="{video_path}" type="video/mp4">
        #         Your browser does not support the video tag.
        #     </video>
        #     """,
        #     unsafe_allow_html=True
        # )

        st.write("Processing video...")
        progress_bar = st.progress(0)

        if 'audio_features' not in st.session_state or 'visual_emotions' not in st.session_state:
            progress_bar.progress(10)
            audio_path = extract_audio_from_video(video_path)

            # Extract audio features
            progress_bar.progress(30)  # 30% progress after extracting audio features
            audio_features = extract_audio_features(audio_path)

            # Extract visual features from the video
            progress_bar.progress(50)  # 50% progress after extracting visual features
            visual_emotions = extract_visual_features_from_video(video_path)

            st.session_state.audio_features = audio_features
            st.session_state.visual_emotions = visual_emotions

        else:
            # Retrieve features from session state
            audio_features = st.session_state.audio_features
            visual_emotions = st.session_state.visual_emotions

        tab1, tab2, tab3 = st.tabs(["🎤 Speech Model", "😊 Face Model", "🕺 Body Language Model"])


        # If visual features are detected, use the first one (for simplicity)
        with tab1:
            st.header("🎤 Speech Confidence Score")

            if visual_emotions:
                visual_emotions = visual_emotions[0]  # Use the first detected emotion

                # Combine audio and visual features into a single feature set
                # Visual feature contains emotion and confidence, so we need to extract confidence
                confidence = visual_emotions[1]

                # Now, call the calculate_confidence_score function with just the audio features
                progress_bar.progress(70)  # 70% progress before calculating confidence score
                confidence_score, probas, emotion_labels = calculate_confidence_score(audio_features)

                # Display the result
                progress_bar.progress(100)  # 100% progress when the score is ready
                st.markdown(f"<div class='speech-score'><h3>Speech Confidence Score: {confidence_score:.2f}</h3>", unsafe_allow_html=True)
                # Build DataFrame for pie chart
                speech_emotion_df = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Probability': probas[0]
                })
                #filter out disgust
                speech_emotion_df = speech_emotion_df[speech_emotion_df['Emotion'] != 'disgust']


                # Create pie chart
                fig_speech = px.pie(speech_emotion_df,
                                    names='Emotion',
                                    values='Probability',
                                    color_discrete_sequence=px.colors.sequential.RdBu,
                                    title="Detected Emotions (Speech Model)")

                fig_speech.update_traces(textposition='inside', textinfo='percent+label')
                fig_speech.update_layout(width=700, height=700)  # Adjust size here 🚀

                # Show in Streamlit
                st.plotly_chart(fig_speech, use_container_width=True)


                # Add a button to show probabilities
                # Initialize toggle state for speech model probabilities
                if "show_probs_speech" not in st.session_state:
                    st.session_state.show_probs_speech = False

                # Toggle button
                if st.button("Show/Hide Details", key="speech_toggle"):
                    st.session_state.show_probs_speech = not st.session_state.show_probs_speech

                # Conditionally display
                if st.session_state.show_probs_speech:
                    st.write("\nProbabilities for each emotion:")
                    for label, prob in zip(emotion_labels, probas[0]):
                        st.write(f"{label.capitalize()}: {prob:.2f}")
            else:
                st.write("No visual emotion detected in the video.")
                progress_bar.progress(100)  # 100% progress even if no visual emotion detected


        # Run Face Model Analysis
        with tab2:
            st.header("😊 Face Confidence Score")

            with st.spinner("Analyzing facial expressions..."):
                face_conf_pct, face_emotion_summary = analyze_video(video_path=video_path)

            # Display face model results
            st.markdown(f"<div class='face-score'><h3>Facial Confidence Score: {face_conf_pct:.2f}</h3></div>", unsafe_allow_html=True)

            # Absolute path to your project
            project_path = os.path.abspath(".")
            face_model_script = os.path.join(project_path, "face_model/face_model.py")


            # Add pie chart for top emotions
            # After face_emotion_summary is ready:
            if face_emotion_summary:
                labels = list(face_emotion_summary.keys())
                sizes = list(face_emotion_summary.values())

                # Create dataframe for Plotly
                df_emotions = pd.DataFrame({
                    'Emotion': labels,
                    'Percentage': sizes
                })

                # Create pie chart with Plotly
                fig = px.pie(df_emotions,
                            names='Emotion',
                            values='Percentage',
                            color_discrete_sequence=px.colors.sequential.RdBu,
                            title="Top 3 Detected Emotions (Face Model)")

                fig.update_traces(textposition='inside', textinfo='percent+label')

                fig.update_layout(width=700, height=700)  # Adjust size here 🚀

                # Show in Streamlit
                st.plotly_chart(fig, use_container_width=True)

        # Initialize toggle state for face model emotions
            if "show_probs_face" not in st.session_state:
                st.session_state.show_probs_face = False

            # Toggle button
            if st.button("Show/Hide Details", key="face_toggle"):
                st.session_state.show_probs_face = not st.session_state.show_probs_face

            # Conditionally display
            if st.session_state.show_probs_face:
                st.write("\nTop Emotions:")
                for emotion, pct in face_emotion_summary.items():
                    st.write(f"{emotion}: {pct:.2f}%")

            # Path to venv
            # Auto detect venv activate path (pyenv safe)
            venv_activate = os.path.join(os.path.dirname(sys.executable), "activate")

            # Project + script path
            project_path = os.path.abspath(".")
            face_model_script = os.path.join(project_path, "face_model/face_model.py")

            # Build command
            command = f'source {venv_activate} && python {face_model_script}'

            # Build osascript AppleScript command
            osa_command = f'''osascript -e 'tell application "Terminal" to do script "cd {project_path} && {command}"' '''

            # Launch button
            if st.button("🎥 Try Live Webcam Face Detection"):
                st.info("Launching webcam in new Terminal window... Close webcam window to return to app.")
                subprocess.Popen(osa_command, shell=True)

        with tab3:
            st.header("🕺 Body Language Confidence Score")


            # Run Body Model Analysis
            with st.spinner("Analyzing body language..."):
                body_model = init_body_model()
                body_scores = analyze_body_video(video_path=video_path, model=body_model)

            # Display Body Language Score
            if body_scores and "Overall Body Language Score" in body_scores:
                st.markdown(f"<div class='body-score'><h3>Body Language Confidence Score: {body_scores['Overall Body Language Score']:.1f}</h3></div>", unsafe_allow_html=True)
            else:
                st.write("No valid body landmarks detected.")

            # Prepare data → exclude "Overall Body Language Score"
            body_scores_filtered = {k: v for k, v in body_scores.items() if k != "Overall Body Language Score"}

            # Convert to DataFrame
            body_scores_df = pd.DataFrame({
                'Metric': list(body_scores_filtered.keys()),
                'Score': list(body_scores_filtered.values())
            })

            # Create bar chart
            fig_body = px.bar(body_scores_df,
                            x='Metric',
                            y='Score',
                            color='Score',
                            color_continuous_scale='RdBu',
                            title="Detailed Body Language Scores",
                            text='Score')

            fig_body.update_layout(xaxis_title="", yaxis_range=[0, 100])  # Scores are on 0-10 scale

            # Show in Streamlit
            st.plotly_chart(fig_body, use_container_width=True)

            # Toggle for detailed body scores
            if "show_body_scores" not in st.session_state:
                st.session_state.show_body_scores = False

            if st.button("Show/Hide Detailed Body Language Scores"):
                st.session_state.show_body_scores = not st.session_state.show_body_scores

            if st.session_state.show_body_scores and body_scores:
                st.write("Body Language Scores:")
                for k, v in body_scores.items():
                    if k != "Overall Body Language Score":
                        st.write(f"{k}: {v:.1f}")


       
        
elif st.session_state.input_mode == "webcam":
    st.header("🎥 Live Webcam Mode")

    # Show "Start" if not running
    if not st.session_state.webcam_launched:
        if st.button("▶️ Start Live Webcam"):
            subprocess.Popen(["open", "-a", "Terminal", "run_face_live.sh"])
            st.toast("Webcam launched!", icon="🎥")
            st.session_state.webcam_launched = True

    # Show "Stop" if running
    else:
        if st.button("🛑 Stop Live Webcam"):
            open("close_webcam.flag", "w").close()
            st.toast("Sent stop signal!", icon="🛑")
            st.session_state.webcam_launched = False

# Close centered container
st.markdown('</div>', unsafe_allow_html=True)

# import streamlit as st

# st.markdown('test')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

import parselmouth
from fer import FER
import joblib
import cv2
import librosa
import numpy as np

from moviepy.editor import VideoFileClip
import numpy as np
import librosa
import os
import parselmouth

import joblib  # Import joblib to save the model

# Step 2: Extract audio features (MFCC, pitch, jitter, shimmer)
def extract_features(audio_path):
    """
    Extract MFCC features, Jitter, Pitch, and Shimmer from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)  # 'sr' is the sampling rate; None means keep the original
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # We are using 13 MFCC coefficients
    mfcc_mean = np.mean(mfcc, axis=1)  # Average the MFCCs over time
    
    # Extract Jitter, Pitch, and Shimmer using parselmouth
    sound = parselmouth.Sound(audio_path)
    
    # Get pitch
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']  # Get pitch values
    
    # Jitter (variability in pitch)
    jitter = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) != 0 else 0  # Jitter calculation
    
    # Shimmer (variability in amplitude)
    shimmer = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) != 0 else 0  # Simplified shimmer calculation
    
    # Calculate mean of each feature
    pitch_mean = np.mean(pitch_values)
    
    # Combine all features into one array
    combined_features = np.hstack((mfcc_mean, pitch_mean, jitter, shimmer))
    
    return combined_features


# Step 3: Process all audio files in a directory
def process_audio_files(directory):
    """
    Process all the audio files in a given directory, extracting features and labels.
    """
    features = []
    labels = []

    # Loop over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            # Extract the label from the filename (e.g., OAF_back_angry.wav -> angry)
            label = filename.split('_')[-1].replace('.wav', '')  # Assuming emotion is the last part of the filename
            
            # Extract features
            audio_path = os.path.join(directory, filename)
            extracted_features = extract_features(audio_path)
            features.append(extracted_features)
            labels.append(label)

    # Convert to numpy array and return
    return np.array(features), np.array(labels)


def train_model(X, y):
    # Encode the labels (e.g., angry -> 0, sad -> 1)
    directory = "notebooks/dataverse_files"  # Directory containing audio files
    X, y = process_audio_files(directory)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize the RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=200)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Print classification report
    print("Model performance on the test set:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save the trained model and the label encoder using joblib
    joblib.dump(model, 'speech_model/trained_model_combined.joblib')  # Save the model
    joblib.dump(le, 'speech_model/label_encoder.joblib')  # Save the label encoder

    print("Model and label encoder saved successfully.")
    
    
model = joblib.load('speech_model/trained_model_combined.joblib')  # Load your pre-trained model
le = joblib.load('speech_model/label_encoder.joblib')  # Load your label encoder

# Helper function for extracting audio features (MFCC, pitch, jitter, shimmer)
def extract_audio_features(audio_path):
    """
    Extract MFCC features, Jitter, Pitch, and Shimmer from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)  # 'sr' is the sampling rate; None means keep the original
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']

    jitter = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) != 0 else 0
    shimmer = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) != 0 else 0
    pitch_mean = np.mean(pitch_values)

    return np.hstack((mfcc_mean, pitch_mean, jitter, shimmer))

# Helper function for audio extraction from video
def extract_audio_from_video(video_path):
    """
    Extract audio from the video and save it as a .wav file with the same name as the video.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_audio_path = f"{base_name}_extracted_audio.wav"

    # Load the video file and extract audio
    video = VideoFileClip(video_path)
    audio = video.audio

    # Write the audio to a .wav file
    audio.write_audiofile(output_audio_path)

    return output_audio_path

# Helper function for extracting visual features from video using FER
def extract_visual_features_from_video(video_path):
    """
    Extract visual emotion features from the video using FER.
    """
    detector = FER()
    video_capture = cv2.VideoCapture(video_path)

    visual_emotions = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect emotions from the frame
        emotions = detector.top_emotion(frame)  # Returns the top emotion (e.g., ('happy', 0.8))

        if emotions:
            emotion, confidence = emotions
            visual_emotions.append((emotion, confidence))

    video_capture.release()

    return visual_emotions

# Combine audio and visual features into a single feature vector
def combine_audio_visual_features(audio_features, visual_features):
    """
    Combine audio and visual features into a single feature vector.
    """
    audio_feature_vector = np.array(audio_features)  # Convert audio features to numpy array
    visual_feature_vector = np.array([confidence for emotion, confidence in visual_features])  # Convert visual features to numpy array

    # Concatenate the two feature vectors
    combined_features = np.concatenate((audio_feature_vector, visual_feature_vector))

    return combined_features    

# Calculate the confidence score based on audio and visual features
def calculate_confidence_score(audio_features):
    """
    Calculate the confidence score based on the model's probabilities for 'happy', 'ps', 'angry', and 'neutral'.
    Higher scores indicate more confidence in the presentation.
    """
    # Extract features from the new audio file
    new_features = np.array(audio_features).reshape(1, -1)  # Reshape to match model input

    # Get the probabilities of each class (emotion)
    probas = model.predict_proba(new_features)

    # Define the class labels for emotions (focus on 'happy', 'ps', 'angry', 'neutral')
    emotion_labels = le.classes_  # e.g., ['angry', 'happy', 'nervous', ...]

    # Extract the relevant probabilities for 'happy', 'ps', 'angry', and 'neutral'
    happy_prob = probas[0][np.where(emotion_labels == 'happy')[0][0]]
    ps_prob = probas[0][np.where(emotion_labels == 'ps')[0][0]]
    angry_prob = probas[0][np.where(emotion_labels == 'angry')[0][0]]
    neutral_prob = probas[0][np.where(emotion_labels == 'neutral')[0][0]]

    # Print probabilities for each emotion of interest
    # print(f"Happy Probability: {happy_prob:.2f}")
    # print(f"PS Probability: {ps_prob:.2f}")
    # print(f"Angry Probability: {angry_prob:.2f}")
    # print(f"Neutral Probability: {neutral_prob:.2f}")

    # ("\nProbabilities for each emotion:")
    # for label, prob in zip(emotion_labels, probas[0]):
    #     (f"{label.capitalize()}: {prob:.2f}")

    # Confidence Score Calculation:
    # Increasing the weight for confidence emotions ('happy', 'ps', 'angry')
    confidence_emotions_prob = (happy_prob * 3 + ps_prob * 3 + angry_prob * 3)  # Increased weight for confidence emotions
    confidence_score = max(confidence_emotions_prob, 0.05) * 100  # Ensure a minimum contribution

    # Reduce the weight of neutral's penalty (dampen the effect)
    neutral_penalty = neutral_prob * 20  # Reduced penalty (previously 50)

    # Subtract the neutral penalty from the confidence score to get the final score
    final_confidence_score = confidence_score - neutral_penalty

    # Boost the confidence score by 20 if the probability of neutral is 0
    if neutral_prob <= 0.01:
        final_confidence_score += 20  # Boost confidence score if neutral is 0

    # Apply a minimum threshold to ensure the score doesn't drop below a certain level (e.g., 10)
    final_confidence_score = max(final_confidence_score, 10)

    # Clamp the score to a range of 0 to 100 for visualization
    final_confidence_score = max(0, min(100, final_confidence_score))

    print(f"Final Confidence Score: {final_confidence_score:.2f}")
    return final_confidence_score, probas, emotion_labels
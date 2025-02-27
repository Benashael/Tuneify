import streamlit as st
import numpy as np
import librosa
import librosa.display
import io
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

def download_gtzan_dataset():
    if not os.path.exists("gtzan_dataset.zip"):
        os.system("wget -q --show-progress http://opihi.cs.uvic.ca/sound/genres.tar.gz -O gtzan_dataset.zip")
        os.system("mkdir -p gtzan_dataset && tar -xzf gtzan_dataset.zip -C gtzan_dataset --strip-components=1")

# Ensure GTZAN dataset is available
download_gtzan_dataset()

# Load GTZAN dataset samples
@st.cache_data
def load_gtzan_samples():
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    dataset_path = "./gtzan_dataset/"
    samples = []
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if os.path.exists(genre_path) and os.listdir(genre_path):
            file = random.choice(os.listdir(genre_path))
            file_path = os.path.join(genre_path, file)
            y, sr = librosa.load(file_path, sr=22050)
            mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
            samples.append({"Title": file, "Artist": "Unknown", "Genre": genre, "MFCC_Features": mfcc_features})
    return samples

dataset = load_gtzan_samples()

def extract_mfcc(audio_bytes, sr=22050, n_mfcc=13):
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def find_closest_match(query_mfcc, dataset, threshold):
    query_mfcc_mean = np.mean(query_mfcc, axis=1)
    for song in dataset:
        distance = np.linalg.norm(np.array(song["MFCC_Features"]) - query_mfcc_mean)
        if distance <= threshold:
            return song
    return None

st.title("🎵 Tuneify: Music Recognition & Recommendation")
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

distance_threshold = st.slider("Set Euclidean Distance Threshold", min_value=0.0, max_value=100.0, value=20.0, step=0.5)

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    audio_bytes = uploaded_file.read()
    query_mfcc = extract_mfcc(audio_bytes)
    
    # Display MFCC
    fig, ax = plt.subplots()
    librosa.display.specshow(query_mfcc, x_axis='time')
    ax.set_title("MFCC Features of Uploaded Audio")
    st.pyplot(fig)
    
    matched_song = find_closest_match(query_mfcc, dataset, distance_threshold)
    
    if matched_song is not None:
        st.subheader("🎶 Identified Song")
        st.write(f"**Title:** {matched_song['Title']}")
        st.write(f"**Artist:** {matched_song['Artist']}")
        st.write(f"**Genre:** {matched_song['Genre']}")
    else:
        st.write("❌ No recommendations found. Try adjusting the threshold.")

import streamlit as st
import librosa
import torch
import noisereduce as nr
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
from transformers import AutoFeatureExtractor, WhisperModel
import joblib
import xgboost as xgb
from audiorecorder import audiorecorder
import numpy as np
from io import BytesIO
import soundfile as sf
import time
from pydub import AudioSegment, effects
from st_audiorec import st_audiorec
from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks
import io


classifier = joblib.load('./svm_classifier.joblib')
vad_model = load_silero_vad(onnx=False)
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
whisper_model = WhisperModel.from_pretrained("openai/whisper-base", add_cross_attention=False)


def apply_noise_reduction(audio_data, sample_rate):
    
    S_full, phase = librosa.magphase(librosa.stft(audio_data)) 

    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(0.2, sr=sample_rate)))
    S_filter = np.minimum(S_full, S_filter) 

    margin_i, margin_v = 10, 1
    power = 3

    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    noise_reduced_audio = librosa.istft(S_foreground * phase)

    noise_reduced_audio = (noise_reduced_audio * 32767).astype(np.int16)
    noise_reduced_audio = AudioSegment(
        noise_reduced_audio.tobytes(),  
        frame_rate=sample_rate,  
        sample_width=2,  
        channels=1  
    )

    noise_reduced_audio = effects.normalize(noise_reduced_audio, headroom=2)

    noise_reduced_audio = np.array(noise_reduced_audio.get_array_of_samples()).astype(np.float32) / 32767

    return noise_reduced_audio

def detect_speech(audio_data, sample_rate):
    time.sleep(1)  
    audio_tensor = torch.tensor(audio_data).float()
    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sample_rate)
    
    if len(speech_timestamps) == 0:
        return "No speech detected", False, speech_timestamps
    
    return "Speech detected", True, speech_timestamps

def classify_gender(noise_reduced_audio, sample_rate):
    inputs = feature_extractor(noise_reduced_audio, return_tensors="pt", sampling_rate=sample_rate)
    input_features = inputs.input_features     
    with torch.no_grad():
        outputs = whisper_model.encoder(input_features)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    gender_prediction = classifier.predict(embeddings.reshape(1, -1))
    gender = "Male" if gender_prediction == 0 else "Female"
    
    return gender

if 'audio' not in st.session_state:
    st.session_state['audio'] = None
if 'noise_reduced_audio' not in st.session_state:
    st.session_state['noise_reduced_audio'] = None
if 'output' not in st.session_state:
    st.session_state['output'] = None

st.title("Audio Gender Classification Demo")
st.write("Upload or record an audio file to detect if someone is speaking and classify the speaker's gender.")
st.write("NOTE: The pipeline will take more time on the first run only")

audio_source = st.radio("Choose audio source:", ("Upload", "Record"))

def clear_outputs():
    st.session_state['audio'] = None
    st.session_state['noise_reduced_audio'] = None
    st.session_state['output'] = None

def update_progress(label_text, progress_value, status_label, progress_bar):
    status_label.text(label_text)
    progress_bar.progress(progress_value)

if audio_source == "Upload":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"], on_change=clear_outputs)

    status_label = st.empty()
    progress_bar = st.progress(0)

    if audio_file is not None:
        if audio_file.name.endswith('.m4a'):
            audio = AudioSegment.from_file(audio_file, format='m4a')
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            
            audio_data, sample_rate = librosa.load(wav_io, sr=16000)
        else:
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
        st.session_state['audio'] = audio_file.getvalue()

        update_progress("Applying noise reduction...", 20, status_label, progress_bar)
        noise_reduced_audio = apply_noise_reduction(audio_data, sample_rate)
        st.session_state['noise_reduced_audio'] = noise_reduced_audio

        update_progress("Detecting speech...", 50, status_label, progress_bar)
        speech_status, contains_speech, speech_timestamps = detect_speech(noise_reduced_audio, sample_rate)
        uncut_noise_reduced_audio = noise_reduced_audio

        if contains_speech:
            update_progress("Classifying gender...", 75, status_label, progress_bar)
            noise_reduced_audio = torch.tensor(noise_reduced_audio).float()
            noise_reduced_audio = collect_chunks(speech_timestamps, noise_reduced_audio)
            gender = classify_gender(noise_reduced_audio, sample_rate)
            result_text = f"Speech detected. The speaker is: {gender}"
        else:
            result_text = "No speech detected."

        update_progress("Task completed", 100, status_label, progress_bar)
        status_label.empty()  

        st.subheader("Task completed")
        st.write(f"**Result:** {result_text}")

        st.write("Original Audio:")
        st.audio(st.session_state['audio'], format='audio/wav', start_time=0)

        st.write("Uncut Filtered Audio:")
        uncut_noise_reduced_bytes = BytesIO()
        sf.write(uncut_noise_reduced_bytes, uncut_noise_reduced_audio, sample_rate, format='WAV')
        st.audio(uncut_noise_reduced_bytes.getvalue(), format='audio/wav')

        st.write("Filtered Audio:")
        noise_reduced_bytes = BytesIO()
        sf.write(noise_reduced_bytes, noise_reduced_audio, sample_rate, format='WAV')
        st.audio(noise_reduced_bytes.getvalue(), format='audio/wav')
        
elif audio_source == "Record":
    audio = audiorecorder("Click to record", "Click to stop recording", show_visualizer=True)
    status_label = st.empty()
    progress_bar = st.progress(0)

    if len(audio) > 0:
        audio_bytes = BytesIO(audio.export().read())
        audio_data, sample_rate = sf.read(audio_bytes)
        
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        st.session_state['audio'] = audio.export().read()

        update_progress("Applying noise reduction...", 20, status_label, progress_bar)
        noise_reduced_audio = apply_noise_reduction(audio_data, sample_rate)
        st.session_state['noise_reduced_audio'] = noise_reduced_audio

        update_progress("Detecting speech...", 50, status_label, progress_bar)
        speech_status, contains_speech, speech_timestamps = detect_speech(noise_reduced_audio, sample_rate)
        uncut_noise_reduced_audio = noise_reduced_audio

        if contains_speech:
            update_progress("Classifying gender...", 75, status_label, progress_bar)
            noise_reduced_audio = torch.tensor(noise_reduced_audio).float()
            noise_reduced_audio = collect_chunks(speech_timestamps, noise_reduced_audio)
            gender = classify_gender(noise_reduced_audio, sample_rate)
            result_text = f"Speech detected. The speaker is: {gender}"
        else:
            result_text = "No speech detected."

        update_progress("Task completed", 100, status_label, progress_bar)
        status_label.empty()  

        st.subheader("Task completed")
        st.write(f"**Result:** {result_text}")

        st.write("Original Recorded Audio:")
        st.audio(st.session_state['audio'], format='audio/wav', start_time=0)

        st.write("Uncut Filtered Audio:")
        uncut_noise_reduced_bytes = BytesIO()
        sf.write(uncut_noise_reduced_bytes, uncut_noise_reduced_audio, sample_rate, format='WAV')
        st.audio(uncut_noise_reduced_bytes.getvalue(), format='audio/wav')

        st.write("Filtered Recorded Audio:")
        noise_reduced_bytes = BytesIO()
        sf.write(noise_reduced_bytes, noise_reduced_audio, sample_rate, format='WAV')
        st.audio(noise_reduced_bytes.getvalue(), format='audio/wav')
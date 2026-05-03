'''
Amber Lynch (001313385)

acoustic algorithm - audio analysis
'''

import os
import sys
import json
import tempfile
import subprocess
from os.path import samefile

import numpy as np
import librosa
import joblib

# avoid window pop ups
subprocess_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

model_path = "models/audio_model.pkl"
meta_path = "models/audio_model_meta.json"
sample_rate = 16000
N_MFCC = 40
threshold_fake = 0.55
threshold_suspicious = 0.25

# model cache
_model = None
_meta = None


def _load_model():
    global _model, _meta
    if _model is not None:
        return _model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Run Audio_Analysis.py first."
        )
    _model = joblib.load(model_path)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            _meta = json.load(f)
        print(f"[Audio] Model loaded. Accuracy: {_meta.get('accuracy', 'N/A')}, "
              f"F1: {_meta.get('f1_score', 'N/A')}")
    else:
        print("[Audio] Model loaded (no metadata found)")
    return _model


# audio convert
def convert_to_wav(input_path):
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()

    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        '-y',
        '-loglevel', 'error',
        temp_wav.name
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess_flags)

    if result.returncode != 0:
        os.unlink(temp_wav.name)
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    if os.path.getsize(temp_wav.name) == 0:
        os.unlink(temp_wav.name)
        raise RuntimeError("FFmpeg produced empty file. No audio track found.")

    return temp_wav.name


def _extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    audio, _ = librosa.effects.trim(audio, top_db=20)

    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    return np.concatenate([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(delta_mfccs, axis=1), np.std(delta_mfccs, axis=1),
        np.mean(delta2_mfccs, axis=1), np.std(delta2_mfccs, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        [np.mean(librosa.feature.zero_crossing_rate(y=audio)),
         np.mean(librosa.feature.rms(y=audio))]
    ])


# predictor
def predict_audio(wav_path):
    model = _load_model()

    try:
        features = _extract_features(wav_path).reshape(1, -1)
    except Exception as e:
        return _error_result(wav_path, f"Feature extraction failed: {e}")

    probs = model.predict_proba(features)[0]
    fake_prob = probs[1]
    real_prob = probs[0]

    if fake_prob >= threshold_fake:
        verdict = "FAKE"
        confidence = fake_prob
        reason = (f"Audio shows strong signs of AI generation "
                  f"({fake_prob:.1%} probability). "
                  f"Acoustic properties inconsistent with natural speech.")
    elif fake_prob >= threshold_suspicious:
        verdict = "SUSPICIOUS"
        confidence = fake_prob
        reason = (f"Audio has unusual acoustic characteristics "
                  f"({fake_prob:.1%} fake probability). "
                  f"Could be AI-generated or low quality.")
    else:
        verdict = "REAL"
        confidence = real_prob
        reason = (f"Audio consistent with natural human speech "
                  f"({real_prob:.1%} real probability).")

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "reason": reason,
        "file": os.path.basename(wav_path),
        "model": "audio"
    }


def predict_from_video(video_path):
    if not os.path.exists(video_path):
        return _error_result(video_path, f"File not found: {video_path}")

    temp_wav = None
    try:
        temp_wav = convert_to_wav(video_path)
        result = predict_audio(temp_wav)
        result["file"] = os.path.basename(video_path)
        return result
    except RuntimeError as e:
        return _error_result(video_path, str(e), verdict="INCONCLUSIVE")
    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)


def _error_result(file_path, reason, verdict="ERROR"):
    return {
        "verdict": verdict,
        "confidence": 0.0,
        "fake_probability": 0.0,
        "real_probability": 0.0,
        "reason": reason,
        "file": os.path.basename(file_path),
        "model": "audio"
    }

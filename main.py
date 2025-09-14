import librosa
import pandas as pd
import numpy as np
import parselmouth
import joblib

import os
import io

from typing import Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse


def extract_audio_features(mp3_content):
    y, sr = librosa.load(mp3_content)
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1)

    # RMS
    rms = librosa.feature.rms(y=y).mean()

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = pitches[pitches>0].mean() if np.any(pitches>0) else np.nan

    # Parselmouth features
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_mean = parselmouth.praat.call(hnr, "Get mean", 0, 0)
        formant = snd.to_formant_burg()
        f1 = formant.get_mean(1)
        f2 = formant.get_mean(2)
        f3 = formant.get_mean(3)
    except:
        jitter_local, shimmer_local, hnr_mean = np.nan, np.nan, np.nan
        f1, f2, f3 = np.nan, np.nan, np.nan

    return mfcc_mean, pitch_mean, rms, jitter_local, shimmer_local, hnr_mean, f1, f2, f3


def load_model():
    f = open('age_model.pkl', 'rb')
    model = joblib.load(f)
    return model


app = FastAPI()
model = load_model()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/get_predictions/")
async def run_model(file: UploadFile):
    mp3_content = await file.read()
    mp3_content = io.BytesIO(mp3_content)
    mfcc_mean, pitch_mean, rms, jitter, shimmer, hnr, f1, f2, f3 = extract_audio_features(mp3_content)

    row = {}
    for i in range(len(mfcc_mean)):
        row[f'MFCC_{i+1}'] = mfcc_mean[i]

    row.update({
                'Pitch_mean': pitch_mean,
                'RMS_energy': rms,
                'Jitter_local': jitter,
                'Shimmer_local': shimmer,
                'HNR_mean': hnr,
                'F1_mean': f1,
                'F2_mean': f2,
                'F3_mean': f3
            })

    row = {k: row[k] for k in model.feature_names_in_}
    row = pd.DataFrame([row])

    result = model.predict(row)[0]

    print('here')
    return JSONResponse(content={"filename": file.filename, "prediction": result})

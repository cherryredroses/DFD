# DFD-DeepFake Detector
Setup & Usage

FYP: A Hybrid, Multi-Modal Browser Extension for Deepfake Video Detection

Amber Lynch (001313385)

University of Greenwich

---

## Project Structure

```
DFD/
├── backend.py              # Flask server, hybrid scoring, yt-dlp integration
├── EAR_Predict.py          # EAR visual model 
├── shape_predictor_68_face_landmarks.dat          <-------- you will need to download (see below)
├── models/
│   ├── audio_model.pkl 
│   └── audio_model_meta.json
└── extension/
    ├── manifest.json
    ├── popup.html
    ├── popup.js
    ├── content.js
    ├── logo.png
    ├── background.js


```

---

## Step 1 — Install Python dependencies

```bash
pip install flask flask-cors requests opencv-python dlib imutils librosa scikit-learn joblib scipy
```

FFmpeg is also required for audio extraction:
- Windows: https://ffmpeg.org/download.html (add to PATH)
- Mac: brew install ffmpeg
- Linux: sudo apt install ffmpeg

---

## Step 2 — Get required model files

**dlib landmark file** (download once, ~99MB):
```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```
Extract and place `shape_predictor_68_face_landmarks.dat` in the project root.

---


## Step 4 — Start the backend server

```bash
python backend.py
```

Server runs at: http://localhost:5000
- Health check: GET  http://localhost:5000/health
- Analyse URL:  POST http://localhost:5000/analyse_url
- Analyse file: POST http://localhost:5000/analyse_file

The backend must be running for the Chrome extension to work.

---

## Step 5 — Load the Chrome extension

1. Open Chrome and go to: chrome://extensions/

    <img width="256" height="47" alt="Screenshot 2026-05-03 035247" src="https://github.com/user-attachments/assets/4575b917-e826-4a1e-a101-a19aee23933c" />
    
2. Enable "Developer mode" (top right toggle)
   
3. Click "Load unpacked"

    <img width="143" height="58" alt="Screenshot 2026-05-03 035313" src="https://github.com/user-attachments/assets/d227daac-701e-43ee-9483-d6a932ab6347" />

4. Select the `extension/` folder

   <img width="766" height="392" alt="Screenshot 2026-05-03 035332" src="https://github.com/user-attachments/assets/85b60c22-309e-42ff-a449-482fb413985a" />
   
   <img width="433" height="273" alt="Screenshot 2026-05-03 035353" src="https://github.com/user-attachments/assets/9f33158c-56a7-4d97-9ba1-f2e8726ec5fd" />
    
   
5. The DFD icon will appear in your toolbar

    <img width="385" height="398" alt="Screenshot 2026-05-03 035443" src="https://github.com/user-attachments/assets/13b04bc3-bc37-44f0-99fc-ad853492f554" />




---

## Step 6 — Using the extension

1. Start the backend: `python backend.py`
2. Navigate to a video on YouTube, TikTok, Instagram, X or Facebook
3. Click the DeepGuard icon in the toolbar
4. Click "Analyse This Video"
5. Wait for results (typically 10-30 seconds)


---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Backend offline` in extension | Run `python backend.py` in terminal |
| `dlib landmark model not found` | Download `.dat` file (see Step 2) |
| `No video detected` on page | Navigate to a page with an actual video playing |
| FFmpeg errors | Install FFmpeg and ensure it's in PATH |

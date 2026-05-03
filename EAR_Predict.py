'''
Amber Lynch (001313385)

EAR algorithm - Blink analysis
'''

import cv2
import dlib
import numpy as np
import os
from imutils import face_utils
from scipy.spatial import distance as dist

# configuration
predictor_path = "shape_predictor_68_face_landmarks.dat"
frame_skip = 3
max_frames = 450
calibration_frames = 30

blink_rate_low_fake = 3
blink_rate_low_suspicious = 8
blink_rate_high_fake = 40
blink_rate_high_suspicious = 30
desync_ratio_threshold = 0.12

# dlib loading
_detector = None
_predictor = None
_lStart = _lEnd = _rStart = _rEnd = None


def _load_dlib():

    global _detector, _predictor, _lStart, _lEnd, _rStart, _rEnd

    if _detector is not None:   # model already loaded
        return

    if not os.path.exists(predictor_path): # model missing
        raise FileNotFoundError(
            f"dlib landmark model not found at '{predictor_path}'. "
            f"Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )

    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(predictor_path)
    (_lStart, _lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (_rStart, _rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# EAR algorithm
def eye_aspect_ratio(eye):

    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


# predictor
def predict_from_video(video_path):

    try:
        _load_dlib()
    except FileNotFoundError as e:
        return _error_result(video_path, str(e))

    if not os.path.exists(video_path):
        return _error_result(video_path, f"File not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _error_result(video_path, "Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    blink_count = 0
    is_blinking = False
    blink_start_frame = 0
    blink_intervals = []
    last_blink_frame = None
    ear_history = []
    dynamic_threshold = 0.2
    calibrated = False
    desync_events = 0
    face_frames = 0
    processed_frames = 0

    # video running
    while True:
        ret, frame = cap.read()
        if not ret or processed_frames >= max_frames:
            break

        processed_frames += 1
        if processed_frames % frame_skip != 0:
            continue

        # resize frame
        h, w = frame.shape[:2]
        scale = 480 / max(w, h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = _detector(gray, 0)

        for rect in rects:
            face_frames += 1
            shape = _predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_ear = eye_aspect_ratio(shape[_lStart:_lEnd])
            right_ear = eye_aspect_ratio(shape[_rStart:_rEnd])
            ear = (left_ear + right_ear) / 2.0

            if calibrated:
                print(
                    f"[EAR] frame={processed_frames} "
                    f"ear={ear:.3f} "
                    f"threshold={dynamic_threshold:.3f} "
                    f"blinking={is_blinking}")

            if face_frames <= calibration_frames:
                ear_history.append(ear)
                if face_frames == calibration_frames:
                    mean_ear = np.mean(ear_history)
                    std_ear = np.std(ear_history)
                    dynamic_threshold = max(0.12, min(mean_ear - (1 * std_ear), 0.25))
                    calibrated = True
                    print(f"[EAR] Calibrated: mean={mean_ear:.3f} "
                          f"std={std_ear:.3f} threshold={dynamic_threshold:.3f}")
                continue

            if not calibrated:
                continue

            # desync
            if abs(left_ear - right_ear) > 0.06:
                desync_events += 1

            if ear < dynamic_threshold:
                if not is_blinking:
                    is_blinking = True
                    blink_start_frame = processed_frames
            else:
                if is_blinking:
                    duration = processed_frames - blink_start_frame
                    if 1 <= duration <= 6:
                        blink_count += 1
                        if last_blink_frame is not None:
                            blink_intervals.append(processed_frames - last_blink_frame)
                        last_blink_frame = processed_frames
                    is_blinking = False

    cap.release()

    if face_frames == 0:
        return _error_result(
            video_path,
            "No face detected.",
            verdict="INCONCLUSIVE"
        )

    duration_sec = processed_frames / fps
    blink_rate = (blink_count / duration_sec) * 60 if duration_sec > 0 else 0
    desync_ratio = desync_events / face_frames
    blink_variance = np.var(blink_intervals) if len(blink_intervals) >= 3 else None

    # scoring
    fake_score = 0.0
    if blink_rate < blink_rate_low_fake or blink_rate > blink_rate_high_fake:
        fake_score += 0.5
    elif blink_rate < blink_rate_low_suspicious or blink_rate > blink_rate_high_suspicious:
        fake_score += 0.25
    if desync_ratio > desync_ratio_threshold:
        fake_score += 0.3
    if blink_variance is not None and blink_variance < 5:
        fake_score += 0.2

    if fake_score >= 0.5:
        verdict = "FAKE"
        confidence = min(fake_score, 1.0)
    elif fake_score >= 0.25:
        verdict = "SUSPICIOUS"
        confidence = fake_score
    else:
        verdict = "REAL"
        confidence = 1.0 - fake_score

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "reason": _build_reason(verdict, blink_rate, desync_ratio, blink_variance),
        "blink_rate": round(blink_rate, 1),
        "blink_count": blink_count,
        "desync_ratio": round(desync_ratio, 4),
        "duration": round(duration_sec, 2),
        "face_frames": face_frames,
        "file": os.path.basename(video_path),
        "model": "ear"
    }


def _build_reason(verdict, blink_rate, desync_ratio, blink_variance):
    if verdict == "REAL":
        return f"Blink pattern consistent with natural human behaviour ({blink_rate:.1f} blinks/min)."
    parts = []
    if blink_rate < 3:
        parts.append(f"blink rate near zero ({blink_rate:.1f}/min)")
    elif blink_rate < 8:
        parts.append(f"low blink rate ({blink_rate:.1f}/min)")
    elif blink_rate > 35:
        parts.append(f"abnormally high blink rate ({blink_rate:.1f}/min)")
    elif blink_rate > 26:
        parts.append(f"elevated blink rate ({blink_rate:.1f}/min)")
    if desync_ratio > desync_ratio_threshold:
        parts.append(f"eye asymmetry detected in {desync_ratio:.0%} of frames")
    if blink_variance is not None and blink_variance < 5:
        parts.append("unnaturally regular blink timing")
    return ("Unusual patterns: " + "; ".join(parts) + ".") if parts else "Some anomalies detected."


def _error_result(file_path, reason, verdict="ERROR"):
    return {
        "verdict": verdict,
        "confidence": 0.0,
        "reason": reason,
        "blink_rate": 0,
        "blink_count": 0,
        "desync_ratio": 0,
        "duration": 0,
        "face_frames": 0,
        "file": os.path.basename(file_path),
        "model": "ear"
    }

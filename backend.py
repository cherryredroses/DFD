'''
Amber Lynch (001313385)

backend - chrome ext connect
'''

import os
import sys
import tempfile
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import EAR_Predict
import Audio_Predict

subprocess_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

app = Flask(__name__)
CORS(app)

supported_platforms = [
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "fb.watch"
]



def download_with_ytdlp(url):
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--output", tmp_path,
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--quiet",
    ]

    # platform specs
    if "instagram.com" in url:
        cmd += ["--add-header", "Referer:https://www.instagram.com/"]

    if "tiktok.com" in url:
        cmd += ["--add-header",
                "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"]
        cmd += ["--add-header", "Referer:https://www.tiktok.com/"]

    cmd.append(url)

    print(f"[Backend] Running yt-dlp for: {url}")
    result_ = subprocess.run(cmd, capture_output=True, text=True, creationflags=subprocess_flags)

    if result_.returncode != 0:
        error_msg = result_.stderr.strip()[:300] if result_.stderr else "Unknown error"
        raise RuntimeError(
            f"yt-dlp could not download this video. "
            f"Make sure yt-dlp is installed (pip install yt-dlp) "
            f"and the URL is a public post. Details: {error_msg}"
        )

    for fname in os.listdir(tmp_dir):
        full_path = os.path.join(tmp_dir, fname)
        if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
            print(f"[Backend] Downloaded: {fname} ({os.path.getsize(full_path)//1024}KB)")
            return full_path

    raise RuntimeError("yt-dlp ran successfully but produced no output file.")


# scoring
def combine_results(ear_result, audio_result):

    ear_weight   = 0.5
    audio_weight = 0.5   #50/50 weight
    scores  = {"REAL": 0, "SUSPICIOUS": 1, "FAKE": 2}

    ear_verdict = ear_result.get("verdict", "INCONCLUSIVE")
    audio_verdict = audio_result.get("verdict", "INCONCLUSIVE")

    ear_inc   = ear_verdict in ("INCONCLUSIVE", "ERROR")
    audio_inc = audio_verdict in ("INCONCLUSIVE", "ERROR")

    if ear_inc and audio_inc:
        return {
            "verdict"    : "INCONCLUSIVE",
            "confidence" : 0.0,
            "reason"     : "Neither model could analyse this video.",
            "ear"        : ear_result,
            "audio"      : audio_result
        }

    if audio_inc:
        return {
            "verdict"    : ear_verdict,
            "confidence" : ear_result.get("confidence", 0.0),
            "reason"     : f"Audio unavailable. Visual: {ear_result.get('reason', '')}",
            "ear"        : ear_result,
            "audio"      : audio_result
        }

    if ear_inc:
        return {
            "verdict"    : audio_verdict,
            "confidence" : audio_result.get("confidence", 0.0),
            "reason"     : f"Visual unavailable. Audio: {audio_result.get('reason', '')}",
            "ear"        : ear_result,
            "audio"      : audio_result
        }

    combined_score = (scores.get(ear_verdict,
                                 1) * ear_weight) + (scores.get(audio_verdict, 1) * audio_weight)
    combined_conf  = (
        ear_result.get("confidence", 0.5) * ear_weight +
        audio_result.get("confidence", 0.5) * audio_weight
    )

    if combined_score >= 1.5:
        verdict = "FAKE"
    elif combined_score >= 0.75:
        verdict = "SUSPICIOUS"
    else:
        verdict = "REAL"

    if ear_verdict == audio_verdict:
        reason = (f"Both models agree: {verdict}. "
                  f"Visual {ear_result.get('reason', '')} "
                  f"Audio {audio_result.get('reason', '')}")
    else:
        reason = (f"Models give mixed verdicts, (Visual: {ear_verdict}, Audio: {audio_verdict}). "
                  f"Visual {ear_result.get('reason', '')} "
                  f"Audio {audio_result.get('reason', '')}")

    return {
        "verdict"    : verdict,
        "confidence" : round(combined_conf, 4),
        "reason"     : reason,
        "ear"        : ear_result,
        "audio"      : audio_result
    }


def analyse_video(video_path):
    print(f"[Backend] Analysing: {os.path.basename(video_path)}")

    print("[Backend] Running EAR (visual) analysis...")
    ear = EAR_Predict.predict_from_video(video_path)
    print(f"[Backend] EAR: {ear['verdict']} ({ear.get('confidence', 0):.1%})")

    print("[Backend] Running audio analysis...")
    audio = Audio_Predict.predict_from_video(video_path)
    print(f"[Backend] Audio: {audio['verdict']} ({audio.get('confidence', 0):.1%})")

    hybrid = combine_results(ear, audio)
    print(f"[Backend] Hybrid verdict: {hybrid['verdict']} ({hybrid['confidence']:.1%})")
    return hybrid


def _run_with_cleanup(acquire_fn):
    tmp = None
    try:
        tmp    = acquire_fn()
        result_ = analyse_video(tmp)
        return jsonify(result_)
    except RuntimeError as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}", "verdict": "ERROR"}), 500
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass


# Flask endpoints

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "DFD backend running."})


@app.route('/analyse_social', methods=['POST'])
def analyse_social():

    data = request.get_json()
    if not data or "page_url" not in data:
        return jsonify({"error": "Missing 'page_url' in request body."}), 400

    url = data["page_url"].strip()

    # validate
    if not any(platform in url for platform in supported_platforms):
        return jsonify({
            "error": f"Unsupported platform. Supported: Instagram, YouTube, TikTok, Twitter/X, Facebook.",
            "verdict": "ERROR"
        }), 400

    return _run_with_cleanup(lambda: download_with_ytdlp(url))


@app.route('/analyse_file', methods=['POST'])
def analyse_file():

    data = request.get_json()
    if not data or "video_path" not in data:
        return jsonify({"error": "Missing 'video_path'."}), 400

    path_ = data["video_path"]
    if not os.path.exists(path_):
        return jsonify({"error": f"File not found: {path_}"}), 404

    try:
        result_ = analyse_video(path_)
        return jsonify(result_)
    except Exception as e:
        return jsonify({"error": str(e), "verdict": "ERROR"}), 500


# display
def print_results(result, label):
    print("\n" + "=" * 60)
    print(f"Result: {label}")
    print("=" * 60)
    print(f"  VERDICT    : {result['verdict']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Reason     : {result['reason']}")
    print("-" * 60)
    ear = result.get("ear", {})
    print(f"    EAR          : {ear.get('verdict','N/A')} ({ear.get('confidence',0):.1%})")
    print(f"    Blink rate   : {ear.get('blink_rate','N/A')} /min")
    print(f"    Blink count  : {ear.get('blink_count','N/A')}")
    print(f"    Desync ratio : {ear.get('desync_ratio','N/A')}")
    print(f"    Reason       : {ear.get('reason','N/A')}")
    print("-" * 60)
    audio = result.get("audio", {})
    print(f"    Audio      : {audio.get('verdict','N/A')} ({audio.get('confidence',0):.1%})")
    print(f"    Fake prob  : {audio.get('fake_probability','N/A')}")
    print(f"    Real prob  : {audio.get('real_probability','N/A')}")
    print(f"    Reason     : {audio.get('reason','N/A')}")
    print("=" * 60)


# main
if __name__ == "__main__":

    if "--test" in sys.argv:
        print("=" * 60)
        print("DFD BACKEND : LOCAL TEST MODE")
        print("=" * 60)

        tests = [
            ("data/testing/fake/heygen_amber_test3.mp4", "Fake video"),
            ("data/testing/real/real-004.mp4",        "Real video"),
        ]

        for path, label in tests:
            print(f"\n[TEST] {label}: {path}")
            if os.path.exists(path):
                result = analyse_video(path)
                print_results(result, label)
            else:
                print(f"  [!] File not found: {path}")

    else:
        print("=" * 60) 
        print("DFD BACKEND")
        print("=" * 60)
        print("  Health check  : GET  http://localhost:5000/health")
        print("  Social media  : POST http://localhost:5000/analyse_social")
        print("  Local file    : POST http://localhost:5000/analyse_file")
        print("  Local test    : python backend.py --test")
        print("=" * 60)
        app.run(host="0.0.0.0", port=5000, debug=False)
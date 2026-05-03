// popup.js — DFD Deepfake Detection Tool
// FYP: Amber Lynch (001313385), University of Greenwich

const BACKEND_URL = "http://localhost:5000";

const SUPPORTED_PLATFORMS = [
  "instagram.com",
  "youtube.com",
  "youtu.be",
  "tiktok.com",
  "twitter.com",
  "x.com",
  "facebook.com",
  "fb.watch"
];


const statusDot     = document.getElementById("statusDot");
const statusText    = document.getElementById("statusText");
const videoInfo     = document.getElementById("videoInfo");
const videoUrlEl    = document.getElementById("videoUrl");
const noVideo       = document.getElementById("noVideo");
const btnAnalyse    = document.getElementById("btnAnalyse");
const loading       = document.getElementById("loading");
const loadingText   = document.getElementById("loadingText");
const resultCard    = document.getElementById("resultCard");
const resultHeader  = document.getElementById("resultHeader");
const resultVerdict = document.getElementById("resultVerdict");
const resultConf    = document.getElementById("resultConfidence");
const resultReason  = document.getElementById("resultReason");
const earBadge      = document.getElementById("earBadge");
const earDetail     = document.getElementById("earDetail");
const audioBadge    = document.getElementById("audioBadge");
const audioDetail   = document.getElementById("audioDetail");
const errorBox      = document.getElementById("errorBox");

let currentPageUrl = null;

function verdictClass(v) {
  return ({REAL:"real",FAKE:"fake",SUSPICIOUS:"suspicious",INCONCLUSIVE:"inconclusive"})[v] || "inconclusive";

}
function showLoading(text) {
  loadingText.textContent = text;
  loading.style.display = "block";
}
function hideLoading() {
  loading.style.display = "none";
}
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.add("visible");
  hideLoading();
  btnAnalyse.disabled = false;
}
function hideError() {
  errorBox.classList.remove("visible");
}
function setBadge(el, verdict) {
  el.textContent = verdict || "—";
  el.className = `badge ${verdictClass(verdict)}`;
}
function isSupportedPlatform(url) {
  return SUPPORTED_PLATFORMS.some(p => url.includes(p));
}

async function checkBackend() {
  statusDot.className = "status-dot checking";
  statusText.textContent = "Checking backend...";
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
    if (res.ok) {
      statusDot.className = "status-dot online";
      statusText.textContent = "Backend connected";
      return true;
    }
  } catch(e) {}
  statusDot.className = "status-dot offline";
  statusText.textContent = "Backend offline — run: python backend.py";
  return false;
}


async function getCurrentTabUrl() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tabs[0]) return null;

  const url = tabs[0].url || "";

  // must be HTTP
  if (!url.startsWith("http")) return null;

  // must be supported platform
  if (!isSupportedPlatform(url)) return null;

  return url;
}

function displayResult(result) {
  hideLoading();
  resultCard.classList.add("visible");

  const cls = verdictClass(result.verdict);
  resultHeader.className    = `result-header ${cls}`;
  resultVerdict.textContent = result.verdict;
  resultVerdict.className   = `result-verdict ${cls}`;
  resultConf.textContent    = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
  resultReason.textContent  = result.reason || "";

  const ear = result.ear || {};
  setBadge(earBadge, ear.verdict);
  earDetail.textContent = ear.blink_rate != null
    ? `${ear.blink_rate} blinks/min · ${ear.blink_count} blinks`
    : "No face detected";

  const audio = result.audio || {};
  setBadge(audioBadge, audio.verdict);
  audioDetail.textContent = audio.fake_probability != null
    ? `Fake prob: ${(audio.fake_probability * 100).toFixed(1)}%`
    : "No audio data";
}


async function init() {
  hideError();
  hideLoading();
  resultCard.classList.remove("visible");
  noVideo.style.display = "none";
  videoInfo.classList.remove("visible");
  btnAnalyse.disabled = true;

  const backendOk = await checkBackend();

  currentPageUrl = await getCurrentTabUrl();

  if (!currentPageUrl) {
    noVideo.style.display = "block";
    return;
  }

  videoInfo.classList.add("visible");
  noVideo.style.display = "none";
  videoUrlEl.textContent = currentPageUrl;
  btnAnalyse.disabled = !backendOk;
}


btnAnalyse.addEventListener("click", async () => {
  if (!currentPageUrl) return;

  hideError();
  resultCard.classList.remove("visible");
  btnAnalyse.disabled = true;

  try {
    showLoading("Connecting to backend...");
    const ok = await checkBackend();
    if (!ok) throw new Error("Backend not running. Start it with: python backend.py");

    showLoading("Analysing video...");

    const res = await fetch(`${BACKEND_URL}/analyse_social`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page_url: currentPageUrl })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res.status}`);
    }

    showLoading("Running analysis...");
    const result = await res.json();
    displayResult(result);

  } catch(err) {
    showError(`Error: ${err.message}`);
  } finally {
    btnAnalyse.disabled = false;
  }
});


init();
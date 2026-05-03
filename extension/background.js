// DFD
// Background service worker 3

// Handles extension lifecycle events

// Log when extension is installed or updated
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === "install") {
    console.log("DFD installed. Start the backend: python backend.py");
  } else if (details.reason === "update") {
    console.log("DFD updated to version", chrome.runtime.getManifest().version);
  }
});

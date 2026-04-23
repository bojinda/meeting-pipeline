const controllerUrlEl = document.getElementById("controllerUrl");
const apiKeyEl = document.getElementById("apiKey");
const saveSettingsBtn = document.getElementById("saveSettings");
const testConnectionBtn = document.getElementById("testConnection");
const statusBoxEl = document.getElementById("statusBox");

function setStatus(message, extra = null) {
  const parts = [message];
  if (extra) {
    parts.push(typeof extra === "string" ? extra : JSON.stringify(extra, null, 2));
  }
  statusBoxEl.textContent = parts.join("\n\n");
}

function sendMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(new Error(err.message));
        return;
      }
      if (!response) {
        reject(new Error("No response from background script."));
        return;
      }
      if (!response.ok) {
        reject(new Error(response.error || "Unknown error"));
        return;
      }
      resolve(response.data ?? response);
    });
  });
}

async function loadSettings() {
  const config = await sendMessage({ type: "getStoredConfig" });
  controllerUrlEl.value = config.controllerUrl || "http://192.168.0.105:8765";
  apiKeyEl.value = config.apiKey || "";
}

saveSettingsBtn.addEventListener("click", async () => {
  try {
    await sendMessage({
      type: "saveSettings",
      controllerUrl: controllerUrlEl.value.trim(),
      apiKey: apiKeyEl.value.trim()
    });
    setStatus("Settings saved.");
  } catch (error) {
    setStatus("Failed to save settings.", error.message);
  }
});

testConnectionBtn.addEventListener("click", async () => {
  try {
    setStatus("Testing controller connection...");
    const config = await sendMessage({ type: "getStoredConfig" });

    if (!config.controllerUrl || !config.apiKey) {
      throw new Error("Controller URL and API key must be configured first.");
    }

    const result = await sendMessage({ type: "getCurrentLesson" });
    setStatus("Connection OK.", result);
  } catch (error) {
    setStatus("Connection test failed.", error.message);
  }
});

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadSettings();
    setStatus("Ready.");
  } catch (error) {
    setStatus("Failed to load settings.", error.message);
  }
});
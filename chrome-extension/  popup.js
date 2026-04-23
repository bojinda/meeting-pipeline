const tabTitleEl = document.getElementById("tabTitle");
const tabUrlEl = document.getElementById("tabUrl");
const controllerUrlEl = document.getElementById("controllerUrl");
const apiKeyEl = document.getElementById("apiKey");
const userLabelEl = document.getElementById("userLabel");
const sessionInfoEl = document.getElementById("sessionInfo");
const statusBoxEl = document.getElementById("statusBox");

const saveSettingsBtn = document.getElementById("saveSettings");
const refreshSessionBtn = document.getElementById("refreshSession");
const startLessonBtn = document.getElementById("startLesson");
const stopLessonBtn = document.getElementById("stopLesson");

let currentTab = null;

function setStatus(message, extra = null) {
  const parts = [message];
  if (extra) {
    parts.push(typeof extra === "string" ? extra : JSON.stringify(extra, null, 2));
  }
  statusBoxEl.textContent = parts.join("\n\n");
}

function renderSession(session) {
  if (!session) {
    sessionInfoEl.textContent = "No session loaded.";
    return;
  }

  sessionInfoEl.textContent = [
    `Session ID: ${session.session_id || "(none)"}`,
    `State: ${session.state || "(unknown)"}`,
    `Title: ${session.title || "(none)"}`,
    `Label: ${session.user_label || "(none)"}`,
    `Started: ${session.started_at || "(none)"}`,
    `Stopped: ${session.stopped_at || "(not stopped yet)"}`
  ].join("\n");
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

async function loadCurrentTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentTab = tab || null;

  tabTitleEl.textContent = currentTab?.title || "(no active tab)";
  tabUrlEl.textContent = currentTab?.url || "(no URL)";
}

async function loadStoredConfig() {
  const config = await sendMessage({ type: "getStoredConfig" });
  controllerUrlEl.value = config.controllerUrl || "http://192.168.0.105:8765";
  apiKeyEl.value = config.apiKey || "";
  renderSession(config.currentSession || null);
}

async function refreshCurrentSession() {
  setStatus("Refreshing current lesson session...");
  const data = await sendMessage({ type: "getCurrentLesson" });
  renderSession(data.session || null);
  setStatus("Current session refreshed.", data.session || null);
}

saveSettingsBtn.addEventListener("click", async () => {
  try {
    await sendMessage({
      type: "saveSettings",
      controllerUrl: controllerUrlEl.value,
      apiKey: apiKeyEl.value
    });
    setStatus("Settings saved.");
  } catch (error) {
    setStatus("Failed to save settings.", error.message);
  }
});

refreshSessionBtn.addEventListener("click", async () => {
  try {
    await refreshCurrentSession();
  } catch (error) {
    setStatus("Failed to refresh session.", error.message);
  }
});

startLessonBtn.addEventListener("click", async () => {
  try {
    if (!currentTab?.url) {
      throw new Error("No active tab URL available.");
    }

    setStatus("Starting lesson...");

    const data = await sendMessage({
      type: "lessonStart",
      title: currentTab.title || "Untitled tab",
      url: currentTab.url,
      userLabel: userLabelEl.value.trim()
    });

    renderSession(data.session || null);
    setStatus("Lesson started.", data.session || null);
  } catch (error) {
    setStatus("Failed to start lesson.", error.message);
  }
});

stopLessonBtn.addEventListener("click", async () => {
  try {
    setStatus("Stopping lesson...");
    const data = await sendMessage({ type: "lessonStop" });
    renderSession(data.session || null);
    setStatus("Lesson stopped.", data.session || null);
  } catch (error) {
    setStatus("Failed to stop lesson.", error.message);
  }
});

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadCurrentTab();
    await loadStoredConfig();
    setStatus("Ready.");
  } catch (error) {
    setStatus("Initialization failed.", error.message);
  }
});
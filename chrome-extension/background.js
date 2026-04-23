const STORAGE_KEYS = {
  controllerUrl: "controllerUrl",
  apiKey: "apiKey",
  currentSessionId: "currentSessionId",
  currentSession: "currentSession"
};

async function getStoredConfig() {
  const data = await chrome.storage.local.get([
    STORAGE_KEYS.controllerUrl,
    STORAGE_KEYS.apiKey,
    STORAGE_KEYS.currentSessionId,
    STORAGE_KEYS.currentSession
  ]);

  return {
    controllerUrl: (data[STORAGE_KEYS.controllerUrl] || "").trim(),
    apiKey: (data[STORAGE_KEYS.apiKey] || "").trim(),
    currentSessionId: data[STORAGE_KEYS.currentSessionId] || null,
    currentSession: data[STORAGE_KEYS.currentSession] || null
  };
}

async function saveStoredValues(values) {
  await chrome.storage.local.set(values);
}

function normalizeBaseUrl(url) {
  return url.replace(/\/+$/, "");
}

function isSessionActive(session) {
  if (!session) return false;
  return ["starting", "recording", "stopping"].includes(session.state);
}

async function updateBadge(session = null) {
  if (!session || !isSessionActive(session)) {
    await chrome.action.setBadgeText({ text: "" });
    await chrome.action.setTitle({
      title: "Lesson Controller"
    });
    return;
  }

  if (session.state === "stopping") {
    await chrome.action.setBadgeBackgroundColor({ color: "#d97706" });
    await chrome.action.setBadgeText({ text: "..." });
  } else {
    await chrome.action.setBadgeBackgroundColor({ color: "#b91c1c" });
    await chrome.action.setBadgeText({ text: "REC" });
  }

  const label = session.user_label || session.title || "Lesson active";
  await chrome.action.setTitle({
    title: `Lesson Controller — ${session.state}: ${label}`
  });
}

async function persistSession(session) {
  await saveStoredValues({
    [STORAGE_KEYS.currentSessionId]: session?.session_id || null,
    [STORAGE_KEYS.currentSession]: session || null
  });
  await updateBadge(session || null);
}

async function callController({ baseUrl, apiKey, path, method = "GET", body = null }) {
  if (!baseUrl) {
    throw new Error("Controller URL is not configured.");
  }
  if (!apiKey) {
    throw new Error("API key is not configured.");
  }

  const response = await fetch(`${normalizeBaseUrl(baseUrl)}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": apiKey
    },
    body: body ? JSON.stringify(body) : undefined
  });

  const text = await response.text();
  let data = null;

  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }

  if (!response.ok) {
    const detail = data?.detail ? JSON.stringify(data.detail) : text || response.statusText;
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }

  return data;
}

async function lessonStart(payload) {
  const config = await getStoredConfig();

  const data = await callController({
    baseUrl: config.controllerUrl,
    apiKey: config.apiKey,
    path: "/api/v1/lessons/start",
    method: "POST",
    body: payload
  });

  const session = data?.session || null;
  await persistSession(session);
  return data;
}

async function lessonStop() {
  const config = await getStoredConfig();

  const data = await callController({
    baseUrl: config.controllerUrl,
    apiKey: config.apiKey,
    path: "/api/v1/lessons/stop",
    method: "POST",
    body: config.currentSessionId
      ? { session_id: config.currentSessionId }
      : {}
  });

  const session = data?.session || null;
  await persistSession(session);
  return data;
}

async function getCurrentLesson() {
  const config = await getStoredConfig();

  const data = await callController({
    baseUrl: config.controllerUrl,
    apiKey: config.apiKey,
    path: "/api/v1/lessons/current",
    method: "GET"
  });

  const session = data?.session || null;
  await persistSession(session);
  return data;
}

async function refreshBadgeFromController() {
  try {
    const config = await getStoredConfig();

    if (!config.controllerUrl || !config.apiKey) {
      await updateBadge(null);
      return;
    }

    const data = await callController({
      baseUrl: config.controllerUrl,
      apiKey: config.apiKey,
      path: "/api/v1/lessons/current",
      method: "GET"
    });

    const session = data?.session || null;
    await persistSession(session);
  } catch {
    const config = await getStoredConfig();
    await updateBadge(config.currentSession || null);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  refreshBadgeFromController();
});

chrome.runtime.onStartup.addListener(() => {
  refreshBadgeFromController();
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      if (message.type === "saveSettings") {
        await saveStoredValues({
          [STORAGE_KEYS.controllerUrl]: (message.controllerUrl || "").trim(),
          [STORAGE_KEYS.apiKey]: (message.apiKey || "").trim()
        });
        await refreshBadgeFromController();
        sendResponse({ ok: true });
        return;
      }

      if (message.type === "lessonStart") {
        const result = await lessonStart({
          title: message.title,
          url: message.url,
          source: "chrome_extension",
          user_label: message.userLabel || null
        });
        sendResponse({ ok: true, data: result });
        return;
      }

      if (message.type === "lessonStop") {
        const result = await lessonStop();
        sendResponse({ ok: true, data: result });
        return;
      }

      if (message.type === "getCurrentLesson") {
        const result = await getCurrentLesson();
        sendResponse({ ok: true, data: result });
        return;
      }

      if (message.type === "getStoredConfig") {
        const config = await getStoredConfig();
        sendResponse({ ok: true, data: config });
        return;
      }

      if (message.type === "refreshBadge") {
        await refreshBadgeFromController();
        sendResponse({ ok: true });
        return;
      }

      sendResponse({ ok: false, error: `Unknown message type: ${message.type}` });
    } catch (error) {
      sendResponse({
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  })();

  return true;
});
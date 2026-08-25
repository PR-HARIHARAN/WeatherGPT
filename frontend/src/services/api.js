// WeatherGPT API service layer — talks to the FastAPI gateway (src/gateway/main.py).

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    return response.ok;
  } catch {
    return false;
  }
}

// Matches gateway POST /api/chat -> {headline, summary, session_id, language, tools_used, degraded}
export async function sendChatMessage({ message, language = 'auto', crop = 'other', sessionId, latitude, longitude }) {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, language, crop, session_id: sessionId, latitude, longitude }),
  });
  if (!response.ok) throw new Error(`Chat request failed (${response.status})`);
  return await response.json();
}

// Matches gateway POST /api/chat/stream (Server-Sent Events). Calls `onEvent`
// once per event: {type:'tool_start'|'tool_end'|'token'|'final'|'error', ...}.
// Returns a function that aborts the stream early (e.g. on unmount).
export function streamChatMessage({ message, language = 'auto', crop = 'other', sessionId, latitude, longitude }, onEvent) {
  const controller = new AbortController();

  (async () => {
    let response;
    try {
      response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, language, crop, session_id: sessionId, latitude, longitude }),
        signal: controller.signal,
      });
    } catch (err) {
      if (err.name !== 'AbortError') onEvent({ type: 'error', message: err.message });
      return;
    }

    if (!response.ok || !response.body) {
      onEvent({ type: 'error', message: `Stream request failed (${response.status})` });
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line; each frame's payload is
        // one or more "data: <json>" lines (we only ever send one per frame).
        const frames = buffer.split('\n\n');
        buffer = frames.pop(); // last chunk may be incomplete, keep it buffered

        for (const frame of frames) {
          const line = frame.split('\n').find((l) => l.startsWith('data: '));
          if (!line) continue;
          try {
            onEvent(JSON.parse(line.slice(6)));
          } catch (e) {
            console.error('Failed to parse SSE frame', e, line);
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') onEvent({ type: 'error', message: err.message });
    }
  })();

  return () => controller.abort();
}

// Matches gateway POST /api/chat/voice (multipart: file, language, crop, session_id)
export async function sendVoiceAudio({ audioBlob, language = 'auto', crop = 'other', sessionId, latitude, longitude }) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'voice.webm');
  formData.append('language', language);
  formData.append('crop', crop);
  if (sessionId) formData.append('session_id', sessionId);
  if (latitude != null) formData.append('latitude', latitude);
  if (longitude != null) formData.append('longitude', longitude);

  const response = await fetch(`${API_BASE_URL}/api/chat/voice`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error(`Voice request failed (${response.status})`);
  return await response.json();
}

// Matches gateway POST /api/tts -> audio/mpeg bytes. Returns a blob: URL the
// caller can hand to an <audio> element, or null if TTS is unavailable (the
// caller should fall back to window.speechSynthesis in that case).
export async function fetchTTS({ text, language = 'en' }) {
  if (!text) return null;
  try {
    const response = await fetch(`${API_BASE_URL}/api/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language }),
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) return null;
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

// Matches the weather_backend app mounted at /backend (gateway/main.py).
export async function fetchCurrentWeather(location) {
  const response = await fetch(
    `${API_BASE_URL}/backend/api/v1/weather/current?location=${encodeURIComponent(location)}`
  );
  if (!response.ok) throw new Error(`Current weather request failed (${response.status})`);
  return await response.json();
}

export async function fetchForecast(location, days = 5) {
  const response = await fetch(
    `${API_BASE_URL}/backend/api/v1/weather/forecast?location=${encodeURIComponent(location)}&days=${days}`
  );
  if (!response.ok) throw new Error(`Forecast request failed (${response.status})`);
  return await response.json();
}

// Matches gateway GET /alerts/recent?limit= — non-expired alerts across all districts.
export async function fetchRecentAlerts(limit = 50) {
  try {
    const response = await fetch(`${API_BASE_URL}/alerts/recent?limit=${limit}`);
    if (!response.ok) throw new Error('recent alerts request failed');
    const data = await response.json();
    return (Array.isArray(data) ? data : []).map(normalizeAlert).filter(Boolean);
  } catch (err) {
    console.warn('Recent alerts unavailable:', err.message);
    return [];
  }
}

export function normalizeAlert(raw) {
  if (!raw) return null;
  return {
    id: raw.alert_id || raw.id,
    district: raw.district,
    title: raw.title,
    severity: raw.severity || 'high',
    advice: raw.action || raw.advice || raw.message || '',
    validUntil: raw.valid_until || raw.validUntil,
  };
}

const SEVERITY_RANK = { critical: 3, high: 2, low: 1, informational: 0 };

export function pickMostSevereAlert(alerts) {
  if (!Array.isArray(alerts) || alerts.length === 0) return null;
  return [...alerts].sort(
    (a, b) => (SEVERITY_RANK[b.severity] ?? 0) - (SEVERITY_RANK[a.severity] ?? 0)
  )[0];
}

// Matches gateway GET /api/v1/alerts/active?district=
export async function fetchActiveAlerts(district) {
  if (!district) return [];
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/v1/alerts/active?district=${encodeURIComponent(district)}`
    );
    if (!response.ok) throw new Error('active alerts request failed');
    const data = await response.json();
    return (Array.isArray(data) ? data : []).map(normalizeAlert).filter(Boolean);
  } catch (err) {
    console.warn('Active alerts unavailable:', err.message);
    return [];
  }
}

// Matches gateway WS /ws/alerts?district=
export function subscribeToDisasterAlerts(onAlertReceived, district = '') {
  let ws;
  let closed = false;
  const url = district
    ? `${WS_BASE_URL}/ws/alerts?district=${encodeURIComponent(district.toLowerCase())}`
    : `${WS_BASE_URL}/ws/alerts`;
  try {
    ws = new WebSocket(url);
    ws.onmessage = (event) => {
      try {
        const raw = JSON.parse(event.data);
        if (raw.severity === 'informational') return;
        const alert = normalizeAlert(raw);
        if (alert) onAlertReceived(alert);
      } catch (e) {
        console.error('Failed to parse WS alert', e);
      }
    };
    ws.onclose = () => {
      if (!closed) setTimeout(() => subscribeToDisasterAlerts(onAlertReceived, district), 5000);
    };
  } catch {
    console.warn('WebSocket unavailable, live alert push disabled');
  }
  return () => {
    closed = true;
    if (ws) ws.close();
  };
}

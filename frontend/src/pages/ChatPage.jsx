import React, { useState, useEffect, useRef, useCallback } from 'react';
import ChatContainer from '../components/ChatContainer';
import VoiceInput from '../components/VoiceInput';
import {
  streamChatMessage,
  sendVoiceAudio,
  fetchTTS,
} from '../services/api';
import { stripMarkdownForSpeech } from '../utils/stripMarkdown';

const WELCOME = {
  en: "Hi, I'm WeatherGPT. Ask me about the weather, safety warnings, or farming advice for any place in India — in your own language.",
  hi: 'नमस्ते, मैं WeatherGPT हूँ। भारत में किसी भी जगह के मौसम, सुरक्षा चेतावनी या खेती सलाह के बारे में अपनी भाषा में पूछें।',
  ta: 'வணக்கம், நான் WeatherGPT. இந்தியாவில் எந்த இடத்திற்கும் வானிலை, பாதுகாப்பு எச்சரிக்கை அல்லது வேளாண் ஆலோசனையை உங்கள் மொழியில் கேளுங்கள்.',
  te: 'నమస్తే, నేను WeatherGPT. భారతదేశంలో ఏ ప్రాంతానికైనా వాతావరణం, భద్రతా హెచ్చరికలు లేదా వ్యవసాయ సలహా గురించి మీ భాషలో అడగండి.',
  kn: 'ನಮಸ್ಕಾರ, ನಾನು WeatherGPT. ಭಾರತದ ಯಾವುದೇ ಸ್ಥಳದ ಹವಾಮಾನ, ಸುರಕ್ಷತಾ ಎಚ್ಚರಿಕೆಗಳು ಅಥವಾ ಕೃಷಿ ಸಲಹೆಯನ್ನು ನಿಮ್ಮ ಭಾಷೆಯಲ್ಲಿ ಕೇಳಿ.',
  bn: 'নমস্কার, আমি WeatherGPT। ভারতের যে কোনো জায়গার আবহাওয়া, সতর্কতা বা কৃষি পরামর্শ সম্পর্কে আপনার ভাষায় জিজ্ঞাসা করুন।',
};

const SPEECH_VOICE_LANG = { hi: 'hi-IN', ta: 'ta-IN', te: 'te-IN', kn: 'kn-IN', bn: 'bn-IN', en: 'en-IN' };

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Asks the browser once per page load and caches the result for every chat
// request after that. Never blocks sending a message: on denial, timeout, or
// an unsupported browser it resolves to nulls and the agent falls back to
// asking the user for a place name instead.
function getBrowserLocation() {
  return new Promise((resolve) => {
    if (!('geolocation' in navigator)) {
      resolve({ latitude: null, longitude: null });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ latitude: pos.coords.latitude, longitude: pos.coords.longitude }),
      () => resolve({ latitude: null, longitude: null }),
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 300000 }
    );
  });
}

export default function ChatPage({ language, onLastDistrict }) {
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    { sender: 'agent', text: WELCOME.en, time: timeNow() },
  ]);
  const sessionIdRef = useRef(null);
  const abortStreamRef = useRef(null);
  const audioElRef = useRef(null);
  const locationRef = useRef({ latitude: null, longitude: null });

  useEffect(() => {
    getBrowserLocation().then((loc) => {
      locationRef.current = loc;
    });
  }, []);

  useEffect(() => () => abortStreamRef.current?.(), []);

  // Speaks text in whatever language was actually detected/used for the reply,
  // never the dropdown value: backend TTS (gTTS) covers Indian languages far
  // more reliably than the browser's installed SpeechSynthesis voices, and it's
  // tried first with speechSynthesis only as an offline fallback.
  const speakText = useCallback(async (textToSpeak, langCode) => {
    if (!textToSpeak) return;
    const lang = langCode || 'en';
    // Speak the plain-language content, not literal markdown syntax
    // ("asterisk asterisk", digits from a table pipe, etc).
    const spoken = stripMarkdownForSpeech(textToSpeak);
    if (!spoken) return;

    const audioUrl = await fetchTTS({ text: spoken, language: lang });
    if (audioUrl) {
      if (audioElRef.current) {
        audioElRef.current.pause();
        URL.revokeObjectURL(audioElRef.current.src);
      }
      const audio = new Audio(audioUrl);
      audio.playbackRate = 1.15;
      audioElRef.current = audio;
      audio.play().catch(() => {});
      return;
    }

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(spoken);
      utterance.lang = SPEECH_VOICE_LANG[lang] || 'en-IN';
      utterance.rate = 1.15;
      window.speechSynthesis.speak(utterance);
    }
  }, []);

  const appendError = (text) => {
    setMessages((prev) => [...prev, { sender: 'agent', text, error: true, time: timeNow() }]);
  };

  const streamReply = (queryText) => {
    setIsLoading(true);

    setMessages((prev) => [
      ...prev,
      { sender: 'agent', text: '', toolCalls: [], streaming: true, time: timeNow() },
    ]);

    const patchLastAgentMessage = (patch) => {
      setMessages((prev) => {
        const next = [...prev];
        const idx = next.length - 1;
        next[idx] = { ...next[idx], ...(typeof patch === 'function' ? patch(next[idx]) : patch) };
        return next;
      });
    };

    abortStreamRef.current = streamChatMessage(
      {
        message: queryText,
        language,
        sessionId: sessionIdRef.current,
        latitude: locationRef.current.latitude,
        longitude: locationRef.current.longitude,
      },
      (event) => {
        if (event.type === 'tool_start') {
          patchLastAgentMessage((msg) => ({
            toolCalls: [...msg.toolCalls, { tool: event.tool, status: 'running' }],
          }));
        } else if (event.type === 'tool_end') {
          patchLastAgentMessage((msg) => ({
            toolCalls: msg.toolCalls.map((t) =>
              t.tool === event.tool && t.status === 'running'
                ? { ...t, status: 'done', degraded: event.degraded }
                : t
            ),
          }));
        } else if (event.type === 'token') {
          patchLastAgentMessage((msg) => ({ text: msg.text + event.text }));
        } else if (event.type === 'final') {
          const replyLang = event.detected_language || event.language || 'en';
          patchLastAgentMessage({
            text: event.text,
            degraded: event.degraded,
            streaming: false,
            language: replyLang,
          });
          if (event.session_id) sessionIdRef.current = event.session_id;
          speakText(event.text, replyLang);
          setIsLoading(false);
        } else if (event.type === 'error') {
          patchLastAgentMessage({ text: event.message, error: true, streaming: false });
          setIsLoading(false);
        }
      }
    );
  };

  const handleSendText = (queryText) => {
    setMessages((prev) => [...prev, { sender: 'user', text: queryText, time: timeNow() }]);
    streamReply(queryText);
  };

  const handleSendVoice = async (audioBlob) => {
    if (!audioBlob) return;
    setIsLoading(true);
    try {
      const result = await sendVoiceAudio({
        audioBlob,
        language,
        sessionId: sessionIdRef.current,
        latitude: locationRef.current.latitude,
        longitude: locationRef.current.longitude,
      });
      const replyLang = result.detected_language || result.language || 'en';
      setMessages((prev) => [
        ...prev,
        { sender: 'user', text: result.userTranscript || '(voice message)', time: timeNow() },
        {
          sender: 'agent',
          text: result.summary,
          toolCalls: (result.tools_used || []).map((tool) => ({ tool, status: 'done' })),
          degraded: result.degraded,
          language: replyLang,
          time: timeNow(),
        },
      ]);
      if (result.session_id) sessionIdRef.current = result.session_id;
      speakText(result.summary, replyLang);
    } catch (err) {
      appendError("I couldn't reach the weather service. Make sure the backend is running and try again.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-page">
      <ChatContainer
        messages={messages}
        isLoading={isLoading}
        onPlayAudio={(text, lang) => speakText(text, lang)}
      />

      <VoiceInput
        onSendText={handleSendText}
        onSendVoice={handleSendVoice}
        isLoading={isLoading}
        language={language}
      />
    </div>
  );
}

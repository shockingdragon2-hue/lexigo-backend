// FIXED server.js (Azure only, ElevenLabs removed)

import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import sdk from "microsoft-cognitiveservices-speech-sdk";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const AZURE_KEY = process.env.AZURE_SPEECH_KEY;
const AZURE_REGION = process.env.AZURE_SPEECH_REGION;

function normalizeLanguage(lang) {
  const l = String(lang || "").toLowerCase();
  if (l.includes("spanish") || l === "es") return "spanish";
  if (l.includes("basque") || l === "eu") return "basque";
  if (l.includes("portuguese") || l === "pt") return "portuguese";
  if (l.includes("french") || l === "fr") return "french";
  if (l.includes("german") || l === "de") return "german";
  if (l.includes("italian") || l === "it") return "italian";
  return "english";
}

function getVoice(language) {
  const lang = normalizeLanguage(language);
  switch (lang) {
    case "basque":
      return "eu-ES-AinhoaNeural";
    case "spanish":
      return "es-ES-ElviraNeural";
    case "portuguese":
      return "pt-BR-FranciscaNeural";
    case "french":
      return "fr-FR-DeniseNeural";
    case "german":
      return "de-DE-KatjaNeural";
    case "italian":
      return "it-IT-ElsaNeural";
    default:
      return "en-US-JennyNeural";
  }
}

app.post("/tts", async (req, res) => {
  try {
    const { text, language } = req.body;
    const voice = getVoice(language);

    const speechConfig = sdk.SpeechConfig.fromSubscription(
      AZURE_KEY,
      AZURE_REGION
    );
    speechConfig.speechSynthesisVoiceName = voice;

    const synthesizer = new sdk.SpeechSynthesizer(speechConfig);

    synthesizer.speakTextAsync(
      text,
      (result) => {
        const audioBuffer = Buffer.from(result.audioData);
        res.setHeader("Content-Type", "audio/mpeg");
        res.send(audioBuffer);
      },
      (err) => {
        console.error(err);
        res.status(500).send("TTS error");
      }
    );
  } catch (err) {
    console.error(err);
    res.status(500).send("Server error");
  }
});

app.listen(3000, () => console.log("Server running on port 3000"));

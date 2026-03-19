// PATCHED server.js (Azure only, ElevenLabs removed safely)

import express from "express";
import dotenv from "dotenv";
import OpenAI from "openai";
import cors from "cors";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import sdk from "microsoft-cognitiveservices-speech-sdk";

dotenv.config();

const app = express();
app.set("trust proxy", true);
app.use(cors());
app.use(express.json({ limit: "25mb" }));

const ROOT_DIR = process.cwd();
const AUDIO_DIR = path.join(ROOT_DIR, "audio");

if (!fs.existsSync(AUDIO_DIR)) {
  fs.mkdirSync(AUDIO_DIR, { recursive: true });
}

app.use("/audio", express.static(AUDIO_DIR));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// 🔥 FORCE AZURE ONLY
function azureVoiceForLanguage(language) {
  const l = String(language || "").toLowerCase();

  if (l.includes("spanish")) return "es-ES-ElviraNeural";
  if (l.includes("basque")) return "eu-ES-AinhoaNeural";
  if (l.includes("portuguese")) return "pt-BR-FranciscaNeural";
  if (l.includes("french")) return "fr-FR-DeniseNeural";
  if (l.includes("german")) return "de-DE-KatjaNeural";
  if (l.includes("italian")) return "it-IT-ElsaNeural";

  return "en-US-JennyNeural";
}

// 🔥 TTS FUNCTION (keeps your URL system)
app.post("/tts", async (req, res) => {
  try {
    const { text, language } = req.body;

    const cleanText = String(text || "").trim();
    const voice = azureVoiceForLanguage(language);

    const hash = crypto
      .createHash("sha256")
      .update(cleanText + voice)
      .digest("hex");

    const fileName = `${hash}.mp3`;
    const filePath = path.join(AUDIO_DIR, fileName);

    if (fs.existsSync(filePath)) {
      return res.json({
        audioUrl: `/audio/${fileName}`,
      });
    }

    const speechConfig = sdk.SpeechConfig.fromSubscription(
      process.env.AZURE_SPEECH_KEY,
      process.env.AZURE_SPEECH_REGION
    );

    speechConfig.speechSynthesisVoiceName = voice;

    const synthesizer = new sdk.SpeechSynthesizer(speechConfig);

    await new Promise((resolve, reject) => {
      synthesizer.speakTextAsync(
        cleanText,
        (result) => {
          fs.writeFileSync(filePath, Buffer.from(result.audioData));
          resolve();
        },
        (err) => reject(err)
      );
    });

    res.json({
      audioUrl: `/audio/${fileName}`,
    });
  } catch (err) {
    console.error(err);
    res.status(500).send("TTS error");
  }
});

// KEEP YOUR ORIGINAL generateSet ROUTE (IMPORTANT)
// 👉 DO NOT TOUCH IT

app.listen(3000, () => console.log("Server running"));

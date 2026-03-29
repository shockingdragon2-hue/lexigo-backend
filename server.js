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

ensureDir(AUDIO_DIR);
app.use(
  "/audio",
  express.static(AUDIO_DIR, {
    maxAge: "30d",
    immutable: true,
  }),
);

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const OPENAI_TTS_MODEL = process.env.OPENAI_TTS_MODEL || "gpt-4o-mini-tts";
const DEFAULT_OPENAI_VOICE = process.env.OPENAI_TTS_VOICE || "alloy";

/* -------------------------------------------------------------------------- */
/*                                    Utils                                   */
/* -------------------------------------------------------------------------- */

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function sha256(input) {
  return crypto.createHash("sha256").update(input).digest("hex");
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function sanitizeText(text, maxLen = 4000) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .trim()
    .slice(0, maxLen);
}

function sanitizeTtsText(text) {
  return sanitizeText(text, 4000);
}

function sanitizeShortLabel(text, maxLen = 200) {
  return sanitizeText(text, maxLen);
}

function safeJsonParse(text, fallback = null) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function escapeSsml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* -------------------------------------------------------------------------- */
/*                              Language Handling                             */
/* -------------------------------------------------------------------------- */

function normalizeLanguage(language) {
  const raw = String(language || "").trim().toLowerCase();

  if (["eu", "euskara", "euskaraz", "basque"].includes(raw)) return "basque";
  if (["es", "español", "espanol", "spanish", "castellano"].includes(raw)) return "spanish";
  if (["en", "english", "inglés", "ingles"].includes(raw)) return "english";
  if (["pt", "portuguese", "português", "portugues"].includes(raw)) return "portuguese";
  if (["fr", "french", "français", "francais"].includes(raw)) return "french";
  if (["de", "german", "deutsch"].includes(raw)) return "german";
  if (["it", "italian", "italiano"].includes(raw)) return "italian";

  return raw || "english";
}

function azureVoiceForLanguage(language) {
  const normalized = normalizeLanguage(language);

  switch (normalized) {
    case "basque":
      return "eu-ES-AinhoaNeural";
    case "spanish":
      return "es-ES-ElviraNeural";
    case "english":
      return "en-US-JennyNeural";
    case "portuguese":
      return "pt-PT-RaquelNeural";
    case "french":
      return "fr-FR-DeniseNeural";
    case "german":
      return "de-DE-KatjaNeural";
    case "italian":
      return "it-IT-ElsaNeural";
    default:
      return null;
  }
}

function azureLangCode(language) {
  const normalized = normalizeLanguage(language);

  switch (normalized) {
    case "basque":
      return "eu-ES";
    case "spanish":
      return "es-ES";
    case "english":
      return "en-US";
    case "portuguese":
      return "pt-PT";
    case "french":
      return "fr-FR";
    case "german":
      return "de-DE";
    case "italian":
      return "it-IT";
    default:
      return "en-US";
  }
}

function openAiVoiceForLanguage(language) {
  const normalized = normalizeLanguage(language);

  switch (normalized) {
    case "spanish":
      return "sage";
    case "basque":
      return "alloy";
    case "english":
      return "alloy";
    case "portuguese":
      return "nova";
    case "french":
      return "shimmer";
    case "german":
      return "echo";
    case "italian":
      return "fable";
    default:
      return DEFAULT_OPENAI_VOICE;
  }
}

function isAzureConfigured() {
  return Boolean(process.env.AZURE_SPEECH_KEY && process.env.AZURE_SPEECH_REGION);
}

function getBaseUrl(req) {
  return process.env.PUBLIC_BASE_URL || `${req.protocol}://${req.get("host")}`;
}

function makeAudioUrl(req, fileName) {
  return `${getBaseUrl(req)}/audio/${fileName}`;
}

async function mapWithConcurrency(items, limit, mapper) {
  const safeLimit = Math.max(1, Math.min(Number(limit) || 1, 6));
  const results = new Array(items.length);
  let nextIndex = 0;

  async function worker() {
    while (nextIndex < items.length) {
      const currentIndex = nextIndex++;
      results[currentIndex] = await mapper(items[currentIndex], currentIndex);
    }
  }

  const workers = Array.from(
    { length: Math.min(safeLimit, items.length) },
    () => worker(),
  );

  await Promise.all(workers);
  return results;
}

/* -------------------------------------------------------------------------- */
/*                          Support-Language Prompts                          */
/* -------------------------------------------------------------------------- */

function promptTextForSupportLanguage(key, supportLanguage, variables = {}) {
  const lang = normalizeLanguage(supportLanguage);
  const targetLanguage = sanitizeShortLabel(
    variables.targetLanguage || "the target language",
    50,
  );
  const term = sanitizeShortLabel(variables.term || "", 250);
  const meaning = sanitizeShortLabel(variables.meaning || "", 250);

  const map = {
    english: {
      listenThinkMeaning: "Listen and think about the meaning.",
      whatDoesThisMean: "What does this mean?",
      howDoYouSayThisIn: `How do you say this in ${targetLanguage}?`,
      tryTranslateSentence: "Try to translate this sentence.",
      sayMeaningOutLoud: "Say what this means out loud.",
      sayWordOutLoud: `How do you say this in ${targetLanguage}? Say it out loud.`,
      sayMissingWordOutLoud: `Listen, then say the missing ${targetLanguage} word out loud.`,
      answerIs: `The answer is ${term}.`,
      meansIs: `${term} means ${meaning}.`,
      xInTargetIsY: `${meaning} in ${targetLanguage} is ${term}.`,
    },
    spanish: {
      listenThinkMeaning: "Escucha y piensa en el significado.",
      whatDoesThisMean: "¿Qué significa esto?",
      howDoYouSayThisIn: `¿Cómo se dice esto en ${targetLanguage}?`,
      tryTranslateSentence: "Intenta traducir esta frase.",
      sayMeaningOutLoud: "Di en voz alta lo que significa.",
      sayWordOutLoud: `¿Cómo se dice esto en ${targetLanguage}? Dilo en voz alta.`,
      sayMissingWordOutLoud: `Escucha y luego di en voz alta la palabra que falta en ${targetLanguage}.`,
      answerIs: `La respuesta es ${term}.`,
      meansIs: `${term} significa ${meaning}.`,
      xInTargetIsY: `${meaning} en ${targetLanguage} es ${term}.`,
    },
    basque: {
      listenThinkMeaning: "Entzun eta pentsatu esanahian.",
      whatDoesThisMean: "Zer esan nahi du honek?",
      howDoYouSayThisIn: `Nola esaten da hau ${targetLanguage} hizkuntzan?`,
      tryTranslateSentence: "Saiatu esaldi hau itzultzen.",
      sayMeaningOutLoud: "Esan ozen zer esan nahi duen.",
      sayWordOutLoud: `Nola esaten da hau ${targetLanguage} hizkuntzan? Esan ozen.`,
      sayMissingWordOutLoud: `Entzun, eta gero esan ozen falta den ${targetLanguage} hitza.`,
      answerIs: `Erantzuna ${term} da.`,
      meansIs: `${term} hitzak ${meaning} esan nahi du.`,
      xInTargetIsY: `${meaning}, ${targetLanguage} hizkuntzan, ${term} da.`,
    },
  };

  const selected = map[lang] || map.english;
  return selected[key] || map.english[key] || "";
}

/* -------------------------------------------------------------------------- */
/*                               Difficulty Rules                             */
/* -------------------------------------------------------------------------- */

function normalizeDifficulty(difficulty) {
  const raw = String(difficulty || "").trim().toLowerCase();

  if (raw === "beginner") return "beginner";
  if (raw === "intermediate") return "intermediate";
  if (raw === "advanced") return "advanced";

  return "beginner";
}

function difficultyGuidance(difficulty) {
  const normalized = normalizeDifficulty(difficulty);

  if (normalized === "beginner") {
    return `
Difficulty level: beginner

Beginner rules:
- Use very common, concrete, high-frequency vocabulary.
- Prefer words useful for everyday life.
- Keep meanings simple and clear.
- Example sentences must be very short and easy.
- Prefer present tense and simple sentence patterns.
- Avoid rare grammar, idioms, slang, figurative language, or long clauses.
- Keep most example sentences around 3 to 6 words when possible.
- Do not make the examples too hard for a true beginner.
`;
  }

  if (normalized === "intermediate") {
    return `
Difficulty level: intermediate

Intermediate rules:
- Use common but slightly broader vocabulary.
- Example sentences should still be natural and clear.
- Some variation in tense is okay.
- Moderate sentence length is okay.
- Avoid highly literary, technical, or obscure phrasing.
`;
  }

  return `
Difficulty level: advanced

Advanced rules:
- You may use more natural and varied vocabulary.
- Example sentences can be longer and more expressive.
- More complex grammar is allowed when still natural.
- Avoid being overly literary or unnecessarily obscure.
`;
}

/* -------------------------------------------------------------------------- */
/*                              Set Generation                                */
/* -------------------------------------------------------------------------- */

function buildGenerateInstructions({
  mode,
  difficulty,
  targetLanguage,
  baseLanguage,
  desiredCount,
}) {
  return `
You are helping build a language-learning audio app.

Return only structured data that matches the schema.

Target language: ${targetLanguage || "unspecified"}
Base/support language for meanings: ${baseLanguage || "English"}
Mode: ${mode}
Desired vocabulary count: ${desiredCount || "use best judgment"}

${difficultyGuidance(difficulty)}

General requirements:
- Output must fit the requested difficulty.
- Keep vocabulary useful and learner-friendly.
- Avoid explicit, unsafe, political, hateful, scary, medical, or disturbing content.
- Prefer words and phrases useful in daily life.
- Meanings should be concise and easy to understand in the base language.
- Each item must include:
  - term
  - meaning
  - guessedWordType
  - promptFamily
  - 2 short safeExampleSentences
  - 2 matching exampleTranslations

Important sentence rules:
- safeExampleSentences must be in the TARGET language.
- exampleTranslations must be in the BASE language.
- Each exampleTranslation must be a full, natural translation of its matching sentence.
- Do not shorten a sentence translation into a single word or partial phrase.
- Include the whole meaning of the sentence, including helpers, objects, and endings.
- The term should appear naturally in at least one example sentence when possible.
- Do not use overly advanced sentence structures for beginner mode.
- Keep punctuation simple.

For MANUAL vocab mode:
- keep the original term exactly as provided
- do not replace the learner's supplied term with another synonym
- if the supplied term is already a phrase, keep it as a phrase

For PROMPT mode:
- generate exactly the requested number of useful vocabulary items when possible

For IMAGE mode:
- use the image and notes together
- prioritize visible objects, simple actions, locations, colors, and basic descriptors

Prompt family:
- choose exactly one of:
  - recall
  - completion
  - production
`;
}

function buildGenerateInput({
  mode,
  vocab,
  prompt,
  notes,
  desiredCount,
  targetLanguage,
  baseLanguage,
  difficulty,
}) {
  const chunks = [
    `Mode: ${mode || "manual"}`,
    `Target language: ${targetLanguage || ""}`,
    `Base language: ${baseLanguage || "English"}`,
    `Difficulty: ${normalizeDifficulty(difficulty)}`,
    `Desired count: ${desiredCount || ""}`,
  ];

  if (mode === "manual" && Array.isArray(vocab) && vocab.length) {
    chunks.push(
      `Vocabulary list:\n${vocab
        .map((word, i) => `${i + 1}. ${String(word || "").trim()}`)
        .join("\n")}`,
    );
  }

  if (mode === "prompt" || mode === "image") {
    if (prompt) chunks.push(`Prompt/theme:\n${String(prompt).trim()}`);
    if (notes) chunks.push(`Notes:\n${String(notes).trim()}`);
  }

  return chunks.join("\n\n");
}

const generationSchema = {
  name: "language_audio_set",
  schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      items: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            term: { type: "string" },
            meaning: { type: "string" },
            guessedWordType: {
              type: "string",
              enum: ["noun", "verb", "adjective", "adverb", "phrase", "other"],
            },
            promptFamily: {
              type: "string",
              enum: ["recall", "completion", "production"],
            },
            safeExampleSentences: {
              type: "array",
              minItems: 2,
              maxItems: 2,
              items: { type: "string" },
            },
            exampleTranslations: {
              type: "array",
              minItems: 2,
              maxItems: 2,
              items: { type: "string" },
            },
          },
          required: [
            "term",
            "meaning",
            "guessedWordType",
            "promptFamily",
            "safeExampleSentences",
            "exampleTranslations",
          ],
        },
      },
    },
    required: ["items"],
  },
};

/* -------------------------------------------------------------------------- */
/*                                   TTS                                      */
/* -------------------------------------------------------------------------- */

function getTtsCacheKey({ text, language, speed = 1.0, voice = "" }) {
  return sha256(
    JSON.stringify({
      text: sanitizeTtsText(text),
      language: normalizeLanguage(language),
      speed: clampNumber(speed, 0.5, 2.0, 1.0),
      voice: String(voice || "").trim(),
    }),
  );
}

async function synthesizeAzureToFile({ text, language, outputPath, speed = 1.0 }) {
  return new Promise((resolve, reject) => {
    const voiceName = azureVoiceForLanguage(language);

    if (!voiceName || !isAzureConfigured()) {
      return reject(new Error("Azure speech not configured for this language"));
    }

    const speechConfig = sdk.SpeechConfig.fromSubscription(
      process.env.AZURE_SPEECH_KEY,
      process.env.AZURE_SPEECH_REGION,
    );

    speechConfig.speechSynthesisVoiceName = voiceName;
    speechConfig.speechSynthesisOutputFormat =
      sdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3;

    const audioConfig = sdk.AudioConfig.fromAudioFileOutput(outputPath);
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig);

    const safeRate = clampNumber(speed, 0.5, 2.0, 1.0);
    const ratePercent = Math.round((safeRate - 1) * 100);
    const cleanText = escapeSsml(sanitizeTtsText(text));
    const langCode = azureLangCode(language);

    const ssml = `
      <speak version="1.0" xml:lang="${langCode}">
        <voice name="${voiceName}">
          <prosody rate="${ratePercent >= 0 ? "+" : ""}${ratePercent}%">
            ${cleanText}
          </prosody>
        </voice>
      </speak>
    `;

    synthesizer.speakSsmlAsync(
      ssml,
      (result) => {
        synthesizer.close();

        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          resolve({ ok: true, provider: "azure", voice: voiceName });
        } else {
          reject(new Error("Azure synthesis failed"));
        }
      },
      (err) => {
        synthesizer.close();
        reject(err);
      },
    );
  });
}

async function synthesizeOpenAIToFile({
  text,
  outputPath,
  voice = DEFAULT_OPENAI_VOICE,
  language = "english",
}) {
  const cleanText = sanitizeTtsText(text);
  const chosenVoice = voice || openAiVoiceForLanguage(language) || DEFAULT_OPENAI_VOICE;

  const mp3 = await client.audio.speech.create({
    model: OPENAI_TTS_MODEL,
    voice: chosenVoice,
    input: cleanText,
  });

  const buffer = Buffer.from(await mp3.arrayBuffer());
  fs.writeFileSync(outputPath, buffer);

  return { ok: true, provider: "openai", voice: chosenVoice };
}

async function getOrCreateTtsFile({
  req,
  text,
  language,
  speed = 1.0,
  voice = "",
}) {
  const cleanText = sanitizeTtsText(text);
  const normalizedLanguage = normalizeLanguage(language);
  const cleanSpeed = clampNumber(speed, 0.5, 2.0, 1.0);

  if (!cleanText) {
    throw new Error("Missing TTS text");
  }

  const cacheKey = getTtsCacheKey({
    text: cleanText,
    language: normalizedLanguage,
    speed: cleanSpeed,
    voice,
  });

  const fileName = `${cacheKey}.mp3`;
  const outputPath = path.join(AUDIO_DIR, fileName);

  if (fs.existsSync(outputPath)) {
    return {
      ok: true,
      cached: true,
      language: normalizedLanguage,
      provider: "cache",
      voice:
        voice ||
        azureVoiceForLanguage(normalizedLanguage) ||
        openAiVoiceForLanguage(normalizedLanguage) ||
        DEFAULT_OPENAI_VOICE,
      text: cleanText,
      audioUrl: makeAudioUrl(req, fileName),
    };
  }

  let providerInfo;

  try {
    providerInfo = await synthesizeAzureToFile({
      text: cleanText,
      language: normalizedLanguage,
      outputPath,
      speed: cleanSpeed,
    });
  } catch (azureError) {
    console.warn("Azure TTS failed, falling back to OpenAI:", azureError.message);

    providerInfo = await synthesizeOpenAIToFile({
      text: cleanText,
      outputPath,
      voice,
      language: normalizedLanguage,
    });
  }

  return {
    ok: true,
    cached: false,
    language: normalizedLanguage,
    provider: providerInfo.provider,
    voice: providerInfo.voice,
    text: cleanText,
    audioUrl: makeAudioUrl(req, fileName),
  };
}

/* -------------------------------------------------------------------------- */
/*                          Session Prompt Construction                        */
/* -------------------------------------------------------------------------- */

function normalizePromptFamily(promptFamily) {
  const raw = String(promptFamily || "").trim().toLowerCase();
  if (["recall", "completion", "production"].includes(raw)) return raw;
  return "recall";
}

function buildPromptText({
  practiceMode = "standard",
  promptFamily = "recall",
  targetLanguage = "Basque",
  baseLanguage = "English",
  sentence = "",
}) {
  const mode = String(practiceMode || "").trim().toLowerCase();
  const family = normalizePromptFamily(promptFamily);
  const cleanTarget = sanitizeShortLabel(targetLanguage, 50);
  const cleanSentence = sanitizeShortLabel(sentence, 250);

  if (mode === "speech") {
    switch (family) {
      case "completion":
        if (cleanSentence) {
          return promptTextForSupportLanguage(
            "sayMissingWordOutLoud",
            baseLanguage,
            { targetLanguage: cleanTarget },
          );
        }
        return promptTextForSupportLanguage(
          "sayWordOutLoud",
          baseLanguage,
          { targetLanguage: cleanTarget },
        );
      case "production":
        return promptTextForSupportLanguage(
          "sayWordOutLoud",
          baseLanguage,
          { targetLanguage: cleanTarget },
        );
      case "recall":
      default:
        return promptTextForSupportLanguage(
          "sayMeaningOutLoud",
          baseLanguage,
          { targetLanguage: cleanTarget },
        );
    }
  }

  switch (family) {
    case "completion":
      if (cleanSentence) {
        return promptTextForSupportLanguage(
          "tryTranslateSentence",
          baseLanguage,
          { targetLanguage: cleanTarget },
        );
      }
      return promptTextForSupportLanguage(
        "howDoYouSayThisIn",
        baseLanguage,
        { targetLanguage: cleanTarget },
      );
    case "production":
      return promptTextForSupportLanguage(
        "howDoYouSayThisIn",
        baseLanguage,
        { targetLanguage: cleanTarget },
      );
    case "recall":
    default:
      return promptTextForSupportLanguage(
        "listenThinkMeaning",
        baseLanguage,
        { targetLanguage: cleanTarget },
      );
  }
}

function buildRevealText({
  practiceMode = "standard",
  promptFamily = "recall",
  targetLanguage = "Basque",
  baseLanguage = "English",
  term = "",
  meaning = "",
}) {
  const mode = String(practiceMode || "").trim().toLowerCase();
  const family = normalizePromptFamily(promptFamily);
  const cleanTerm = sanitizeShortLabel(term, 250);
  const cleanMeaning = sanitizeShortLabel(meaning, 250);
  const cleanTarget = sanitizeShortLabel(targetLanguage, 50);

  if (mode === "speech") {
    switch (family) {
      case "completion":
        return promptTextForSupportLanguage("answerIs", baseLanguage, {
          targetLanguage: cleanTarget,
          term: cleanTerm,
          meaning: cleanMeaning,
        });
      case "production":
        return promptTextForSupportLanguage("xInTargetIsY", baseLanguage, {
          targetLanguage: cleanTarget,
          term: cleanTerm,
          meaning: cleanMeaning,
        });
      case "recall":
      default:
        return promptTextForSupportLanguage("meansIs", baseLanguage, {
          targetLanguage: cleanTarget,
          term: cleanTerm,
          meaning: cleanMeaning,
        });
    }
  }

  switch (family) {
    case "completion":
      return cleanTerm;
    case "production":
      return cleanTerm;
    case "recall":
    default:
      return cleanMeaning;
  }
}

function buildTargetAudioText({
  promptFamily = "recall",
  term = "",
  meaning = "",
  sentence = "",
}) {
  const family = normalizePromptFamily(promptFamily);
  const cleanTerm = sanitizeShortLabel(term, 250);
  const cleanMeaning = sanitizeShortLabel(meaning, 250);
  const cleanSentence = sanitizeShortLabel(sentence, 250);

  switch (family) {
    case "completion":
      return cleanSentence || cleanTerm;
    case "production":
      return cleanMeaning;
    case "recall":
    default:
      return cleanTerm;
  }
}

function buildRevealAudioLanguage({
  promptFamily = "recall",
  targetLanguage = "Basque",
  baseLanguage = "English",
  practiceMode = "standard",
}) {
  const family = normalizePromptFamily(promptFamily);
  const mode = String(practiceMode || "").trim().toLowerCase();

  if (mode === "speech") {
    return baseLanguage;
  }

  switch (family) {
    case "completion":
    case "production":
      return targetLanguage;
    case "recall":
    default:
      return baseLanguage;
  }
}

function buildTargetAudioLanguage({
  promptFamily = "recall",
  targetLanguage = "Basque",
  baseLanguage = "English",
}) {
  const family = normalizePromptFamily(promptFamily);

  switch (family) {
    case "production":
      return baseLanguage;
    case "completion":
    case "recall":
    default:
      return targetLanguage;
  }
}

/* -------------------------------------------------------------------------- */
/*                                   Routes                                   */
/* -------------------------------------------------------------------------- */

app.get("/", (req, res) => {
  res.json({
    ok: true,
    message: "AI engine connected",
    azureConfigured: isAzureConfigured(),
    audioBaseUrl: `${req.protocol}://${req.get("host")}/audio/`,
  });
});
app.get("/health", (req, res) => {
  res.json({
    ok: true,
    uptime: process.uptime(),
    azureConfigured: isAzureConfigured(),
  });
});
app.post("/generateSet", async (req, res) => {
  try {
    const {
      mode = "manual",
      difficulty = "beginner",
      vocab = [],
      prompt = "",
      notes = "",
      desiredCount = 10,
      imageBase64 = "",
      imageMimeType = "image/jpeg",
      targetLanguage = "Basque",
      baseLanguage = "English",
    } = req.body || {};

    const cleanMode = String(mode).trim().toLowerCase();
    const cleanDifficulty = normalizeDifficulty(difficulty);
    const cleanDesiredCount = clampNumber(desiredCount, 1, 25, 10);

    if (!["manual", "prompt", "image"].includes(cleanMode)) {
      return res.status(400).json({ error: "Invalid mode" });
    }

    if (cleanMode === "manual" && (!Array.isArray(vocab) || vocab.length === 0)) {
      return res.status(400).json({ error: "Manual mode needs vocab[]" });
    }

    if (cleanMode === "prompt" && !String(prompt).trim()) {
      return res.status(400).json({ error: "Prompt mode needs prompt" });
    }

    if (cleanMode === "image" && !String(imageBase64).trim()) {
      return res.status(400).json({ error: "Image mode needs imageBase64" });
    }

    const instructions = buildGenerateInstructions({
      mode: cleanMode,
      difficulty: cleanDifficulty,
      targetLanguage,
      baseLanguage,
      desiredCount: cleanDesiredCount,
    });

    const textInput = buildGenerateInput({
      mode: cleanMode,
      vocab,
      prompt,
      notes,
      desiredCount: cleanDesiredCount,
      targetLanguage,
      baseLanguage,
      difficulty: cleanDifficulty,
    });

    let input;

    if (cleanMode === "image") {
      const imageDataUrl = `data:${imageMimeType};base64,${imageBase64}`;
      input = [
        {
          role: "user",
          content: [
            { type: "input_text", text: textInput },
            { type: "input_image", image_url: imageDataUrl },
          ],
        },
      ];
    } else {
      input = textInput;
    }

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      instructions,
      input,
      text: {
        format: {
          type: "json_schema",
          name: "language_audio_set",
          schema: generationSchema.schema,
          strict: true,
        },
      },
    });

    const parsed = safeJsonParse(response.output_text, { items: [] });

    const items = (parsed.items || [])
      .slice(0, cleanDesiredCount)
      .map((item) => {
        const term = String(item.term ?? "").trim();
        const meaning = String(item.meaning ?? "").trim();
        const wordType = String(item.guessedWordType ?? "other").trim();
        const promptType = String(item.promptFamily ?? "recall").trim();

        let safeExampleSentences = Array.isArray(item.safeExampleSentences)
          ? item.safeExampleSentences.map((s) => String(s ?? "").trim()).filter(Boolean)
          : [];

        if (safeExampleSentences.length === 0) {
          safeExampleSentences = [term, term];
        } else if (safeExampleSentences.length === 1) {
          safeExampleSentences = [safeExampleSentences[0], safeExampleSentences[0]];
        } else {
          safeExampleSentences = safeExampleSentences.slice(0, 2);
        }

        let exampleTranslations = Array.isArray(item.exampleTranslations)
          ? item.exampleTranslations.map((s) => String(s ?? "").trim()).filter(Boolean)
          : [];

        if (exampleTranslations.length === 0) {
          exampleTranslations = safeExampleSentences.map(() => meaning);
        } else if (exampleTranslations.length === 1) {
          exampleTranslations = [exampleTranslations[0], exampleTranslations[0]];
        } else {
          exampleTranslations = exampleTranslations.slice(0, 2);
        }

        while (exampleTranslations.length < safeExampleSentences.length) {
          exampleTranslations.push(meaning);
        }

        return {
          term,
          meaning,
          wordType,
          promptType,
          safeExampleSentences,
          exampleTranslations,
        };
      })
      .filter((item) => item.term);

    res.json({
      ok: true,
      mode: cleanMode,
      difficulty: cleanDifficulty,
      targetLanguage,
      baseLanguage,
      items,
    });
  } catch (error) {
    console.error("generateSet error:", error);
    res.status(500).json({
      error: "Failed to generate set",
      details: error.message,
    });
  }
});

app.post("/tts", async (req, res) => {
  try {
    const { text, language, speed = 1.0, voice = "" } = req.body || {};

    console.log("TTS request:", {
      language,
      speed,
      voice,
      text: String(text || "").slice(0, 120),
    });

    const result = await getOrCreateTtsFile({
      req,
      text,
      language,
      speed,
      voice,
    });

    console.log("TTS response:", {
      provider: result.provider,
      voice: result.voice,
      language: result.language,
      audioUrl: result.audioUrl,
    });

    res.json({
      ok: true,
      cached: result.cached,
      language: result.language,
      provider: result.provider,
      voice: result.voice,
      textLength: result.text.length,
      audioUrl: result.audioUrl,
    });
  } catch (error) {
    console.error("tts error:", error);
    res.status(500).json({
      error: "TTS failed",
      details: error.message,
    });
  }
});

app.post("/ttsBatch", async (req, res) => {
  try {
    const { lines = [], speed = 1.0, voice = "" } = req.body || {};

    if (!Array.isArray(lines) || lines.length === 0) {
      return res.status(400).json({ error: "lines[] is required" });
    }

    const cleanSpeed = clampNumber(speed, 0.5, 2.0, 1.0);
    const deduped = [];
    const seen = new Set();

    for (const line of lines.slice(0, 100)) {
      const text = String(line.text || "").trim();
      const language = String(line.language || "").trim();
      if (!text || !language) continue;
      const key = `${normalizeLanguage(language)}|${cleanSpeed.toFixed(2)}|${text}`;
      if (seen.has(key)) continue;
      seen.add(key);
      deduped.push({ text, language, key });
    }

    const items = await mapWithConcurrency(deduped, 5, async (line) => {
      const audio = await getOrCreateTtsFile({
        req,
        text: line.text,
        language: line.language,
        speed: cleanSpeed,
        voice,
      });

      return {
        key: line.key,
        text: line.text,
        language: normalizeLanguage(line.language),
        audioUrl: audio.audioUrl,
      };
    });

    res.json({
      ok: true,
      count: items.length,
      items,
    });
  } catch (error) {
    console.error("ttsBatch error:", error);
    res.status(500).json({
      error: "TTS batch failed",
      details: error.message,
    });
  }
});

app.post("/buildSessionAudio", async (req, res) => {
  try {
    const {
      practiceMode = "standard",
      promptFamily = "recall",
      targetLanguage = "Basque",
      baseLanguage = "English",
      term = "",
      meaning = "",
      exampleSentence = "",
      answerDelaySeconds = 3,
      speed = 1.0,
      voice = "",
      includeExampleAudio = false,
    } = req.body || {};

    const cleanPracticeMode = String(practiceMode || "standard").trim().toLowerCase();
    const cleanPromptFamily = normalizePromptFamily(promptFamily);
    const cleanTerm = sanitizeShortLabel(term, 250);
    const cleanMeaning = sanitizeShortLabel(meaning, 250);
    const cleanExampleSentence = sanitizeShortLabel(exampleSentence, 250);
    const cleanDelay = clampNumber(answerDelaySeconds, 0, 15, 3);
    const cleanSpeed = clampNumber(speed, 0.5, 2.0, 1.0);

    if (!cleanTerm && !cleanMeaning && !cleanExampleSentence) {
      return res.status(400).json({ error: "Missing card content" });
    }

    const promptText = buildPromptText({
      practiceMode: cleanPracticeMode,
      promptFamily: cleanPromptFamily,
      targetLanguage,
      baseLanguage,
      sentence: cleanExampleSentence,
    });

    const targetAudioText = buildTargetAudioText({
      promptFamily: cleanPromptFamily,
      term: cleanTerm,
      meaning: cleanMeaning,
      sentence: cleanExampleSentence,
    });

    const revealText = buildRevealText({
      practiceMode: cleanPracticeMode,
      promptFamily: cleanPromptFamily,
      targetLanguage,
      baseLanguage,
      term: cleanTerm,
      meaning: cleanMeaning,
    });

    const promptAudioLanguage = baseLanguage;
    const targetAudioLanguage = buildTargetAudioLanguage({
      promptFamily: cleanPromptFamily,
      targetLanguage,
      baseLanguage,
    });
    const revealAudioLanguage = buildRevealAudioLanguage({
      promptFamily: cleanPromptFamily,
      targetLanguage,
      baseLanguage,
      practiceMode: cleanPracticeMode,
    });

    const promptAudio = await getOrCreateTtsFile({
      req,
      text: promptText,
      language: promptAudioLanguage,
      speed: cleanSpeed,
      voice,
    });

    const targetAudio = await getOrCreateTtsFile({
      req,
      text: targetAudioText,
      language: targetAudioLanguage,
      speed: cleanSpeed,
      voice,
    });

    const revealAudio = await getOrCreateTtsFile({
      req,
      text: revealText,
      language: revealAudioLanguage,
      speed: cleanSpeed,
      voice,
    });

    let exampleAudio = null;

    if (includeExampleAudio && cleanExampleSentence) {
      exampleAudio = await getOrCreateTtsFile({
        req,
        text: cleanExampleSentence,
        language: targetLanguage,
        speed: cleanSpeed,
        voice,
      });
    }

    const sequence = [
      {
        key: "prompt",
        label: "Prompt",
        text: promptText,
        language: normalizeLanguage(promptAudioLanguage),
        audioUrl: promptAudio.audioUrl,
      },
      {
        key: "target",
        label: "Target",
        text: targetAudioText,
        language: normalizeLanguage(targetAudioLanguage),
        audioUrl: targetAudio.audioUrl,
      },
      {
        key: "thinking_pause",
        label: "Thinking Pause",
        pauseMs: Math.round(cleanDelay * 1000),
      },
      {
        key: "reveal",
        label: "Reveal",
        text: revealText,
        language: normalizeLanguage(revealAudioLanguage),
        audioUrl: revealAudio.audioUrl,
      },
    ];

    if (exampleAudio) {
      sequence.push({
        key: "example",
        label: "Example",
        text: cleanExampleSentence,
        language: normalizeLanguage(targetLanguage),
        audioUrl: exampleAudio.audioUrl,
      });
    }

    res.json({
      ok: true,
      practiceMode: cleanPracticeMode,
      promptFamily: cleanPromptFamily,
      targetLanguage: normalizeLanguage(targetLanguage),
      baseLanguage: normalizeLanguage(baseLanguage),
      card: {
        promptText,
        targetText: targetAudioText,
        revealText,
        exampleSentence: cleanExampleSentence || null,
        answerDelaySeconds: cleanDelay,
      },
      audio: {
        prompt: {
          text: promptText,
          language: normalizeLanguage(promptAudioLanguage),
          audioUrl: promptAudio.audioUrl,
        },
        target: {
          text: targetAudioText,
          language: normalizeLanguage(targetAudioLanguage),
          audioUrl: targetAudio.audioUrl,
        },
        reveal: {
          text: revealText,
          language: normalizeLanguage(revealAudioLanguage),
          audioUrl: revealAudio.audioUrl,
        },
        example: exampleAudio
          ? {
              text: cleanExampleSentence,
              language: normalizeLanguage(targetLanguage),
              audioUrl: exampleAudio.audioUrl,
            }
          : null,
      },
      sequence,
    });
  } catch (error) {
    console.error("buildSessionAudio error:", error);
    res.status(500).json({
      error: "Failed to build session audio",
      details: error.message,
    });
  }
});

app.post("/precacheSessionAudio", async (req, res) => {
  try {
    const {
      items = [],
      practiceMode = "standard",
      targetLanguage = "Basque",
      baseLanguage = "English",
      answerDelaySeconds = 3,
      speed = 1.0,
      includeExampleAudio = false,
      voice = "",
    } = req.body || {};

    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ error: "items[] is required" });
    }

    const cappedItems = items.slice(0, 100);
    const results = await mapWithConcurrency(cappedItems, 4, async (item) => {
      const term = String(item.term || "").trim();
      const meaning = String(item.meaning || "").trim();
      const promptFamily = normalizePromptFamily(
        item.promptType || item.promptFamily || "recall",
      );
      const exampleSentence =
        Array.isArray(item.safeExampleSentences) && item.safeExampleSentences.length > 0
          ? String(item.safeExampleSentences[0] || "").trim()
          : String(item.exampleSentence || "").trim();

      const promptText = buildPromptText({
        practiceMode,
        promptFamily,
        targetLanguage,
        baseLanguage,
        sentence: exampleSentence,
      });

      const targetAudioText = buildTargetAudioText({
        promptFamily,
        term,
        meaning,
        sentence: exampleSentence,
      });

      const revealText = buildRevealText({
        practiceMode,
        promptFamily,
        targetLanguage,
        baseLanguage,
        term,
        meaning,
      });

      const promptAudioLanguage = baseLanguage;
      const targetAudioLanguage = buildTargetAudioLanguage({
        promptFamily,
        targetLanguage,
        baseLanguage,
      });
      const revealAudioLanguage = buildRevealAudioLanguage({
        promptFamily,
        targetLanguage,
        baseLanguage,
        practiceMode,
      });

      const [promptAudio, targetAudio, revealAudio, exampleAudio] = await Promise.all([
        getOrCreateTtsFile({
          req,
          text: promptText,
          language: promptAudioLanguage,
          speed,
          voice,
        }),
        getOrCreateTtsFile({
          req,
          text: targetAudioText,
          language: targetAudioLanguage,
          speed,
          voice,
        }),
        getOrCreateTtsFile({
          req,
          text: revealText,
          language: revealAudioLanguage,
          speed,
          voice,
        }),
        includeExampleAudio && exampleSentence
          ? getOrCreateTtsFile({
              req,
              text: exampleSentence,
              language: targetLanguage,
              speed,
              voice,
            })
          : Promise.resolve(null),
      ]);

      return {
        term,
        meaning,
        promptFamily,
        answerDelaySeconds: clampNumber(answerDelaySeconds, 0, 15, 3),
        audio: {
          prompt: {
            text: promptText,
            language: normalizeLanguage(promptAudioLanguage),
            audioUrl: promptAudio.audioUrl,
          },
          target: {
            text: targetAudioText,
            language: normalizeLanguage(targetAudioLanguage),
            audioUrl: targetAudio.audioUrl,
          },
          reveal: {
            text: revealText,
            language: normalizeLanguage(revealAudioLanguage),
            audioUrl: revealAudio.audioUrl,
          },
          example: exampleAudio
            ? {
                text: exampleSentence,
                language: normalizeLanguage(targetLanguage),
                audioUrl: exampleAudio.audioUrl,
              }
            : null,
        },
      };
    });

    res.json({
      ok: true,
      count: results.length,
      items: results,
    });
  } catch (error) {
    console.error("precacheSessionAudio error:", error);
    res.status(500).json({
      error: "Failed to precache session audio",
      details: error.message,
    });
  }
});

/* -------------------------------------------------------------------------- */
/*                                  Startup                                   */
/* -------------------------------------------------------------------------- */

const PORT = Number(process.env.PORT) || 3000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Audio directory: ${AUDIO_DIR}`);
  console.log(`Azure configured: ${isAzureConfigured() ? "yes" : "no"}`);
});

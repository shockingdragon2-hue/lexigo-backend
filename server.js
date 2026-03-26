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
const TMP_AUDIO_DIR = path.join(AUDIO_DIR, ".tmp");
const MIN_AUDIO_FILE_BYTES = 512;

ensureDir(AUDIO_DIR);
ensureDir(TMP_AUDIO_DIR);


app.get("/audio/:fileName", async (req, res) => {
  try {
    const requestedName = path.basename(String(req.params.fileName || ""));

    if (!requestedName || !requestedName.endsWith(".mp3")) {
      return res.status(400).json({ error: "Invalid audio file name" });
    }

    const filePath = path.join(AUDIO_DIR, requestedName);
    const ready = await waitForFileReady(filePath, {
      minBytes: MIN_AUDIO_FILE_BYTES,
      attempts: 6,
      delayMs: 120,
    });

    if (!ready) {
      return res.status(404).json({ error: "Audio file not ready" });
    }

    const stat = fs.statSync(filePath);
    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Content-Length", String(stat.size));
    res.setHeader("Cache-Control", "public, max-age=2592000, immutable");
    res.setHeader("Accept-Ranges", "bytes");
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.sendFile(filePath);
  } catch (error) {
    console.error("audio route error:", error);
    res.status(500).json({ error: "Failed to serve audio", details: error.message });
  }
});

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const OPENAI_TTS_MODEL = process.env.OPENAI_TTS_MODEL || "gpt-4o-mini-tts";
const DEFAULT_OPENAI_VOICE = process.env.OPENAI_TTS_VOICE || "alloy";
const ELEVENLABS_MODEL_ID = process.env.ELEVENLABS_MODEL_ID || "eleven_multilingual_v2";
const ELEVENLABS_OUTPUT_FORMAT = process.env.ELEVENLABS_OUTPUT_FORMAT || "mp3_44100_128";
const ELEVENLABS_VOICE_STABILITY = clampNumber(
  process.env.ELEVENLABS_VOICE_STABILITY,
  0,
  1,
  0.45,
);
const ELEVENLABS_VOICE_SIMILARITY = clampNumber(
  process.env.ELEVENLABS_VOICE_SIMILARITY,
  0,
  1,
  0.8,
);
const ELEVENLABS_VOICE_STYLE = clampNumber(
  process.env.ELEVENLABS_VOICE_STYLE,
  0,
  1,
  0.15,
);
const ELEVENLABS_SPEAKER_BOOST =
  String(process.env.ELEVENLABS_SPEAKER_BOOST || "true").trim().toLowerCase() !== "false";
const inflightTtsJobs = new Map();
const TTS_JOB_TIMEOUT_MS = clampNumber(process.env.TTS_JOB_TIMEOUT_MS, 3000, 30000, 12000);
const TTS_BATCH_CONCURRENCY = Math.max(1, Math.floor(clampNumber(process.env.TTS_BATCH_CONCURRENCY, 1, 12, 4)));
const TTS_PRECACHED_ITEM_CONCURRENCY = Math.max(1, Math.floor(clampNumber(process.env.TTS_PRECACHED_ITEM_CONCURRENCY, 1, 12, 3)));
const TTS_AUDIO_PART_CONCURRENCY = Math.max(1, Math.floor(clampNumber(process.env.TTS_AUDIO_PART_CONCURRENCY, 1, 6, 3)));

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


function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function withTimeout(promise, ms, label = "operation") {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);

    Promise.resolve(promise).then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

async function mapWithConcurrency(items, concurrency, mapper) {
  const list = Array.isArray(items) ? items : [];
  if (list.length === 0) return [];

  const results = new Array(list.length);
  let nextIndex = 0;

  async function worker() {
    while (true) {
      const current = nextIndex;
      nextIndex += 1;
      if (current >= list.length) return;
      results[current] = await mapper(list[current], current);
    }
  }

  const workerCount = Math.max(1, Math.min(concurrency, list.length));
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
}

async function settleWithConcurrency(items, concurrency, mapper) {
  return mapWithConcurrency(items, concurrency, async (item, index) => {
    try {
      const value = await mapper(item, index);
      return { status: "fulfilled", value };
    } catch (error) {
      return {
        status: "rejected",
        reason: error instanceof Error ? error.message : String(error),
      };
    }
  });
}

function makeTempAudioPath(outputPath) {
  const tempName = `${path.basename(outputPath)}.${process.pid}.${Date.now()}.${Math.random()
    .toString(16)
    .slice(2)}.tmp`;
  return path.join(TMP_AUDIO_DIR, tempName);
}

function isFileReady(filePath, minBytes = MIN_AUDIO_FILE_BYTES) {
  try {
    const stat = fs.statSync(filePath);
    return stat.isFile() && stat.size >= minBytes;
  } catch {
    return false;
  }
}

async function waitForFileReady(filePath, { minBytes = MIN_AUDIO_FILE_BYTES, attempts = 8, delayMs = 120 } = {}) {
  for (let i = 0; i < attempts; i += 1) {
    if (isFileReady(filePath, minBytes)) {
      return true;
    }
    await sleep(delayMs);
  }
  return false;
}

function removeFileIfExists(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (error) {
    console.warn(`Failed to remove file ${filePath}:`, error.message);
  }
}

function finalizeGeneratedAudio(tempPath, outputPath) {
  if (!isFileReady(tempPath)) {
    removeFileIfExists(tempPath);
    throw new Error(`Generated audio file was empty or invalid for ${path.basename(outputPath)}`);
  }

  fs.renameSync(tempPath, outputPath);

  if (!isFileReady(outputPath)) {
    removeFileIfExists(outputPath);
    throw new Error(`Final audio file was not ready after rename for ${path.basename(outputPath)}`);
  }
}

function sanitizeText(text, maxLen = 4000) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .trim()
    .slice(0, maxLen);
}

function collapseImmediateWordRepeats(text) {
  const clean = sanitizeText(text, 4000);
  if (!clean) return "";

  const parts = clean.split(/(\s+)/);
  const collapsed = [];
  let lastWord = null;

  for (const part of parts) {
    if (!part.trim()) {
      collapsed.push(part);
      continue;
    }

    const normalized = part
      .toLowerCase()
      .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");

    if (normalized && normalized === lastWord) {
      continue;
    }

    collapsed.push(part);
    if (normalized) {
      lastWord = normalized;
    }
  }

  return collapsed.join("").replace(/\s+/g, " ").trim();
}

function normalizeGeneratedSentence(text, { difficulty = "beginner" } = {}) {
  let clean = collapseImmediateWordRepeats(text);

  if (!clean) return "";

  if (normalizeDifficulty(difficulty) === "beginner") {
    clean = clean
      .replace(/i have come/gi, "I come")
      .replace(/i have arrived/gi, "I arrive")
      .replace(/i came early/gi, "I come early")
      .replace(/i have come early/gi, "I come early");
  }

  return clean;
}

function sanitizeTtsText(text) {
  return sanitizeText(text, 4000);
}

function sanitizeShortLabel(text, maxLen = 200) {
  return sanitizeText(text, maxLen);
}

function uniqueNonEmptyStrings(values, maxItems = 2) {
  const result = [];
  const seen = new Set();

  for (const value of Array.isArray(values) ? values : []) {
    const clean = sanitizeText(value, 400).trim();
    if (!clean) continue;
    const key = clean.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(clean);
    if (result.length >= maxItems) break;
  }

  return result;
}

function dedupeLessonItems(items, maxItems = 25) {
  const result = [];
  const seen = new Set();

  for (const item of Array.isArray(items) ? items : []) {
    const term = sanitizeShortLabel(item?.term || "", 250);
    const meaning = stripMeaningNoise(item?.meaning || "");
    if (!term || !meaning) continue;

    const key = `${normalizeLanguage(item?.language || "")}|${term.toLowerCase()}|${meaning.toLowerCase()}`;
    if (seen.has(key)) continue;
    seen.add(key);

    result.push({
      ...item,
      term,
      meaning,
      safeExampleSentences: uniqueNonEmptyStrings(item?.safeExampleSentences || [], 2),
      exampleTranslations: uniqueNonEmptyStrings(item?.exampleTranslations || [], 2),
    });

    if (result.length >= maxItems) break;
  }

  return result;
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


function normalizeMeaningForLookup(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[.,!?]/g, "")
    .replace(/\s+/g, " ");
}

function basqueNumberForMeaning(meaning) {
  const normalized = normalizeMeaningForLookup(meaning);
  const map = {
    "0": "zero",
    "zero": "zero",
    "cero": "zero",
    "1": "bat",
    "one": "bat",
    "uno": "bat",
    "una": "bat",
    "2": "bi",
    "two": "bi",
    "dos": "bi",
    "3": "hiru",
    "three": "hiru",
    "tres": "hiru",
    "4": "lau",
    "four": "lau",
    "cuatro": "lau",
    "5": "bost",
    "five": "bost",
    "cinco": "bost",
    "6": "sei",
    "six": "sei",
    "seis": "sei",
    "7": "zazpi",
    "seven": "zazpi",
    "siete": "zazpi",
    "8": "zortzi",
    "eight": "zortzi",
    "ocho": "zortzi",
    "9": "bederatzi",
    "nine": "bederatzi",
    "nueve": "bederatzi",
    "10": "hamar",
    "ten": "hamar",
    "diez": "hamar",
    "11": "hamaika",
    "eleven": "hamaika",
    "once": "hamaika",
    "12": "hamabi",
    "twelve": "hamabi",
    "doce": "hamabi",
    "13": "hamahiru",
    "thirteen": "hamahiru",
    "trece": "hamahiru",
    "14": "hamalau",
    "fourteen": "hamalau",
    "catorce": "hamalau",
    "15": "hamabost",
    "fifteen": "hamabost",
    "quince": "hamabost",
    "16": "hamasei",
    "sixteen": "hamasei",
    "dieciseis": "hamasei",
    "dieciséis": "hamasei",
    "17": "hamazazpi",
    "seventeen": "hamazazpi",
    "diecisiete": "hamazazpi",
    "18": "hemezortzi",
    "eighteen": "hemezortzi",
    "dieciocho": "hemezortzi",
    "19": "hemeretzi",
    "nineteen": "hemeretzi",
    "diecinueve": "hemeretzi",
    "20": "hogei",
    "twenty": "hogei",
    "veinte": "hogei",
  };
  return map[normalized] || null;
}


function stripMeaningNoise(value) {
  let text = sanitizeShortLabel(value || "", 250);
  text = text
    .replace(/^(means|significa|veut dire|bedeutet|esan nahi du)\s+/i, "")
    .replace(/^(color|colour|colore|couleur|farbe)\s+/i, "")
    .replace(/^(the color|el color|la couleur|die farbe)\s+/i, "")
    .trim();
  return text;
}

function normalizeBasqueSurfaceForm(term, meaning) {
  const cleanTerm = sanitizeShortLabel(term || "", 250)
    .replace(/^etori$/i, "etorri");
  const normalizedMeaning = normalizeMeaningForLookup(stripMeaningNoise(meaning));
  const colorMap = {
    green: "berde",
    verde: "berde",
    red: "gorri",
    rojo: "gorri",
    blue: "urdin",
    azul: "urdin",
    yellow: "hori",
    amarillo: "hori",
    black: "beltz",
    negro: "beltz",
    white: "zuri",
    blanco: "zuri",
    orange: "laranja",
    naranja: "laranja",
    purple: "more",
    morado: "more",
    brown: "marroi",
    marron: "marroi",
    marrón: "marroi",
    gray: "gris",
    grey: "gris",
    gris: "gris",
  };
  if (colorMap[normalizedMeaning]) {
    return colorMap[normalizedMeaning];
  }
  return cleanTerm;
}

function isFunctionWord(term, targetLanguage) {
  const normalizedTarget = normalizeLanguage(targetLanguage);
  const normalizedTerm = String(term || "").trim().toLowerCase();

  if (!normalizedTerm) return false;

  if (normalizedTarget === "basque") {
    return [
      "ez", "eta", "ere", "baina", "edo", "baita", "ba", "al",
      "da", "naiz", "gara", "zara", "dira", "dut", "duzu", "du", "dugu", "duzue", "dute",
    ].includes(normalizedTerm);
  }

  return false;
}

function shouldFilterStandaloneItem({ term, wordType, targetLanguage, difficulty, mode }) {
  const cleanDifficulty = normalizeDifficulty(difficulty);
  const cleanMode = String(mode || "").trim().toLowerCase();
  const normalizedType = String(wordType || "").trim().toLowerCase();

  if (cleanMode === "manual") return false;
  if (!["intermediate", "advanced"].includes(cleanDifficulty)) return false;

  return isFunctionWord(term, targetLanguage) || ["other", "adverb"].includes(normalizedType);
}

function enforceLevelAwareItemRules({ item, targetLanguage, baseLanguage, difficulty, mode }) {
  const next = { ...item };
  const cleanDifficulty = normalizeDifficulty(difficulty);

  next.term = sanitizeShortLabel(next.term || "", 250);
  next.meaning = stripMeaningNoise(next.meaning || "");
  next.safeExampleSentences = Array.isArray(next.safeExampleSentences)
    ? next.safeExampleSentences.map((s) => normalizeGeneratedSentence(s, { difficulty: cleanDifficulty })).filter(Boolean)
    : [];
  next.exampleTranslations = Array.isArray(next.exampleTranslations)
    ? next.exampleTranslations.map((s) => normalizeGeneratedSentence(s, { difficulty: cleanDifficulty })).filter(Boolean)
    : [];

  if (shouldFilterStandaloneItem({
    term: next.term,
    wordType: next.wordType,
    targetLanguage,
    difficulty: cleanDifficulty,
    mode,
  })) {
    return null;
  }

  if (isFunctionWord(next.term, targetLanguage)) {
    if (normalizeLanguage(targetLanguage) === "basque" && next.term.toLowerCase() === "ez") {
      next.meaning = cleanDifficulty === "beginner" ? "no / not (negative marker)" : "negative marker used to negate a sentence";
    }
    next.wordType = "other";
    next.promptType = "recall";
  }

  return next;
}

function applyKnownTermCorrections({ targetLanguage, baseLanguage, items, difficulty = "beginner", mode = "manual" }) {
  const normalizedTarget = normalizeLanguage(targetLanguage);
  if (!Array.isArray(items) || items.length === 0) return [];

  return items
    .map((item) => enforceLevelAwareItemRules({
      item,
      targetLanguage,
      baseLanguage,
      difficulty,
      mode,
    }))
    .filter(Boolean)
    .map((next) => {
      if (normalizedTarget === "basque") {
        const correctedNumber = basqueNumberForMeaning(next.meaning);
        if (correctedNumber) {
          next.term = correctedNumber;
        }

        next.term = normalizeBasqueSurfaceForm(next.term, next.meaning);

        if (normalizeMeaningForLookup(next.term) === "hemezortzi" && basqueNumberForMeaning(next.meaning) === "hemeretzi") {
          next.term = "hemeretzi";
        }
      }

      return next;
    });
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
      return "en-US-AriaNeural";
    case "portuguese":
      return "pt-BR-FranciscaNeural";
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
      return "pt-BR";
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

function isElevenLabsConfigured() {
  return false;
}

function elevenLabsLanguageCode(language) {
  const normalized = normalizeLanguage(language);

  switch (normalized) {
    case "english":
      return "en";
    case "spanish":
      return "es";
    case "portuguese":
      return "pt";
    case "french":
      return "fr";
    case "german":
      return "de";
    case "italian":
      return "it";
    default:
      return null;
  }
}

function elevenLabsVoiceForLanguage(language, explicitVoice = "") {
  const normalized = normalizeLanguage(language);
  const cleanExplicit = String(explicitVoice || "").trim();

  if (cleanExplicit) {
    return cleanExplicit;
  }

  const byLanguage = {
    english:
      process.env.ELEVENLABS_ENGLISH_VOICE_ID ||
      process.env.ELEVENLABS_DEFAULT_VOICE_ID ||
      "",
    spanish:
      process.env.ELEVENLABS_SPANISH_VOICE_ID ||
      process.env.ELEVENLABS_DEFAULT_VOICE_ID ||
      "",
    portuguese:
      process.env.ELEVENLABS_PORTUGUESE_VOICE_ID ||
      process.env.ELEVENLABS_DEFAULT_VOICE_ID ||
      "",
    french:
      process.env.ELEVENLABS_FRENCH_VOICE_ID ||
      process.env.ELEVENLABS_DEFAULT_VOICE_ID ||
      "",
    german:
      process.env.ELEVENLABS_GERMAN_VOICE_ID ||
      process.env.ELEVENLABS_DEFAULT_VOICE_ID ||
      "",
    italian:
      process.env.ELEVENLABS_ITALIAN_VOICE_ID ||
      process.env.ELEVENLABS_DEFAULT_VOICE_ID ||
      "",
  };

  return String(byLanguage[normalized] || process.env.ELEVENLABS_DEFAULT_VOICE_ID || "").trim();
}

function shouldUseElevenLabs(language, voice = "") {
  return false;
}

function getBaseUrl(req) {
  return process.env.PUBLIC_BASE_URL || `${req.protocol}://${req.get("host")}`;
}

function makeAudioUrl(req, fileName) {
  return `${getBaseUrl(req)}/audio/${fileName}`;
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
      listenThinkMeaning: "What do you think this means?",
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
    portuguese: {
      listenThinkMeaning: "O que você acha que isso significa?",
      whatDoesThisMean: "O que isso significa?",
      howDoYouSayThisIn: `Como se diz isso em ${targetLanguage}?`,
      tryTranslateSentence: "Tente traduzir esta frase.",
      sayMeaningOutLoud: "Diga em voz alta o que isso significa.",
      sayWordOutLoud: `Como se diz isso em ${targetLanguage}? Diga em voz alta.`,
      sayMissingWordOutLoud: `Escute e depois diga em voz alta a palavra que falta em ${targetLanguage}.`,
      answerIs: `A resposta é ${term}.`,
      meansIs: `${term} significa ${meaning}.`,
      xInTargetIsY: `${meaning} em ${targetLanguage} é ${term}.`,
    },
    french: {
      listenThinkMeaning: "Qu'est-ce que tu penses que cela veut dire ?",
      whatDoesThisMean: "Qu'est-ce que cela veut dire ?",
      howDoYouSayThisIn: `Comment dit-on cela en ${targetLanguage} ?`,
      tryTranslateSentence: "Essaie de traduire cette phrase.",
      sayMeaningOutLoud: "Dis à voix haute ce que cela veut dire.",
      sayWordOutLoud: `Comment dit-on cela en ${targetLanguage} ? Dis-le à voix haute.`,
      sayMissingWordOutLoud: `Écoute puis dis à voix haute le mot manquant en ${targetLanguage}.`,
      answerIs: `La réponse est ${term}.`,
      meansIs: `${term} veut dire ${meaning}.`,
      xInTargetIsY: `${meaning} en ${targetLanguage}, c'est ${term}.`,
    },
    german: {
      listenThinkMeaning: "Was glaubst du, was das bedeutet?",
      whatDoesThisMean: "Was bedeutet das?",
      howDoYouSayThisIn: `Wie sagt man das auf ${targetLanguage}?`,
      tryTranslateSentence: "Versuche, diesen Satz zu übersetzen.",
      sayMeaningOutLoud: "Sag laut, was das bedeutet.",
      sayWordOutLoud: `Wie sagt man das auf ${targetLanguage}? Sag es laut.`,
      sayMissingWordOutLoud: `Hör zu und sag dann das fehlende ${targetLanguage}-Wort laut.`,
      answerIs: `Die Antwort ist ${term}.`,
      meansIs: `${term} bedeutet ${meaning}.`,
      xInTargetIsY: `${meaning} auf ${targetLanguage} ist ${term}.`,
    },
    italian: {
      listenThinkMeaning: "Che cosa pensi che significhi?",
      whatDoesThisMean: "Che cosa significa?",
      howDoYouSayThisIn: `Come si dice questo in ${targetLanguage}?`,
      tryTranslateSentence: "Prova a tradurre questa frase.",
      sayMeaningOutLoud: "Di' ad alta voce che cosa significa.",
      sayWordOutLoud: `Come si dice questo in ${targetLanguage}? Dillo ad alta voce.`,
      sayMissingWordOutLoud: `Ascolta e poi di' ad alta voce la parola ${targetLanguage} che manca.`,
      answerIs: `La risposta è ${term}.`,
      meansIs: `${term} significa ${meaning}.`,
      xInTargetIsY: `${meaning} in ${targetLanguage} è ${term}.`,
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
- Do not use present perfect or other tense shifts for beginner translations unless explicitly required.
- Avoid duplicated words like "goiz goiz" unless the user explicitly provided them.
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
- Prefer sentence-based learning over isolated grammar particles.
- Do not surface negation markers, auxiliaries, or function words as plain standalone cards unless the user explicitly asked for grammar study.
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
- Meanings must be the plain translation only, not a full sentence and not something like 'means green' or 'color green'.
- Terms must be the clean dictionary-style target-language form only.
- Each item must include:
  - term
  - meaning
  - guessedWordType
  - promptFamily
  - 2 short safeExampleSentences
  - 2 exampleTranslations

Important sentence rules:
- safeExampleSentences must be in the TARGET language.
- exampleTranslations must translate the matching example sentences into the BASE language.
- Keep each translation natural, clear, and complete.
- The term should appear naturally in at least one example sentence when possible.
- Do not use overly advanced sentence structures for beginner mode.
- Keep punctuation simple.
- Do not teach grammar/function words as plain standalone beginner vocab cards unless clearly labeled.
- For intermediate and advanced sets, prefer sentence-based teaching for grammar markers, negation, auxiliaries, particles, and connectors.
- Avoid returning items like a bare negation marker translated as a simple standalone word.

For MANUAL vocab mode:
- keep the original term exactly as provided
- do not replace the learner's supplied term with another synonym
- if the supplied term is already a phrase, keep it as a phrase

For PROMPT mode:
- generate exactly the requested number of useful vocabulary items when possible

For IMAGE mode:
- use the image and notes together
- prioritize visible objects, simple actions, locations, colors, and basic descriptors
- if the notes include grammar markers or helper words, use them in sentences instead of weak standalone flashcard-style items

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

function resolvedTtsRouting({ language, voice = "" }) {
  const normalizedLanguage = normalizeLanguage(language);
  const explicitVoice = String(voice || "").trim();

  if (isAzureConfigured() && azureVoiceForLanguage(normalizedLanguage)) {
    return {
      provider: "azure",
      voice: azureVoiceForLanguage(normalizedLanguage),
      language: normalizedLanguage,
    };
  }

  return {
    provider: "openai",
    voice: explicitVoice || openAiVoiceForLanguage(normalizedLanguage) || DEFAULT_OPENAI_VOICE,
    language: normalizedLanguage,
  };
}

function getTtsCacheKey({ text, language, voice = "", speed = 1.0 }) {
  const route = resolvedTtsRouting({ language, voice });
  return sha256(
    JSON.stringify({
      text: sanitizeTtsText(text),
      language: route.language,
      provider: route.provider,
      voice: route.voice,
      speed: Number(speed || 1).toFixed(2),
    }),
  );
}

function looksLikeQuestion(text) {
  return /\?$/.test(text) ||
    /^(what|how|why|when|where|which|who|do|does|did|is|are|can|could|would|will|should|what's|how's|comment|pourquoi|quand|donde|dónde|como|cómo|que|qué)\b/i.test(text);
}

function isLikelyPromptLine(text, language) {
  const cleanText = sanitizeTtsText(text).toLowerCase();
  const normalizedLanguage = normalizeLanguage(language);

  if (!cleanText || normalizedLanguage !== "english") {
    return false;
  }

  const promptStarters = [
    "what do you think",
    "what does this mean",
    "how do you say",
    "say this",
    "try to translate",
    "translate this sentence",
    "listen",
    "hear the word",
    "say your answer",
    "good work",
    "nice job",
    "you got it",
    "alright, next one",
  ];

  return promptStarters.some((starter) => cleanText.startsWith(starter));
}

function prepareTtsText(text, language) {
  let cleanText = sanitizeTtsText(text);
  const normalizedLanguage = normalizeLanguage(language);

  if (looksLikeQuestion(cleanText) && !cleanText.endsWith("?")) {
    cleanText = `${cleanText.replace(/[.!]+$/g, "")}?`;
  }

  const isSingleWordOrShortPhrase =
    !/[.!?]$/.test(cleanText) &&
    cleanText.split(/\s+/).filter(Boolean).length <= 3;

  if (isSingleWordOrShortPhrase) {
    cleanText = `${cleanText.replace(/[,:;]+$/g, "")}.`;
  }

  if (normalizedLanguage === "english") {
    cleanText = cleanText
      .replace(/^Listen to this word\.?\s*/i, "Listen to this word, ")
      .replace(/^Hear the word first\.?\s*/i, "Hear the word first, ")
      .replace(/^What do you think this means$/i, "What do you think this means?");
  }

  return cleanText;
}

function applyPronunciationOverrides(text, language) {
  const cleanText = prepareTtsText(text, language);
  const normalized = normalizeLanguage(language);

  if (normalized === "basque") {
    const stripped = cleanText.toLowerCase().replace(/[.!?]+$/g, "").trim();
    const basqueSingleWordOverrides = {
      ama: '<phoneme alphabet="ipa" ph="ama">ama</phoneme><break time="220ms"/>',
      guk: '<phoneme alphabet="ipa" ph="ɡuk">guk</phoneme><break time="220ms"/>',
      goiz: '<phoneme alphabet="ipa" ph="ɡois̻">goiz</phoneme><break time="220ms"/>',
      egun: '<phoneme alphabet="ipa" ph="eɡun">egun</phoneme><break time="220ms"/>',
      etorri: '<phoneme alphabet="ipa" ph="etorːi">etorri</phoneme><break time="220ms"/>',
      ogia: '<phoneme alphabet="ipa" ph="oɡia">ogia</phoneme><break time="220ms"/>',
    };

    if (basqueSingleWordOverrides[stripped]) {
      return basqueSingleWordOverrides[stripped];
    }

    if (stripped === "lo egiten") {
      return '<phoneme alphabet="ipa" ph="lo eɣiten">lo egiten</phoneme><break time="220ms"/>';
    }
  }

  return escapeSsml(cleanText);
}

function buildAzureSsml({ text, language, voiceName, speed = 1.0 }) {
  const langCode = azureLangCode(language);
  const normalizedLanguage = normalizeLanguage(language);
  const content = applyPronunciationOverrides(text, language);
  const isEnglishPrompt = normalizedLanguage === "english" && isLikelyPromptLine(text, language);
  const questionLike = looksLikeQuestion(sanitizeTtsText(text));

  const normalizedSpeed = clampNumber(speed, 0.7, 1.35, 1.0);
  const ratePercent = Math.round((normalizedSpeed - 1) * 100);
  const rate = ratePercent >= 0 ? `+${ratePercent}%` : `${ratePercent}%`;

  if (isEnglishPrompt) {
    const contour = questionLike ? ' pitch="+2Hz"' : "";
    return `
      <speak version="1.0" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="${langCode}">
        <voice name="${voiceName}">
          <mstts:express-as style="friendly">
            <prosody rate="${rate}"${contour}>
              ${content}
            </prosody>
          </mstts:express-as>
        </voice>
      </speak>
    `;
  }

  return `
    <speak version="1.0" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="${langCode}">
      <voice name="${voiceName}">
        <prosody rate="${rate}">
          ${content}
        </prosody>
      </voice>
    </speak>
  `;
}

async function synthesizeElevenLabsToFile({
  text,
  language,
  outputPath,
  voice = "",
}) {
  const cleanText = sanitizeTtsText(text);
  const languageCode = elevenLabsLanguageCode(language);
  const voiceId = elevenLabsVoiceForLanguage(language, voice);

  if (!isElevenLabsConfigured()) {
    throw new Error("ElevenLabs API key is not configured");
  }

  if (!languageCode) {
    throw new Error(`ElevenLabs is not enabled for language: ${normalizeLanguage(language)}`);
  }

  if (!voiceId) {
    throw new Error(`No ElevenLabs voice configured for language: ${normalizeLanguage(language)}`);
  }

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(
      voiceId,
    )}?output_format=${encodeURIComponent(ELEVENLABS_OUTPUT_FORMAT)}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": process.env.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      body: JSON.stringify({
        text: cleanText,
        model_id: ELEVENLABS_MODEL_ID,
        language_code: languageCode,
        voice_settings: {
          stability: ELEVENLABS_VOICE_STABILITY,
          similarity_boost: ELEVENLABS_VOICE_SIMILARITY,
          style: ELEVENLABS_VOICE_STYLE,
          use_speaker_boost: ELEVENLABS_SPEAKER_BOOST,
        },
      }),
    },
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`ElevenLabs TTS failed (${response.status}): ${errorText}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const tempPath = makeTempAudioPath(outputPath);
  fs.writeFileSync(tempPath, Buffer.from(arrayBuffer));
  finalizeGeneratedAudio(tempPath, outputPath);

  return {
    ok: true,
    provider: "elevenlabs",
    voice: voiceId,
  };
}

async function synthesizeAzureToFile({ text, language, outputPath, speed = 1.0 }) {
  return new Promise((resolve, reject) => {
    const tempPath = makeTempAudioPath(outputPath);
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

    const audioConfig = sdk.AudioConfig.fromAudioFileOutput(tempPath);
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig, audioConfig);

    const ssml = buildAzureSsml({
      text,
      language,
      voiceName,
      speed,
    });

    synthesizer.speakSsmlAsync(
      ssml,
      (result) => {
        synthesizer.close();

        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          try {
            finalizeGeneratedAudio(tempPath, outputPath);
            resolve({ ok: true, provider: "azure", voice: voiceName });
          } catch (fileError) {
            reject(fileError);
          }
        } else {
          removeFileIfExists(tempPath);
          reject(new Error("Azure synthesis failed"));
        }
      },
      (err) => {
        synthesizer.close();
        removeFileIfExists(tempPath);
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
  speed = 1.0,
}) {
  const cleanText = sanitizeTtsText(text);
  const chosenVoice = voice || openAiVoiceForLanguage(language) || DEFAULT_OPENAI_VOICE;

  const mp3 = await client.audio.speech.create({
    model: OPENAI_TTS_MODEL,
    voice: chosenVoice,
    input: cleanText,
  });

  const buffer = Buffer.from(await mp3.arrayBuffer());
  const tempPath = makeTempAudioPath(outputPath);
  fs.writeFileSync(tempPath, buffer);
  finalizeGeneratedAudio(tempPath, outputPath);

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

  if (!cleanText) {
    throw new Error("Missing TTS text");
  }

  const route = resolvedTtsRouting({
    language: normalizedLanguage,
    voice,
  });

  const cacheKey = getTtsCacheKey({
    text: cleanText,
    language: normalizedLanguage,
    voice,
    speed,
  });

  const fileName = `${cacheKey}.mp3`;
  const outputPath = path.join(AUDIO_DIR, fileName);

  if (isFileReady(outputPath)) {
    return {
      ok: true,
      cached: true,
      language: normalizedLanguage,
      provider: "cache",
      voice: route.voice,
      text: cleanText,
      audioUrl: makeAudioUrl(req, fileName),
      fileName,
      filePath: outputPath,
    };
  }

  if (fs.existsSync(outputPath) && !isFileReady(outputPath)) {
    console.warn(`Removing incomplete audio cache file: ${outputPath}`);
    removeFileIfExists(outputPath);
  }

  if (inflightTtsJobs.has(cacheKey)) {
    await withTimeout(inflightTtsJobs.get(cacheKey), TTS_JOB_TIMEOUT_MS, `Waiting for inflight TTS job (${cacheKey.slice(0, 8)})`);
    const ready = await waitForFileReady(outputPath, {
      minBytes: MIN_AUDIO_FILE_BYTES,
      attempts: 8,
      delayMs: 120,
    });

    if (!ready) {
      throw new Error(`Audio file was still not ready after TTS job completed for ${fileName}`);
    }

    return {
      ok: true,
      cached: true,
      language: normalizedLanguage,
      provider: "cache",
      voice: route.voice,
      text: cleanText,
      audioUrl: makeAudioUrl(req, fileName),
      fileName,
      filePath: outputPath,
    };
  }

  const job = withTimeout((async () => {
    let providerInfo;

    try {
      providerInfo = await synthesizeAzureToFile({
        text: cleanText,
        language: normalizedLanguage,
        outputPath,
        speed,
      });
    } catch (azureError) {
      console.warn("Azure TTS failed, falling back to OpenAI:", azureError.message);
      providerInfo = await synthesizeOpenAIToFile({
        text: cleanText,
        outputPath,
        voice,
        language: normalizedLanguage,
        speed,
      });
    }
    return providerInfo;
  })(), TTS_JOB_TIMEOUT_MS, `TTS generation (${route.provider}:${route.voice})`);

  inflightTtsJobs.set(cacheKey, job);

  let providerInfo;
  try {
    providerInfo = await job;
  } finally {
    inflightTtsJobs.delete(cacheKey);
  }

  const ready = await waitForFileReady(outputPath, {
    minBytes: MIN_AUDIO_FILE_BYTES,
    attempts: 8,
    delayMs: 120,
  });

  if (!ready) {
    removeFileIfExists(outputPath);
    throw new Error(`Generated audio file was not ready for playback: ${fileName}`);
  }

  return {
    ok: true,
    cached: false,
    language: normalizedLanguage,
    provider: providerInfo.provider,
    voice: providerInfo.voice,
    text: cleanText,
    audioUrl: makeAudioUrl(req, fileName),
    fileName,
    filePath: outputPath,
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
      return promptTextForSupportLanguage("meansIs", baseLanguage, {
        targetLanguage: cleanTarget,
        term: cleanTerm,
        meaning: cleanMeaning,
      });
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
    elevenLabsConfigured: isElevenLabsConfigured(),
    audioBaseUrl: `${req.protocol}://${req.get("host")}/audio/`,
    ttsJobTimeoutMs: TTS_JOB_TIMEOUT_MS,
    ttsBatchConcurrency: TTS_BATCH_CONCURRENCY,
  });
});
app.get("/health", (req, res) => {
  res.json({
    ok: true,
    uptime: process.uptime(),
    azureConfigured: isAzureConfigured(),
    elevenLabsConfigured: isElevenLabsConfigured(),
    ttsJobTimeoutMs: TTS_JOB_TIMEOUT_MS,
    ttsBatchConcurrency: TTS_BATCH_CONCURRENCY,
    ttsPrecacheItemConcurrency: TTS_PRECACHED_ITEM_CONCURRENCY,
    ttsAudioPartConcurrency: TTS_AUDIO_PART_CONCURRENCY,
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

    const items = applyKnownTermCorrections({
      targetLanguage,
      baseLanguage,
      difficulty: cleanDifficulty,
      mode: cleanMode,
      items: dedupeLessonItems(
        (parsed.items || [])
          .slice(0, cleanDesiredCount)
          .map((item) => {
            const term = String(item.term ?? "").trim();
            const meaning = String(item.meaning ?? "").trim();
            const wordType = String(item.guessedWordType ?? "other").trim();
            const promptType = String(item.promptFamily ?? "recall").trim();

            const safeExampleSentences = uniqueNonEmptyStrings(
              Array.isArray(item.safeExampleSentences)
                ? item.safeExampleSentences.map((s) => String(s ?? "").trim())
                : [term],
              2,
            );

            const exampleTranslations = uniqueNonEmptyStrings(
              Array.isArray(item.exampleTranslations)
                ? item.exampleTranslations.map((s) => String(s ?? "").trim())
                : [meaning],
              2,
            );

            return {
              term,
              meaning,
              wordType,
              promptType,
              safeExampleSentences:
                safeExampleSentences.length > 0 ? safeExampleSentences : [term],
              exampleTranslations:
                exampleTranslations.length > 0 ? exampleTranslations : [meaning],
            };
          }),
        cleanDesiredCount,
      ),
    });

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

    const fileSize = fs.existsSync(result.filePath) ? fs.statSync(result.filePath).size : 0;

    console.log("TTS response:", {
      provider: result.provider,
      voice: result.voice,
      language: result.language,
      audioUrl: result.audioUrl,
      fileSize,
    });

    res.json({
      ok: true,
      cached: result.cached,
      language: result.language,
      provider: result.provider,
      voice: result.voice,
      textLength: result.text.length,
      audioUrl: result.audioUrl,
      fileName: result.fileName,
      fileSize,
    });
  } catch (error) {
    console.error("tts error:", error);
    res.status(500).json({
      error: "TTS failed",
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

    const audioTasks = [
      {
        key: "prompt",
        required: true,
        run: () =>
          getOrCreateTtsFile({
            req,
            text: promptText,
            language: promptAudioLanguage,
            speed: cleanSpeed,
            voice,
          }),
      },
      {
        key: "target",
        required: true,
        run: () =>
          getOrCreateTtsFile({
            req,
            text: targetAudioText,
            language: targetAudioLanguage,
            speed: cleanSpeed,
            voice,
          }),
      },
      {
        key: "reveal",
        required: true,
        run: () =>
          getOrCreateTtsFile({
            req,
            text: revealText,
            language: revealAudioLanguage,
            speed: cleanSpeed,
            voice,
          }),
      },
      ...(includeExampleAudio && cleanExampleSentence
        ? [{
            key: "example",
            required: false,
            run: () =>
              getOrCreateTtsFile({
                req,
                text: cleanExampleSentence,
                language: targetLanguage,
                speed: cleanSpeed,
                voice,
              }),
          }]
        : []),
    ];

    const settledAudioTasks = await settleWithConcurrency(
      audioTasks,
      TTS_AUDIO_PART_CONCURRENCY,
      async (task) => withTimeout(task.run(), TTS_JOB_TIMEOUT_MS, `buildSessionAudio:${task.key}`),
    );

    const audioMap = Object.fromEntries(
      settledAudioTasks.map((result, index) => [audioTasks[index].key, result]),
    );

    for (const task of audioTasks) {
      const result = audioMap[task.key];
      if (task.required && result?.status !== "fulfilled") {
        throw new Error(`Failed to prepare ${task.key} audio: ${result?.reason || "unknown error"}`);
      }
    }

    const promptAudio = audioMap.prompt.value;
    const targetAudio = audioMap.target.value;
    const revealAudio = audioMap.reveal.value;
    const exampleAudio = audioMap.example?.status === "fulfilled" ? audioMap.example.value : null;

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


app.post('/ttsBatch', async (req, res) => {
  try {
    const lines = Array.isArray(req.body?.lines) ? req.body.lines : [];
    const voice = String(req.body?.voice || '').trim();
    const speed = clampNumber(req.body?.speed, 0.7, 1.35, 1.0);

    if (lines.length === 0) {
      return res.status(400).json({ error: 'lines[] is required' });
    }

    const seenBatchKeys = new Set();
    const capped = lines
      .map((line) => ({
        text: String(line?.text || '').trim(),
        language: String(line?.language || '').trim(),
      }))
      .filter((line) => line.text && line.language)
      .filter((line) => {
        const key = `${line.language.toLowerCase()}|${line.text.toLowerCase()}`;
        if (seenBatchKeys.has(key)) return false;
        seenBatchKeys.add(key);
        return true;
      })
      .slice(0, 160);

    const dedupedCapped = [];
    const seenBatchKeys = new Set();
    for (const line of capped) {
      const batchKey = `${line.language.trim().toLowerCase()}|${line.text.trim().toLowerCase()}`;
      if (seenBatchKeys.has(batchKey)) continue;
      seenBatchKeys.add(batchKey);
      dedupedCapped.push(line);
    }

    const settled = await settleWithConcurrency(dedupedCapped, TTS_BATCH_CONCURRENCY, async (line) => {
      const audio = await withTimeout(
        getOrCreateTtsFile({
          req,
          text: line.text,
          language: line.language,
          voice,
          speed,
        }),
        TTS_JOB_TIMEOUT_MS,
        `ttsBatch:${line.language}:${line.text.slice(0, 40)}`
      );
      return {
        key: `${line.language.trim().toLowerCase()}|${line.text.trim()}`,
        audioUrl: audio.audioUrl,
        language: normalizeLanguage(line.language),
        text: line.text,
      };
    });

    const items = settled
      .filter((result) => result.status === "fulfilled")
      .map((result) => result.value);
    const failures = settled
      .map((result, index) => ({ result, line: dedupedCapped[index] }))
      .filter(({ result }) => result.status === "rejected")
      .map(({ result, line }) => ({
        key: `${line.language.trim().toLowerCase()}|${line.text.trim()}`,
        language: normalizeLanguage(line.language),
        text: line.text,
        error: result.reason,
      }));

    res.json({
      ok: failures.length === 0,
      count: items.length,
      failedCount: failures.length,
      items,
      failures,
    });
  } catch (error) {
    console.error('ttsBatch error:', error);
    res.status(500).json({
      error: 'Failed to build batch TTS',
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
    const cleanDelay = clampNumber(answerDelaySeconds, 0, 15, 3);
    const cleanSpeed = clampNumber(speed, 0.5, 2.0, 1.0);

    const settledItems = await settleWithConcurrency(
      cappedItems,
      TTS_PRECACHED_ITEM_CONCURRENCY,
      async (item) => {
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

        const audioTasks = [
          {
            key: "prompt",
            required: true,
            run: () =>
              getOrCreateTtsFile({
                req,
                text: promptText,
                language: promptAudioLanguage,
                speed: cleanSpeed,
                voice,
              }),
          },
          {
            key: "target",
            required: true,
            run: () =>
              getOrCreateTtsFile({
                req,
                text: targetAudioText,
                language: targetAudioLanguage,
                speed: cleanSpeed,
                voice,
              }),
          },
          {
            key: "reveal",
            required: true,
            run: () =>
              getOrCreateTtsFile({
                req,
                text: revealText,
                language: revealAudioLanguage,
                speed: cleanSpeed,
                voice,
              }),
          },
          ...(includeExampleAudio && exampleSentence
            ? [{
                key: "example",
                required: false,
                run: () =>
                  getOrCreateTtsFile({
                    req,
                    text: exampleSentence,
                    language: targetLanguage,
                    speed: cleanSpeed,
                    voice,
                  }),
              }]
            : []),
        ];

        const settledAudio = await settleWithConcurrency(
          audioTasks,
          TTS_AUDIO_PART_CONCURRENCY,
          async (task) => withTimeout(task.run(), TTS_JOB_TIMEOUT_MS, `precache:${task.key}:${term.slice(0, 40)}`),
        );

        const audioMap = Object.fromEntries(
          settledAudio.map((result, index) => [audioTasks[index].key, result]),
        );

        const requiredFailure = audioTasks.find(
          (task) => task.required && audioMap[task.key]?.status !== "fulfilled",
        );

        if (requiredFailure) {
          throw new Error(
            `${requiredFailure.key} audio failed for "${term || meaning || "item"}": ${audioMap[requiredFailure.key]?.reason || "unknown error"}`
          );
        }

        return {
          term,
          meaning,
          promptFamily,
          answerDelaySeconds: cleanDelay,
          audio: {
            prompt: {
              text: promptText,
              language: normalizeLanguage(promptAudioLanguage),
              audioUrl: audioMap.prompt.value.audioUrl,
            },
            target: {
              text: targetAudioText,
              language: normalizeLanguage(targetAudioLanguage),
              audioUrl: audioMap.target.value.audioUrl,
            },
            reveal: {
              text: revealText,
              language: normalizeLanguage(revealAudioLanguage),
              audioUrl: audioMap.reveal.value.audioUrl,
            },
            example: audioMap.example?.status === "fulfilled"
              ? {
                  text: exampleSentence,
                  language: normalizeLanguage(targetLanguage),
                  audioUrl: audioMap.example.value.audioUrl,
                }
              : null,
          },
        };
      },
    );

    const results = settledItems
      .filter((result) => result.status === "fulfilled")
      .map((result) => result.value);
    const failures = settledItems
      .map((result, index) => ({ result, item: cappedItems[index] }))
      .filter(({ result }) => result.status === "rejected")
      .map(({ result, item }) => ({
        term: String(item.term || "").trim(),
        meaning: String(item.meaning || "").trim(),
        error: result.reason,
      }));

    res.json({
      ok: failures.length === 0,
      count: results.length,
      failedCount: failures.length,
      items: results,
      failures,
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

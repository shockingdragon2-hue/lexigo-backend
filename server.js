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

function applyKnownTermCorrections({ targetLanguage, baseLanguage, items }) {
  const normalizedTarget = normalizeLanguage(targetLanguage);
  if (!Array.isArray(items) || items.length === 0) return [];

  return items.map((item) => {
    const next = {
      ...item,
      term: sanitizeShortLabel(item.term || "", 250),
      meaning: sanitizeShortLabel(item.meaning || "", 250),
      safeExampleSentences: Array.isArray(item.safeExampleSentences)
        ? item.safeExampleSentences.map((s) => sanitizeText(s, 250))
        : [],
      exampleTranslations: Array.isArray(item.exampleTranslations)
        ? item.exampleTranslations.map((s) => sanitizeText(s, 250))
        : [],
    };

    if (normalizedTarget === "basque") {
      const correctedNumber = basqueNumberForMeaning(next.meaning);
      if (correctedNumber) {
        next.term = correctedNumber;
      }

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

function isElevenLabsConfigured() {
  return Boolean(process.env.ELEVENLABS_API_KEY);
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
  return Boolean(
    isElevenLabsConfigured() &&
      elevenLabsLanguageCode(language) &&
      elevenLabsVoiceForLanguage(language, voice),
  );
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
  - 2 exampleTranslations

Important sentence rules:
- safeExampleSentences must be in the TARGET language.
- exampleTranslations must translate the matching example sentences into the BASE language.
- Keep each translation natural, clear, and complete.
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

function getTtsCacheKey({ text, language, voice = "" }) {
  return sha256(
    JSON.stringify({
      text: sanitizeTtsText(text),
      language: normalizeLanguage(language),
      voice: String(voice || "").trim(),
    }),
  );
}

function applyPronunciationOverrides(text, language) {
  const cleanText = sanitizeTtsText(text);
  const normalized = normalizeLanguage(language);
  if (normalized === 'basque') {
    if (cleanText.toLowerCase() === 'lo egiten') {
      return '<phoneme alphabet="ipa" ph="lo eɣiten">lo egiten</phoneme>';
    }
  }
  return escapeSsml(cleanText);
}

function looksLikeQuestion(text) {
  const cleanText = sanitizeTtsText(text);
  if (!cleanText) return false;
  if (/[?¿]\s*$/.test(cleanText)) return true;

  const lower = cleanText.toLowerCase();
  return [
    'what ', 'what do ', 'what does ', 'how ', 'how do ', 'how does ', 'which ',
    'when ', 'where ', 'why ', 'who ', 'can ', 'could ', 'would ', 'should ',
    'do ', 'does ', 'did ', 'is ', 'are ', 'am ', 'will ',
    '¿',
    'que ', 'qué ', 'como ', 'cómo ', 'cual ', 'cuál ', 'donde ', 'dónde ', 'por que ', 'por qué ',
    'zer ', 'nola ', 'noiz ', 'non ',
    'comment ', 'quand ', 'où ', 'pourquoi ', 'quel ', 'quelle ', "qu'est-ce",
    'wie ', 'was ', 'wann ', 'wo ', 'warum ',
    'come ', 'che cosa ', 'cosa ', 'quando ', 'dove ', 'perché ',
    'o que ', 'como ', 'quando ', 'onde ', 'por que ', 'por quê '
  ].some((prefix) => lower.startsWith(prefix));
}

function addNaturalMidPromptPauses(text) {
  let next = sanitizeTtsText(text);
  if (!next) return next;

  const replacements = [
    [/then\s+say/gi, 'then, say'],
    [/and then\s+say/gi, 'and then, say'],
    [/listen\s+first,?\s+then/gi, 'Listen first, then'],
    [/listen,?\s+then/gi, 'Listen, then'],
    [/say it out loud/gi, 'say it out loud'],
    [/dilo en voz alta/gi, 'dilo en voz alta'],
    [/di tu respuesta en voz alta/gi, 'di tu respuesta en voz alta'],
    [/y luego/gi, 'y luego'],
    [/escucha y luego/gi, 'escucha, y luego'],
    [/entzun eta gero/gi, 'entzun, eta gero'],
    [/ecoute et puis/gi, 'écoute, et puis'],
    [/hör zu und dann/gi, 'hör zu, und dann'],
    [/ascolta e poi/gi, 'ascolta, e poi'],
  ];

  for (const [pattern, replacement] of replacements) {
    next = next.replace(pattern, replacement);
  }

  next = next.replace(/\s+,/g, ',');
  next = next.replace(/,{2,}/g, ',');
  next = next.replace(/\s{2,}/g, ' ').trim();
  return next;
}

function prepareElevenLabsText(text, language) {
  let cleanText = addNaturalMidPromptPauses(text);
  const normalized = normalizeLanguage(language);

  if (!cleanText) return cleanText;

  const bareWordLike = /^[\p{L}\p{M}0-9][\p{L}\p{M}0-9'’ -]*[\p{L}\p{M}0-9]$/u.test(cleanText);
  const tokenCount = cleanText.split(/\s+/).filter(Boolean).length;
  const hasTerminalPunctuation = /[.!?…]$/.test(cleanText);

  if (looksLikeQuestion(cleanText) && !/[?؟]$/.test(cleanText)) {
    cleanText = `${cleanText.replace(/[.!,;:…]+$/g, '').trim()}?`;
  } else if (!hasTerminalPunctuation && bareWordLike) {
    if (tokenCount <= 2 || normalized === 'basque') {
      cleanText = `${cleanText}.`;
    }
  } else if (!hasTerminalPunctuation && tokenCount >= 3 && !looksLikeQuestion(cleanText)) {
    cleanText = `${cleanText}.`;
  }

  return cleanText;
}

async function synthesizeElevenLabsToFile({
  text,
  language,
  outputPath,
  voice = "",
}) {
  const cleanText = prepareElevenLabsText(text, language);
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
        text: providerText,
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
  fs.writeFileSync(outputPath, Buffer.from(arrayBuffer));

  return {
    ok: true,
    provider: "elevenlabs",
    voice: voiceId,
  };
}

async function synthesizeAzureToFile({ text, language, outputPath }) {
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

    const cleanText = applyPronunciationOverrides(text, language);
    const langCode = azureLangCode(language);

    const ssml = `
      <speak version="1.0" xml:lang="${langCode}">
        <voice name="${voiceName}">
          ${cleanText}
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
  const providerText = shouldUseElevenLabs(normalizedLanguage, voice)
    ? prepareElevenLabsText(cleanText, normalizedLanguage)
    : cleanText;

  if (!cleanText) {
    throw new Error("Missing TTS text");
  }

  const cacheKey = getTtsCacheKey({
    text: providerText,
    language: normalizedLanguage,
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

  if (inflightTtsJobs.has(cacheKey)) {
    await inflightTtsJobs.get(cacheKey);
    return {
      ok: true,
      cached: fs.existsSync(outputPath),
      language: normalizedLanguage,
      provider: fs.existsSync(outputPath) ? "cache" : "generated",
      voice:
        voice ||
        azureVoiceForLanguage(normalizedLanguage) ||
        openAiVoiceForLanguage(normalizedLanguage) ||
        DEFAULT_OPENAI_VOICE,
      text: cleanText,
      audioUrl: makeAudioUrl(req, fileName),
    };
  }

  const job = (async () => {
    let providerInfo;

    if (shouldUseElevenLabs(normalizedLanguage, voice)) {
      try {
        providerInfo = await synthesizeElevenLabsToFile({
          text: providerText,
          language: normalizedLanguage,
          outputPath,
          voice,
        });
        return providerInfo;
      } catch (elevenError) {
        console.warn("ElevenLabs TTS failed, falling back:", elevenError.message);
      }
    }

    try {
      providerInfo = await synthesizeAzureToFile({
        text: cleanText,
        language: normalizedLanguage,
        outputPath,
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
    return providerInfo;
  })();

  inflightTtsJobs.set(cacheKey, job);

  let providerInfo;
  try {
    providerInfo = await job;
  } finally {
    inflightTtsJobs.delete(cacheKey);
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
  });
});
app.get("/health", (req, res) => {
  res.json({
    ok: true,
    uptime: process.uptime(),
    azureConfigured: isAzureConfigured(),
    elevenLabsConfigured: isElevenLabsConfigured(),
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
      items: (parsed.items || [])
        .slice(0, cleanDesiredCount)
        .map((item) => {
          const term = String(item.term ?? "").trim();
          const meaning = String(item.meaning ?? "").trim();
          const wordType = String(item.guessedWordType ?? "other").trim();
          const promptType = String(item.promptFamily ?? "recall").trim();

          let safeExampleSentences = Array.isArray(item.safeExampleSentences)
            ? item.safeExampleSentences.map((s) => String(s ?? "").trim()).filter(Boolean)
            : [];

          let exampleTranslations = Array.isArray(item.exampleTranslations)
            ? item.exampleTranslations.map((s) => String(s ?? "").trim()).filter(Boolean)
            : [];

          if (safeExampleSentences.length === 0) {
            safeExampleSentences = [term, term];
          } else if (safeExampleSentences.length === 1) {
            safeExampleSentences = [safeExampleSentences[0], safeExampleSentences[0]];
          } else {
            safeExampleSentences = safeExampleSentences.slice(0, 2);
          }

          if (exampleTranslations.length === 0) {
            exampleTranslations = [meaning, meaning];
          } else if (exampleTranslations.length === 1) {
            exampleTranslations = [exampleTranslations[0], exampleTranslations[0]];
          } else {
            exampleTranslations = exampleTranslations.slice(0, 2);
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
        .filter((item) => item.term),
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
        voice,
    });

    const targetAudio = await getOrCreateTtsFile({
      req,
      text: targetAudioText,
      language: targetAudioLanguage,
        voice,
    });

    const revealAudio = await getOrCreateTtsFile({
      req,
      text: revealText,
      language: revealAudioLanguage,
        voice,
    });

    let exampleAudio = null;

    if (includeExampleAudio && cleanExampleSentence) {
      exampleAudio = await getOrCreateTtsFile({
        req,
        text: cleanExampleSentence,
        language: targetLanguage,
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


app.post('/ttsBatch', async (req, res) => {
  try {
    const lines = Array.isArray(req.body?.lines) ? req.body.lines : [];
    const voice = String(req.body?.voice || '').trim();

    if (lines.length === 0) {
      return res.status(400).json({ error: 'lines[] is required' });
    }

    const capped = lines
      .map((line) => ({
        text: String(line?.text || '').trim(),
        language: String(line?.language || '').trim(),
      }))
      .filter((line) => line.text && line.language)
      .slice(0, 80);

    const results = await Promise.all(
      capped.map(async (line) => {
        const audio = await getOrCreateTtsFile({
          req,
          text: line.text,
          language: line.language,
          voice,
        });
        return {
          key: `${line.language.trim().toLowerCase()}|${line.text.trim()}`,
          audioUrl: audio.audioUrl,
          language: normalizeLanguage(line.language),
          text: line.text,
        };
      }),
    );

    res.json({
      ok: true,
      count: results.length,
      items: results,
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
    const results = [];

    for (const item of cappedItems) {
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
              voice,
            })
          : Promise.resolve(null),
      ]);

      results.push({
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
      });
    }

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

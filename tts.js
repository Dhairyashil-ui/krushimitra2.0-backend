"use strict";

const fs = require("fs");
const path = require("path");
const https = require("https");

const LANGUAGE_CODES = {
	hi: "hi-IN",
	"hi-in": "hi-IN",
	mr: "mr-IN",
	"mr-in": "mr-IN",
	ml: "ml-IN",
	"ml-in": "ml-IN",
	bn: "bn-IN",
	"bn-in": "bn-IN",
	te: "te-IN",
	"te-in": "te-IN",
	ta: "ta-IN",
	"ta-in": "ta-IN",
	kn: "kn-IN",
	"kn-in": "kn-IN",
	gu: "gu-IN",
	"gu-in": "gu-IN",
	or: "or-IN",
	"or-in": "or-IN",
	as: "as-IN",
	"as-in": "as-IN",
	en: "en-IN",
	"en-in": "en-IN",
	"en-us": "en-US",
	"en-gb": "en-GB",
};

const DEFAULT_LANGUAGE = "en-IN";
const GOOGLE_TTS_ENDPOINT = "https://translate.googleapis.com/translate_tts";
const MAX_CHARS_PER_REQUEST = 180; // translate_tts limit ~200 characters
const DEFAULT_SPEED = normalizeSpeed(process.env.SIMPLE_TTS_SPEED ?? 1.05);

function normalizeSpeed(value) {
	const num = Number(value);
	if (!Number.isFinite(num)) {
		return 1;
	}
	return Math.min(4, Math.max(0.25, num));
}

function normalizeLanguageCode(lang = "hi") {
	if (!lang || typeof lang !== "string") {
		return "hi-IN";
	}
	const trimmed = lang.trim().toLowerCase();
	if (LANGUAGE_CODES[trimmed]) {
		return LANGUAGE_CODES[trimmed];
	}
	const short = trimmed.split("-")[0];
	return LANGUAGE_CODES[short] || DEFAULT_LANGUAGE;
}

function chunkText(input) {
	if (input.length <= MAX_CHARS_PER_REQUEST) {
		return [input];
	}
	const parts = [];
	let current = "";
	for (const word of input.split(/\s+/)) {
		const candidate = current ? `${current} ${word}` : word;
		if (candidate.length > MAX_CHARS_PER_REQUEST) {
			if (current) {
				parts.push(current);
			}
			current = word;
		} else {
			current = candidate;
		}
	}
	if (current) {
		parts.push(current);
	}
	return parts;
}

function requestTTSChunk(textChunk, languageCode, speed) {
	return new Promise((resolve, reject) => {
		const url = new URL(GOOGLE_TTS_ENDPOINT);
		url.searchParams.set("ie", "UTF-8");
		url.searchParams.set("client", "tw-ob");
		url.searchParams.set("q", textChunk);
		url.searchParams.set("tl", languageCode);
		if (speed && Number.isFinite(speed)) {
			url.searchParams.set("ttsspeed", speed.toFixed(2));
		}

		https
			.get(url, {
				headers: {
					"User-Agent":
						"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
					Accept: "*/*",
				},
			})
			.on("response", (res) => {
				if (res.statusCode !== 200) {
					return reject(
						new Error(`Google TTS responded with status ${res.statusCode}`)
					);
				}
				const buffers = [];
				res.on("data", (chunk) => buffers.push(chunk));
				res.on("end", () => resolve(Buffer.concat(buffers)));
			})
			.on("error", reject);
	});
}

async function generateSpeech(text, lang = "hi", options = {}) {
	if (!text || typeof text !== "string") {
		throw new Error("generateSpeech: 'text' must be a non-empty string");
	}

	const trimmed = text.trim();
	if (!trimmed) {
		throw new Error("generateSpeech: text is empty after trimming");
	}

	const languageCode = normalizeLanguageCode(lang);
	const speed = normalizeSpeed(options.speed ?? DEFAULT_SPEED);
	const chunks = chunkText(trimmed);
	const audioBuffers = [];

	for (let i = 0; i < chunks.length; i += 1) {
		const chunk = chunks[i];
		console.log(
			`[Simple Google TTS] chunk ${i + 1}/${chunks.length} (${chunk.length} chars) lang=${languageCode} speed=${speed}`
		);
		const buffer = await requestTTSChunk(chunk, languageCode, speed);
		audioBuffers.push(buffer);
	}

	const merged = Buffer.concat(audioBuffers);
	const outputFile = options.outputFile;

	if (outputFile) {
		await fs.promises.mkdir(path.dirname(outputFile), { recursive: true });
		await fs.promises.writeFile(outputFile, merged);
		return outputFile;
	}

	return merged;
}

module.exports = {
	generateSpeech,
};

if (require.main === module) {
	(async () => {
		try {
			const sampleText = "Namaste! Yeh Google translate TTS ka test hai.";
			const outputPath = path.join(__dirname, "sample-simple-tts.mp3");
			await generateSpeech(sampleText, "hi-IN", { outputFile: outputPath });
			console.log("Sample speech saved to", outputPath);
		} catch (error) {
			console.error("Simple TTS test failed", error);
			process.exit(1);
		}
	})();
}


const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

// Temporary directory for audio files
const TEMP_DIR = path.join(__dirname, "temp_audio");
if (!fs.existsSync(TEMP_DIR)) {
	fs.mkdirSync(TEMP_DIR, { recursive: true });
}

async function generateSpeech(text, lang = "en", options = {}) {
	if (!text) throw new Error("No text provided");

	const speed = options.speed ? Math.floor(options.speed * 175) : 175; // Base speed ~175 wpm (slightly fast)

	// Create a unique filename
	// Create a unique filename
	// Check if output file is provided in options
	let outputFilePath;
	if (options.outputFile) {
		outputFilePath = options.outputFile;
		// Ensure dir exists
		const dir = path.dirname(outputFilePath);
		if (!fs.existsSync(dir)) {
			fs.mkdirSync(dir, { recursive: true });
		}
	} else {
		const timestamp = Date.now();
		const filename = `tts_${timestamp}.mp3`; // Use mp3 for Edge TTS
		outputFilePath = path.join(TEMP_DIR, filename);
	}

	return new Promise((resolve, reject) => {
		const pythonProcess = spawn('python', [
			'scripts/tts_engine.py',
			text,
			outputFilePath,
			'--speed', String(speed),
			'--lang', lang
		]);

		let errorOutput = '';

		pythonProcess.stderr.on('data', (data) => {
			errorOutput += data.toString();
		});

		pythonProcess.on('close', async (code) => {
			if (code !== 0) {
				console.error("TTS Python Error:", errorOutput);
				return reject(new Error("TTS generation failed: " + errorOutput));
			}

			try {
				if (fs.existsSync(outputFilePath)) {
					// Return the PATH, not the buffer, as expected by server.js
					resolve(outputFilePath);
				} else {
					reject(new Error("Audio file was not created by Python script"));
				}
			} catch (err) {
				reject(err);
			}
		});
	});
}

module.exports = { generateSpeech };

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


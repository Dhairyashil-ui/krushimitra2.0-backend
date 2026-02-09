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
	const timestamp = Date.now();
	const filename = `tts_${timestamp}.wav`; // pyttsx3 creates wav usually
	const outputFilePath = path.join(TEMP_DIR, filename);

	return new Promise((resolve, reject) => {
		const pythonProcess = spawn('python', [
			'scripts/tts_engine.py',
			text,
			outputFilePath,
			'--speed', String(speed)
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
					const audioBuffer = await fs.promises.readFile(outputFilePath);
					// Optionally clean up immediately or via a cron job
					// await fs.promises.unlink(outputFilePath); 
					resolve(audioBuffer);
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


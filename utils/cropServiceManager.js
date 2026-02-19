const { spawn } = require('child_process');
const path = require('path');
const fetch = require('node-fetch'); // Ensure node-fetch is available or use built-in fetch in Node 18+
const { logger } = require('../logger');

class CropServiceManager {
    constructor() {
        this.pythonProcess = null;
        this.serviceUrl = 'http://127.0.0.1:5002';
        this.scriptPath = path.join(__dirname, '..', 'scripts', 'crop_service.py');
        this.isStarting = false;
        this.maxRetries = 5;
    }

    async startService() {
        if (this.pythonProcess || this.isStarting) {
            return;
        }

        this.isStarting = true;
        logger.info('Starting Crop Disease Analysis Service...');

        try {
            this.pythonProcess = spawn('python', [this.scriptPath], {
                stdio: ['ignore', 'pipe', 'pipe'], // Capture stdout/stderr
                detached: false
            });

            this.pythonProcess.stdout.on('data', (data) => {
                logger.info(`Validating Crop Service: ${data.toString().trim()}`);
            });

            this.pythonProcess.stderr.on('data', (data) => {
                logger.error(`Crop Service Error: ${data.toString().trim()}`);
            });

            this.pythonProcess.on('close', (code) => {
                logger.warn(`Crop Service exited with code ${code}`);
                this.pythonProcess = null;
                this.isStarting = false;
            });

            // Wait for health check to pass
            const healthy = await this.waitForHealth(20000); // 20s timeout for model loading
            if (healthy) {
                logger.info('✅ Crop Disease Service is ready.');
            } else {
                logger.error('❌ Crop Disease Service failed to start (timeout).');
                this.stopService();
            }

        } catch (error) {
            logger.error('Failed to spawn Crop Service:', error);
        } finally {
            this.isStarting = false;
        }
    }

    stopService() {
        if (this.pythonProcess) {
            logger.info('Stopping Crop Service...');
            this.pythonProcess.kill();
            this.pythonProcess = null;
        }
    }

    async waitForHealth(timeoutMs) {
        const start = Date.now();
        while (Date.now() - start < timeoutMs) {
            try {
                const res = await fetch(`${this.serviceUrl}/health`);
                if (res.ok) {
                    return true;
                }
            } catch (e) {
                // Ignore connection errors while starting
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        return false;
    }

    async analyze(imagePath) {
        // Ensure service is running
        if (!this.pythonProcess) {
            await this.startService();
        }

        // Double check health fast
        try {
            const health = await fetch(`${this.serviceUrl}/health`, { timeout: 1000 });
            if (!health.ok) throw new Error("Service unhealthy");
        } catch (e) {
            logger.warn("Crop service not responding, attempting restart...");
            this.stopService();
            await this.startService();
        }

        try {
            const response = await fetch(`${this.serviceUrl}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_path: imagePath }),
                timeout: 10000 // 10s timeout for analysis
            });

            if (!response.ok) {
                throw new Error(`Service returned ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            logger.error('Error querying Crop Service:', error);
            throw error;
        }
    }
}

const cropServiceManager = new CropServiceManager();
module.exports = { cropServiceManager };

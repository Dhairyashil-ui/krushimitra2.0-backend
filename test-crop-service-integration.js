const { cropServiceManager } = require('./utils/cropServiceManager');
const { logger } = require('./logger');

// Mock logger to see output
logger.info = console.log;
logger.error = console.error;
logger.warn = console.warn;

async function testService() {
    console.log('Testing Crop Service Integration...');

    try {
        console.log('1. Starting Service...');
        await cropServiceManager.startService();
        console.log('Service started.');

        console.log('2. Checking Health...');
        const healthy = await cropServiceManager.waitForHealth(5000);
        if (healthy) {
            console.log('✅ Health Check Passed');
        } else {
            console.error('❌ Health Check Failed');
        }

        // We can't easily test image analysis without a valid image file and ML libraries installed in this environment
        // But we can verify the service is running and responding to health checks.

        console.log('3. Stopping Service...');
        cropServiceManager.stopService();
        console.log('Service stopped.');

    } catch (error) {
        console.error('Test failed:', error);
        cropServiceManager.stopService();
    }
}

testService();

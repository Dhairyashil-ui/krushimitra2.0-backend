require('dotenv').config();

// Configuration for PlantNet API
// ⚠️ Later, ensure PLANTNET_API_KEY is set in your .env file for security.
const config = {
    PLANTNET_API_KEY: process.env.PLANTNET_API_KEY || "PASTE_YOUR_API_KEY_HERE",
    PLANTNET_URL: "https://my-api.plantnet.org/v2/identify/all"
};

module.exports = config;

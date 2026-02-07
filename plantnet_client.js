const fs = require('fs');
const FormData = require('form-data');
const fetch = require('node-fetch');
const config = require('./config');

/**
 * Uses PlantNet API to identify plant from image
 * @param {string} imagePath - Path to the image file
 * @param {string} organ - Organ type: leaf | fruit | flower | bark (default: leaf)
 * @returns {Promise<Object>} - Result object
 */
async function identifyPlant(imagePath, organ = "leaf") {
    try {
        if (!fs.existsSync(imagePath)) {
            return {
                success: false,
                message: "Image file not found"
            };
        }

        const formData = new FormData();
        formData.append('images', fs.createReadStream(imagePath));
        formData.append('organs', organ);

        const url = `${config.PLANTNET_URL}?api-key=${config.PLANTNET_API_KEY}`;

        console.log(`🌿 Identifying plant with PlantNet... Organ: ${organ}`);

        const response = await fetch(url, {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders(),
            timeout: 30000 // 30 seconds timeout
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('PlantNet API Error:', errorText);
            return {
                success: false,
                message: "PlantNet API error",
                details: errorText
            };
        }

        const data = await response.json();

        if (!data.results || data.results.length === 0) {
            return {
                success: true,
                plant: "Unknown Plant",
                confidence: 0.0,
                source: "PlantNet"
            };
        }

        const best = data.results[0];

        // Extract common name safely
        const commonName = (best.species.commonNames && best.species.commonNames.length > 0)
            ? best.species.commonNames[0]
            : "Unknown";

        return {
            success: true,
            plant_scientific: best.species.scientificNameWithoutAuthor,
            plant_common: commonName,
            confidence: Number(best.score.toFixed(3)),
            source: "PlantNet",
            raw_data: best // Include raw result for debugging/details if needed
        };

    } catch (error) {
        console.error('Error in identifyPlant:', error);
        return {
            success: false,
            message: "Internal server error during identification",
            details: error.message
        };
    }
}

module.exports = { identifyPlant };

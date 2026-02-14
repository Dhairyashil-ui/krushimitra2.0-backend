/**
 * AI Database Query Helper (RAG Implementation)
 * 
 * This module provides helper functions for the AI to interact with the database
 * to retrieve Real Context (RAG) from the 6 key features.
 */

const queryTemplates = require('./ai-db-queries.json');

/**
 * Get latest mandi prices for a crop in a location
 * @param {Collection} collection - MongoDB mandiprices collection
 * @param {string} crop - Crop name (optional)
 * @param {string} location - Location name (optional)
 * @returns {Promise<Array>} Array of latest mandi price objects
 */
async function getLatestMandiPrices(collection, crop, location) {
  if (!collection) return [];
  // console.log(`RAG: Fetching mandi prices for ${crop || 'all'} in ${location || 'all'}`);

  const query = {};
  if (crop) query.crop = new RegExp(crop, 'i');
  if (location) query.location = new RegExp(location, 'i');

  try {
    return await collection.find(query).sort({ date: -1 }).limit(5).toArray();
  } catch (e) {
    console.error("RAG Error (Mandi):", e.message);
    return [];
  }
}

/**
 * Get active government schemes for a farmer's location
 * @param {Collection} collection - MongoDB schemes collection
 * @param {string} farmerLocation - Farmer's location/state
 * @returns {Promise<Array>} Array of active scheme objects
 */
async function getActiveSchemes(collection, farmerLocation) {
  if (!collection) return [];
  // console.log(`RAG: Fetching schemes for ${farmerLocation}`);

  const query = {};

  if (farmerLocation) {
    query.$or = [
      { location: 'all' },
      { location: 'All' },
      { location: new RegExp(farmerLocation, 'i') }
    ];
  } else {
    query.location = { $in: ['all', 'All'] };
  }

  try {
    return await collection.find(query).limit(5).toArray();
  } catch (e) {
    console.error("RAG Error (Schemes):", e.message);
    return [];
  }
}

/**
 * Get recent crop health diagnoses for a farmer
 * @param {Collection} collection - MongoDB crop_health collection
 * @param {string} farmerId - Farmer's ID
 * @returns {Promise<Array>} Array of recent crop health diagnosis objects
 */
async function getCropHealthHistory(collection, farmerId) {
  if (!collection || !farmerId) return [];
  // console.log(`RAG: Fetching health history for ${farmerId}`);

  try {
    return await collection.find({ farmerId }).sort({ timestamp: -1 }).limit(3).toArray();
  } catch (e) {
    console.error("RAG Error (CropHealth):", e.message);
    return [];
  }
}

/**
 * Get recent farming activities
 * @param {Collection} collection - MongoDB activities collection
 * @param {string} farmerId - Farmer's ID
 * @returns {Promise<Array>} Array of recent activities
 */
async function getRecentActivities(collection, farmerId) {
  if (!collection || !farmerId) return [];
  // console.log(`RAG: Fetching activities for ${farmerId}`);

  try {
    return await collection.find({ farmerId }).sort({ date: -1 }).limit(5).toArray();
  } catch (e) {
    console.error("RAG Error (Activities):", e.message);
    return [];
  }
}

module.exports = {
  getLatestMandiPrices,
  getActiveSchemes,
  getCropHealthHistory,
  getRecentActivities,
  queryTemplates
};
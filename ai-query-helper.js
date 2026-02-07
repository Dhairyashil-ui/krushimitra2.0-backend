/**
 * AI Database Query Helper
 * 
 * This module provides helper functions for the AI to interact with the database
 * using natural language-like query templates.
 */



const queryTemplates = require('./ai-db-queries.json');

/**
 * Get farmer by phone number
 * @param {string} phone - Farmer's phone number
 * @returns {Promise<Object|null>} Farmer object or null if not found
 */
async function getFarmerByPhone(phone) {
  // Template: FIND farmers WHERE phone = {phone}
  console.log(`Executing: FIND farmers WHERE phone = ${phone}`);

  // In a real implementation, this would query the database:
  // return await db.farmers.findOne({ phone });

  // For demo purposes, return mock data
  return {
    _id: 'farmer123',
    name: 'Test Farmer',
    phone: phone,
    language: 'Hindi',
    location: 'Pune, Maharashtra',
    crops: ['Wheat', 'Soybean'],
    joinedAt: new Date()
  };
}

/**
 * Save activity for a farmer
 * @param {Object} activityData - Activity data to save
 * @returns {Promise<Object>} Inserted activity object
 */
async function saveActivity(activityData) {
  const { farmerId, activityType, description, date, aiSuggestions } = activityData;

  // Template: INSERT into activities { farmerId, activityType, description, date, aiSuggestions }
  console.log(`Executing: INSERT into activities { farmerId: ${farmerId}, activityType: ${activityType}, description: ${description}, date: ${date}, aiSuggestions: ${JSON.stringify(aiSuggestions)} }`);

  // In a real implementation, this would insert into the database:
  // return await db.activities.insertOne(activityData);

  // For demo purposes, return mock data
  return {
    _id: 'activity456',
    ...activityData
  };
}

/**
 * Get latest mandi prices for a crop in a location
 * @param {string} crop - Crop name
 * @param {string} location - Location name
 * @returns {Promise<Array>} Array of latest mandi price objects
 */
async function getLatestMandiPrices(crop, location) {
  // Template: FIND mandiprices WHERE crop = {crop} AND location = {location} SORT date DESC LIMIT 5
  console.log(`Executing: FIND mandiprices WHERE crop = ${crop} AND location = ${location} SORT date DESC LIMIT 5`);

  // In a real implementation, this would query the database:
  // return await db.mandiprices.find({ crop, location }).sort({ date: -1 }).limit(5);

  // For demo purposes, return mock data
  return [
    {
      _id: 'price1',
      crop,
      location,
      price: 2400,
      date: new Date()
    }
  ];
}

/**
 * Check if current mandi price is a golden chance (>10% above 7-day average)
 * @param {string} crop - Crop name
 * @param {string} location - Location name
 * @returns {Promise<boolean>} True if current price is >10% above 7-day average
 */
async function checkGoldenChance(crop, location) {
  // Template: COMPUTE 7-day-average price and compare current price; if > 10% then true
  console.log(`Executing: COMPUTE 7-day-average price for ${crop} in ${location} and compare with latest price`);

  // In a real implementation, this would:
  // 1. Get the latest price
  // 2. Get prices from the last 7 days
  // 3. Calculate average
  // 4. Compare and return boolean

  // For demo purposes, return mock data
  return true;
}

/**
 * Fetch recent AI interactions for a farmer to provide context
 * @param {string} farmerId - Farmer's ID
 * @returns {Promise<Array>} Array of recent AI interaction objects
 */
async function fetchFarmerMemoryForChat(farmerId) {
  // Template: FIND aiinteractions WHERE farmerId = {farmerId} SORT timestamp DESC LIMIT 10
  console.log(`Executing: FIND aiinteractions WHERE farmerId = ${farmerId} SORT timestamp DESC LIMIT 10`);

  // In a real implementation, this would query the database:
  // return await db.aiinteractions.find({ farmerId }).sort({ timestamp: -1 }).limit(10);

  // For demo purposes, return mock data
  return [
    {
      _id: 'interaction1',
      farmerId,
      question: 'What crops should I plant this season?',
      response: 'Based on your location and current weather patterns, I recommend planting wheat or barley.',
      timestamp: new Date()
    }
  ];
}

/**
 * Get active government schemes for a farmer's location
 * @param {string} farmerLocation - Farmer's location
 * @returns {Promise<Array>} Array of active scheme objects
 */
async function getActiveSchemes(farmerLocation) {
  // Template: FIND schemes WHERE startDate <= {currentDate} AND endDate >= {currentDate} AND (location = {farmerLocation} OR location = 'all')
  console.log(`Executing: FIND schemes WHERE startDate <= ${new Date().toISOString()} AND endDate >= ${new Date().toISOString()} AND (location = ${farmerLocation} OR location = 'all')`);

  // In a real implementation, this would query the database:
  // return await db.schemes.find({ 
  //   startDate: { $lte: new Date() }, 
  //   endDate: { $gte: new Date() },
  //   $or: [{ location: farmerLocation }, { location: 'all' }]
  // });

  // For demo purposes, return mock data
  return [
    {
      _id: 'scheme1',
      title: 'PM Kisan Samman Nidhi',
      description: 'Financial assistance to small and marginal farmer families',
      startDate: new Date('2023-01-01'),
      endDate: new Date('2023-12-31'),
      location: 'all'
    }
  ];
}

/**
 * Get recent crop health diagnoses for a farmer
 * @param {string} farmerId - Farmer's ID
 * @returns {Promise<Array>} Array of recent crop health diagnosis objects
 */
async function getCropHealthHistory(farmerId) {
  // Template: FIND crop_health WHERE farmerId = {farmerId} SORT diagnosedAt DESC LIMIT 5
  console.log(`Executing: FIND crop_health WHERE farmerId = ${farmerId} SORT diagnosedAt DESC LIMIT 5`);

  // In a real implementation, this would query the database:
  // return await db.crop_health.find({ farmerId }).sort({ diagnosedAt: -1 }).limit(5);

  // For demo purposes, return mock data
  return [
    {
      _id: 'diagnosis1',
      farmerId,
      detectedIssue: 'Yellow Rust',
      solution: 'Apply fungicide treatment and ensure proper drainage',
      confidence: 0.85,
      diagnosedAt: new Date()
    }
  ];
}

/**
 * Get recent weather alerts for a farmer
 * @param {string} farmerId - Farmer's ID
 * @returns {Promise<Array>} Array of recent active weather alert objects
 */
async function getWeatherAlerts(farmerId) {
  // Template: FIND alerts WHERE farmerId = {farmerId} AND type = 'weather' AND status = 'active' SORT createdAt DESC LIMIT 3
  console.log(`Executing: FIND alerts WHERE farmerId = ${farmerId} AND type = 'weather' AND status = 'active' SORT createdAt DESC LIMIT 3`);

  // In a real implementation, this would query the database:
  // return await db.alerts.find({ 
  //   farmerId, 
  //   type: 'weather', 
  //   status: 'active' 
  // }).sort({ createdAt: -1 }).limit(3);

  // For demo purposes, return mock data
  return [
    {
      _id: 'alert1',
      farmerId,
      type: 'weather',
      message: 'Heavy rainfall expected in your area. Take necessary precautions for your crops.',
      status: 'active',
      createdAt: new Date()
    }
  ];
}

module.exports = {
  getFarmerByPhone,
  saveActivity,
  getLatestMandiPrices,
  checkGoldenChance,
  fetchFarmerMemoryForChat,
  getActiveSchemes,
  getCropHealthHistory,
  getWeatherAlerts,
  queryTemplates
};
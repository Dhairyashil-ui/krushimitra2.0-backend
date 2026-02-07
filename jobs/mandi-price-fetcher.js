/**
 * Mandi Price Fetcher Job
 * 
 * This background job runs every hour to fetch external mandi price data,
 * insert it into the database, compute golden chances, and create alerts
 * for farmers who have opted for price alerts.
 */

const { MongoClient } = require('mongodb');
require('dotenv').config();

// MongoDB connection
const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
const client = new MongoClient(uri);

// Job execution statistics
let stats = {
  pricesFetched: 0,
  pricesInserted: 0,
  goldenChancesDetected: 0,
  alertsCreated: 0,
  farmersNotified: 0
};

// Sources for mandi price data
const sources = [
  {
    name: "Maharashtra Mandi Board",
    url: "https://maharashtramandiboard.example.com/api/prices",
    enabled: true
  },
  {
    name: "National Agriculture Market",
    url: "https://nammarket.example.com/api/mandiprices",
    enabled: true
  }
];

/**
 * Fetch external mandi price data from a source
 * @param {Object} source - The data source configuration
 * @returns {Promise<Array>} Array of price data
 */
async function fetchExternalPrices(source) {
  console.log(`Fetching prices from ${source.name}...`);
  
  // In a real implementation, this would make HTTP requests to external APIs
  // For demo purposes, we'll return mock data
  
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Mock data
  const mockPrices = [
    {
      crop: "Wheat",
      location: "Pune",
      price: 2400,
      date: new Date()
    },
    {
      crop: "Soybean",
      location: "Pune",
      price: 5200,
      date: new Date()
    },
    {
      crop: "Rice",
      location: "Nagpur",
      price: 2800,
      date: new Date()
    }
  ];
  
  stats.pricesFetched += mockPrices.length;
  console.log(`Fetched ${mockPrices.length} prices from ${source.name}`);
  
  return mockPrices;
}

/**
 * Insert mandi prices into the database
 * @param {Array} prices - Array of price objects to insert
 * @returns {Promise<void>}
 */
async function insertPrices(prices) {
  try {
    const db = client.db("KrushiMitraDB");
    const collection = db.collection("mandiprices");
    
    if (prices.length > 0) {
      const result = await collection.insertMany(prices);
      stats.pricesInserted += result.insertedCount;
      console.log(`Inserted ${result.insertedCount} prices into database`);
    }
  } catch (error) {
    console.error("Error inserting prices:", error);
    throw error;
  }
}

/**
 * Calculate 7-day average price for a crop in a location
 * @param {string} crop - Crop name
 * @param {string} location - Location name
 * @returns {Promise<number>} Average price
 */
async function calculate7DayAverage(crop, location) {
  try {
    const db = client.db("KrushiMitraDB");
    const collection = db.collection("mandiprices");
    
    // Calculate date 7 days ago
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    
    // Find prices from the last 7 days
    const prices = await collection.find({
      crop: crop,
      location: location,
      date: { $gte: sevenDaysAgo }
    }).toArray();
    
    if (prices.length === 0) {
      return 0;
    }
    
    // Calculate average
    const sum = prices.reduce((acc, price) => acc + price.price, 0);
    return sum / prices.length;
  } catch (error) {
    console.error("Error calculating 7-day average:", error);
    throw error;
  }
}

/**
 * Check if current price is a golden chance (>10% above 7-day average)
 * @param {number} currentPrice - Current price
 * @param {number} averagePrice - 7-day average price
 * @returns {boolean} True if golden chance
 */
function isGoldenChance(currentPrice, averagePrice) {
  if (averagePrice === 0) return false;
  const percentageIncrease = ((currentPrice - averagePrice) / averagePrice) * 100;
  return percentageIncrease > 10;
}

/**
 * Find farmers who grow a specific crop in a location and have opted for price alerts
 * @param {string} crop - Crop name
 * @param {string} location - Location name
 * @returns {Promise<Array>} Array of farmer objects
 */
async function findEligibleFarmers(crop, location) {
  try {
    const db = client.db("KrushiMitraDB");
    const collection = db.collection("farmers");
    
    const farmers = await collection.find({
      crops: crop,
      location: { $regex: new RegExp(location, 'i') },
      priceAlerts: true
    }).toArray();
    
    return farmers;
  } catch (error) {
    console.error("Error finding eligible farmers:", error);
    throw error;
  }
}

/**
 * Create price alerts for farmers
 * @param {Array} farmers - Array of farmer objects
 * @param {Object} priceData - Price data including crop, location, price
 * @returns {Promise<void>}
 */
async function createPriceAlerts(farmers, priceData) {
  try {
    const db = client.db("KrushiMitraDB");
    const collection = db.collection("alerts");
    
    // Calculate 7-day average for this crop and location
    const averagePrice = await calculate7DayAverage(priceData.crop, priceData.location);
    const goldenChance = isGoldenChance(priceData.price, averagePrice);
    
    if (goldenChance) {
      stats.goldenChancesDetected++;
    }
    
    // Create alerts for each farmer
    for (const farmer of farmers) {
      const alert = {
        farmerId: farmer._id,
        type: "price",
        message: goldenChance 
          ? `Great opportunity! ${priceData.crop} prices in ${priceData.location} are ${((priceData.price - averagePrice) / averagePrice * 100).toFixed(2)}% above the 7-day average at ₹${priceData.price} per quintal`
          : `Price update for ${priceData.crop} in ${priceData.location}: ₹${priceData.price} per quintal`,
        status: "active",
        createdAt: new Date(),
        metadata: {
          crop: priceData.crop,
          location: priceData.location,
          price: priceData.price,
          isGoldenChance: goldenChance,
          previousAverage: averagePrice,
          percentageIncrease: goldenChance ? ((priceData.price - averagePrice) / averagePrice * 100) : 0
        }
      };
      
      await collection.insertOne(alert);
      stats.alertsCreated++;
    }
    
    stats.farmersNotified += farmers.length;
    console.log(`Created ${farmers.length} price alerts for ${priceData.crop} in ${priceData.location}`);
  } catch (error) {
    console.error("Error creating price alerts:", error);
    throw error;
  }
}

/**
 * Main function to run the mandi price fetcher job
 * @returns {Promise<Object>} Job execution report
 */
async function runMandiPriceFetcher() {
  const startTime = new Date();
  console.log("Starting Mandi Price Fetcher job...");
  
  // Reset statistics
  stats = {
    pricesFetched: 0,
    pricesInserted: 0,
    goldenChancesDetected: 0,
    alertsCreated: 0,
    farmersNotified: 0
  };
  
  try {
    // Connect to database
    await client.connect();
    console.log("Connected to database");
    
    // Process each data source
    for (const source of sources) {
      if (!source.enabled) {
        console.log(`Skipping disabled source: ${source.name}`);
        continue;
      }
      
      try {
        // Fetch prices from external source
        const prices = await fetchExternalPrices(source);
        
        // Insert prices into database
        await insertPrices(prices);
        
        // Process each price for golden chance detection and alerts
        for (const price of prices) {
          // Find eligible farmers
          const farmers = await findEligibleFarmers(price.crop, price.location);
          
          if (farmers.length > 0) {
            // Create alerts for farmers
            await createPriceAlerts(farmers, price);
          }
        }
      } catch (sourceError) {
        console.error(`Error processing source ${source.name}:`, sourceError);
        // Continue with other sources even if one fails
      }
    }
    
    const endTime = new Date();
    const durationSeconds = (endTime - startTime) / 1000;
    
    // Create success report
    const report = {
      jobName: "Mandi Price Fetcher",
      status: "SUCCESS",
      startedAt: startTime,
      completedAt: endTime,
      durationSeconds: durationSeconds,
      statistics: { ...stats },
      details: {
        sources: sources.map(source => ({
          name: source.name,
          status: "SUCCESS"
        }))
      }
    };
    
    console.log("Mandi Price Fetcher job completed successfully");
    return report;
  } catch (error) {
    const endTime = new Date();
    const durationSeconds = (endTime - startTime) / 1000;
    
    // Create failure report
    const report = {
      jobName: "Mandi Price Fetcher",
      status: "FAILURE",
      startedAt: startTime,
      completedAt: endTime,
      durationSeconds: durationSeconds,
      error: {
        code: "JOB_EXECUTION_ERROR",
        message: error.message,
        details: error.stack
      },
      statistics: { ...stats }
    };
    
    console.error("Mandi Price Fetcher job failed:", error);
    return report;
  } finally {
    // Close database connection
    await client.close();
    console.log("Database connection closed");
  }
}

// Export the job function
module.exports = { runMandiPriceFetcher };

// Run the job if this file is executed directly
if (require.main === module) {
  runMandiPriceFetcher()
    .then(report => {
      console.log("Job Report:", JSON.stringify(report, null, 2));
      process.exit(0);
    })
    .catch(error => {
      console.error("Job failed with unhandled error:", error);
      process.exit(1);
    });
}
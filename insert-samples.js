const { connectToDatabase } = require('./db');
const { logger } = require('./logger');

async function main() {
  try {
    logger.info('Inserting sample data');

    const client = await connectToDatabase('write');
    const db = client.db("KrushiMitraDB");


    // Sample data
    const farmersCollection = db.collection('farmers');
    const activitiesCollection = db.collection('activities');
    const mandipricesCollection = db.collection('mandiprices');
    const schemesCollection = db.collection('schemes');
    const aiinteractionsCollection = db.collection('aiinteractions');
    const cropHealthCollection = db.collection('crop_health');
    const alertsCollection = db.collection('alerts');

    // Create geospatial index for mandi prices
    await mandipricesCollection.createIndex({ location: "2dsphere" });
    logger.info('Created 2dsphere index on mandiprices.location');


    // Insert sample farmer
    const sampleFarmer = {
      name: "Rajesh Kumar",
      phone: "+919876543210",
      language: "Hindi",
      location: "Pune, Maharashtra",
      crops: ["Wheat", "Soybean"],
      landSize: 5.5,
      soilType: "Black soil",
      joinedAt: new Date(),
      updatedAt: new Date()
    };

    const farmerResult = await farmersCollection.insertOne(sampleFarmer);
    logger.info('Sample farmer inserted', { farmerId: farmerResult.insertedId.toString() });

    // Insert sample activity
    const sampleActivity = {
      farmerId: farmerResult.insertedId,
      description: "Installed app / first chat",
      type: "app_install",
      date: new Date(),
      details: {}
    };

    const activityResult = await activitiesCollection.insertOne(sampleActivity);
    logger.info('Sample activity inserted', { activityId: activityResult.insertedId.toString() });

    // Real Pune Mandi Data
    const realPuneMandis = [
      // Major Wholesale Mandies (APMC)
      { name: "Gultekdi Market Yard", location: "Maharshi Nagar", coords: [73.8700, 18.4900] },
      { name: "Pimpri Sub-Market Yard", location: "Pimpri", coords: [73.8070, 18.6229] },
      { name: "Moshi Market Yard", location: "Moshi", coords: [73.8468, 18.6623] },
      { name: "Chakan Market Yard", location: "Chakan", coords: [73.8550, 18.7561] },
      { name: "Manjari APMC Market", location: "Manjari", coords: [73.9514, 18.5140] },

      // Local and Historic Retail Mandies
      { name: "Mahatma Phule Mandai", location: "Shukrawar Peth", coords: [73.8562, 18.5129] },
      { name: "Hadapsar Bhaji Mandai", location: "Hadapsar", coords: [73.9340, 18.5004] },
      { name: "Shivaji Bhaji Market", location: "Shivajinagar", coords: [73.8443, 18.5240] },
      { name: "Khadki Old Bhaji Market", location: "Khadki", coords: [73.8324, 18.5615] },
      { name: "Bhosari Vegetable Market", location: "Bhosari", coords: [73.8400, 18.6200] },
      { name: "Mundhwa Vegetable Market", location: "Mundhwa", coords: [73.9234, 18.5284] },

      // Major Weekly Farmers' Markets
      { name: "Aundh Farmers' Market", location: "Aundh", coords: [73.8129, 18.5631] },
      { name: "Kothrud Farmers' Market", location: "Kothrud", coords: [73.8093, 18.5011] },
      { name: "Wanowrie Farmers' Market", location: "Wanowrie", coords: [73.8960, 18.4890] },
      { name: "Sinhgad Road Farmers' Market", location: "Sinhgad Road", coords: [73.8236, 18.4795] }
    ];

    const crops = [
      { name: "Onion", priceBase: 35, variance: 5, category: "Vegetables", unit: "per kg" },
      { name: "Tomato", priceBase: 45, variance: 10, category: "Vegetables", unit: "per kg" },
      { name: "Potato", priceBase: 30, variance: 5, category: "Vegetables", unit: "per kg" },
      { name: "Wheat", priceBase: 2400, variance: 100, category: "Cereals", unit: "per quintal" },
      { name: "Soybean", priceBase: 4800, variance: 200, category: "Pulses", unit: "per quintal" }
    ];

    const sampleMandiPrices = [];

    // Generate price entries for each mandi and crop
    realPuneMandis.forEach(mandi => {
      crops.forEach(crop => {
        // Add some random variance to price
        const price = crop.priceBase + Math.floor(Math.random() * crop.variance * 2) - crop.variance;

        sampleMandiPrices.push({
          crop: crop.name,
          location: {
            type: "Point",
            coordinates: mandi.coords, // [Longitude, Latitude]
            name: mandi.name,
            address: mandi.location
          },
          price: price,
          date: new Date(),
          category: crop.category,
          unit: crop.unit,
          change: Math.floor(Math.random() * 10) - 5, // Random change between -5 and 5
          changePercent: (Math.random() * 10 - 5).toFixed(2) // Random % change
        });
      });
    });

    const mandiPricesResult = await mandipricesCollection.insertMany(sampleMandiPrices);
    logger.info('Sample mandi prices inserted', { count: mandiPricesResult.insertedCount });

    // Insert sample scheme
    const sampleScheme = {
      title: "PM Kisan Samman Nidhi",
      description: "Financial assistance to small and marginal farmer families",
      eligibility: "Landholding farmers with less than 2 hectares",
      benefits: "₹6000 per year in 3 equal installments",
      applicationProcess: "Through Common Service Centers or Banks",
      deadline: new Date("2024-03-31"),
      category: "financial"
    };

    const schemeResult = await schemesCollection.insertOne(sampleScheme);
    logger.info('Sample scheme inserted', { schemeId: schemeResult.insertedId.toString() });

    // Insert sample AI interaction
    const sampleAIInteraction = {
      farmerId: farmerResult.insertedId,
      query: "What are the current mandi prices for wheat in Pune?",
      response: "Current mandi price for wheat in Pune is ₹2200 per quintal",
      timestamp: new Date(),
      context: {
        crop: "Wheat",
        location: "Pune"
      }
    };

    const aiInteractionResult = await aiinteractionsCollection.insertOne(sampleAIInteraction);
    logger.info('Sample AI interaction inserted', { interactionId: aiInteractionResult.insertedId.toString() });

    // Insert sample crop health record
    const sampleCropHealth = {
      farmerId: farmerResult.insertedId,
      crop: "Wheat",
      imageUrl: "https://example.com/image1.jpg",
      diagnosis: "Healthy crop",
      confidence: 0.95,
      recommendations: ["Continue regular watering", "Apply nitrogen fertilizer"],
      detectedAt: new Date()
    };

    const cropHealthResult = await cropHealthCollection.insertOne(sampleCropHealth);
    logger.info('Sample crop health record inserted', { healthRecordId: cropHealthResult.insertedId.toString() });

    // Insert sample alert
    const sampleAlert = {
      farmerId: farmerResult.insertedId,
      type: "scheme",
      message: "New scheme available: PM Kisan Samman Nidhi",
      status: "active",
      priority: "medium",
      createdAt: new Date()
    };

    const alertResult = await alertsCollection.insertOne(sampleAlert);
    logger.info('Sample alert inserted', { alertId: alertResult.insertedId.toString() });

    // Return inserted IDs
    const insertedIds = {
      farmerId: farmerResult.insertedId,
      activityId: activityResult.insertedId,
      mandiPricesIds: mandiPricesResult.insertedIds,
      schemeId: schemeResult.insertedId,
      aiInteractionId: aiInteractionResult.insertedId,
      cropHealthId: cropHealthResult.insertedId,
      alertId: alertResult.insertedId
    };

    logger.info('All sample data inserted successfully');
    console.log(JSON.stringify(insertedIds, null, 2));
  } catch (error) {
    logger.error('Error inserting sample data', { error: error.message });
    console.error('Error:', error.message);
    process.exit(1);
  } finally {
    // Close the client
    // Note: In a real application, you might want to keep the connection open
    // or use a connection pool. For this script, we'll close it.
  }
}

main();
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
    
    // Insert sample mandi prices
    const sampleMandiPrices = [
      {
        crop: "Wheat",
        location: "Pune",
        price: 2200,
        date: new Date()
      },
      {
        crop: "Soybean",
        location: "Pune",
        price: 4500,
        date: new Date()
      }
    ];
    
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
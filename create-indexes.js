const { MongoClient } = require('mongodb');
require('dotenv').config();

// Check if environment variables are set
if (!process.env.DB_USER || !process.env.DB_PASS || !process.env.CLUSTER_HOST) {
  console.log(JSON.stringify({
    status: "ERROR_CONNECT",
    message: "Missing environment variables: DB_USER, DB_PASS, and/or CLUSTER_HOST must be set in .env file"
  }, null, 2));
  process.exit(1);
}

// MongoDB connection string with placeholders
const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(uri);

async function connectToDatabase() {
  try {
    // Connect the client to the server
    await client.connect();
    
    // Send a ping to confirm a successful connection
    await client.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment. You successfully connected to MongoDB!");
    
    return client;
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
    throw error;
  }
}

async function createIndexes() {
  const indexesCreated = [];
  
  try {
    const client = await connectToDatabase();
    const db = client.db("KrushiMitraDB");
    
    // 1. Unique index on farmers.phone
    const farmersCollection = db.collection('farmers');
    await farmersCollection.createIndex({ phone: 1 }, { unique: true });
    indexesCreated.push('farmers_phone_unique');
    
    // 2. Index on activities.farmerId and activities.date (compound)
    const activitiesCollection = db.collection('activities');
    await activitiesCollection.createIndex({ farmerId: 1, date: 1 });
    indexesCreated.push('activities_farmerId_date');
    
    // 3. Index on mandiprices for (crop, location, date) descending by date
    const mandipricesCollection = db.collection('mandiprices');
    await mandipricesCollection.createIndex({ crop: 1, location: 1, date: -1 });
    indexesCreated.push('mandiprices_crop_location_date');
    
    // 4. Index on aiinteractions.farmerId and aiinteractions.timestamp
    const aiinteractionsCollection = db.collection('aiinteractions');
    await aiinteractionsCollection.createIndex({ farmerId: 1, timestamp: 1 });
    indexesCreated.push('aiinteractions_farmerId_timestamp');
    
    // 5. Index on alerts.farmerId and alerts.status
    const alertsCollection = db.collection('alerts');
    await alertsCollection.createIndex({ farmerId: 1, status: 1 });
    indexesCreated.push('alerts_farmerId_status');
    
    return {
      status: "INDEXES_CREATED",
      indexesCreated: indexesCreated
    };
  } catch (error) {
    return {
      status: "ERROR_CONNECT",
      message: error.message
    };
  } finally {
    // Ensures that the client will close when you finish/error
    try {
      await client.close();
    } catch (closeError) {
      // Ignore close errors
    }
  }
}

const { createIndexes } = require('./db');
const { logger } = require('./logger');

async function main() {
  try {
    logger.info('Creating indexes');
    const result = await createIndexes();
    logger.info('Indexes created successfully', { result });
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    logger.error('Error creating indexes', { error: error.message });
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();

// Run the function and output the result
createIndexes().then(result => {
  console.log(JSON.stringify(result, null, 2));
}).catch(error => {
  console.error("Error:", error);
  console.log(JSON.stringify({
    status: "ERROR_CONNECT",
    message: error.message
  }, null, 2));
});
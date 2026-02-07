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

async function runQueries() {
  const client = new MongoClient(uri);
  
  try {
    // Connect to the database
    await client.connect();
    const db = client.db("KrushiMitraDB");
    
    const results = {};
    
    // 1. Find farmer by phone +919900112233
    const farmersCollection = db.collection('farmers');
    const farmer = await farmersCollection.findOne({ phone: "+919900112233" });
    results.farmers = farmer;
    
    if (!farmer) {
      return {
        status: "QUERIES_OK",
        results: {
          farmers: null,
          activities: [],
          mandiprice: null,
          schemes: []
        }
      };
    }
    
    const farmerId = farmer._id;
    
    // 2. Get last 5 activities for that farmer (sorted desc by date)
    const activitiesCollection = db.collection('activities');
    const activities = await activitiesCollection
      .find({ farmerId: farmerId })
      .sort({ date: -1 })
      .limit(5)
      .toArray();
    results.activities = activities;
    
    // 3. Get latest mandi price for Wheat in Pune
    const mandipricesCollection = db.collection('mandiprices');
    const mandiprice = await mandipricesCollection
      .findOne({ crop: "Wheat", location: "Pune" }, { sort: { date: -1 } });
    results.mandiprice = mandiprice;
    
    // 4. Get active schemes where startDate <= today <= endDate
    const schemesCollection = db.collection('schemes');
    const today = new Date();
    const schemes = await schemesCollection
      .find({
        startDate: { $lte: today },
        endDate: { $gte: today }
      })
      .toArray();
    results.schemes = schemes;
    
    return {
      status: "QUERIES_OK",
      results: results
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

// Run the function and output the result
runQueries().then(result => {
  console.log(JSON.stringify(result, null, 2));
}).catch(error => {
  console.error("Error:", error);
  console.log(JSON.stringify({
    status: "ERROR_CONNECT",
    message: error.message
  }, null, 2));
});
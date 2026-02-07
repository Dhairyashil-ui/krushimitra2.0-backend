const { MongoClient } = require('mongodb');
require('dotenv').config({ path: './SIH_APP-main/backend/.env' });

async function checkCollections() {
  console.log("Checking database collections...");
  
  const uri = `mongodb+srv://${process.env.DB_READER_USER}:${process.env.DB_READER_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  const client = new MongoClient(uri);
  
  try {
    console.log("Connecting to MongoDB...");
    await client.connect();
    console.log("✅ Connected to MongoDB");
    
    const db = client.db("KrushiMitraDB");
    
    // List all collections
    const collections = await db.listCollections().toArray();
    console.log("\nCollections in KrushiMitraDB:");
    if (collections.length === 0) {
      console.log("No collections found");
    } else {
      collections.forEach((collection, index) => {
        console.log(`${index + 1}. ${collection.name}`);
      });
    }
    
    // Close connection
    await client.close();
    console.log("\nConnection closed");
  } catch (error) {
    console.log("❌ Error:", error.message);
    
    // Close connection if it was opened
    try {
      await client.close();
    } catch (closeError) {
      // Ignore close errors
    }
  }
}

checkCollections();
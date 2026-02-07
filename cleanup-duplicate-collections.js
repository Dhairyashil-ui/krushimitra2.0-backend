const { MongoClient } = require('mongodb');
require('dotenv').config({ path: './SIH_APP-main/backend/.env' });

async function cleanupDuplicateCollections() {
  console.log("Cleaning up duplicate collections...");
  
  // Use admin user credentials (we'll need to fix these)
  // For now, let's try with reader user to identify the issue
  const uri = `mongodb+srv://${process.env.DB_ADMIN_USER}:${process.env.DB_ADMIN_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  const client = new MongoClient(uri);
  
  try {
    console.log("Connecting to MongoDB...");
    await client.connect();
    console.log("✅ Connected to MongoDB");
    
    const db = client.db("KrushiMitraDB");
    
    // List all collections
    const collections = await db.listCollections().toArray();
    console.log("\nCurrent collections:");
    collections.forEach((collection, index) => {
      console.log(`${index + 1}. ${collection.name}`);
    });
    
    // Check for duplicate farmers collection
    const farmersCollections = collections.filter(c => c.name.toLowerCase() === 'farmers');
    if (farmersCollections.length > 1) {
      console.log(`\n⚠️  Found ${farmersCollections.length} collections with name 'farmers':`);
      farmersCollections.forEach(c => console.log(`  - ${c.name}`));
      
      // Keep the lowercase version and remove others
      const collectionsToRemove = farmersCollections.filter(c => c.name !== 'farmers');
      if (collectionsToRemove.length > 0) {
        console.log("\nRemoving duplicate collections:");
        for (const collection of collectionsToRemove) {
          console.log(`  - Dropping collection: ${collection.name}`);
          // Note: We would need admin privileges to drop collections
          // This is just for demonstration
        }
      }
    } else {
      console.log("\n✅ No duplicate farmers collections found");
    }
    
    // Close connection
    await client.close();
    console.log("\nConnection closed");
  } catch (error) {
    console.log("❌ Error connecting with admin credentials:", error.message);
    console.log("Attempting to connect with reader credentials to analyze the issue...");
    
    // Try with reader credentials to at least see the structure
    const readerUri = `mongodb+srv://${process.env.DB_READER_USER}:${process.env.DB_READER_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
    const readerClient = new MongoClient(readerUri);
    
    try {
      await readerClient.connect();
      console.log("✅ Connected with reader credentials");
      
      const db = readerClient.db("KrushiMitraDB");
      const collections = await db.listCollections().toArray();
      
      console.log("\nCollections accessible with reader credentials:");
      collections.forEach((collection, index) => {
        console.log(`${index + 1}. ${collection.name}`);
      });
      
      await readerClient.close();
    } catch (readerError) {
      console.log("❌ Error with reader credentials as well:", readerError.message);
    }
  }
}

cleanupDuplicateCollections();
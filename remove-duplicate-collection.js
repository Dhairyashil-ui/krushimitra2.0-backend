const { MongoClient } = require('mongodb');
const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '.env') });

async function removeDuplicateCollection() {
  console.log("=== Removing Duplicate Collection ===\n");
  
  // We need to use admin user to remove collections
  // But first let's verify we have the correct credentials
  if (!process.env.DB_ADMIN_USER || !process.env.DB_ADMIN_PASS || !process.env.CLUSTER_HOST) {
    console.log("❌ Missing admin credentials in .env file");
    return;
  }
  
  const uri = `mongodb+srv://${process.env.DB_ADMIN_USER}:${process.env.DB_ADMIN_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  const client = new MongoClient(uri, {
    serverSelectionTimeoutMS: 5000,
    connectTimeoutMS: 5000
  });
  
  try {
    console.log("Connecting with admin credentials...");
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("✅ Admin user connected successfully\n");
    
    const db = client.db("KrushiMitraDB");
    
    // List all collections to confirm duplicates
    const collections = await db.listCollections().toArray();
    const farmersCollections = collections.filter(c => c.name.toLowerCase() === 'farmers');
    
    console.log(`Found ${farmersCollections.length} collections with 'farmers' name:`);
    farmersCollections.forEach(c => console.log(`   - ${c.name}`));
    
    if (farmersCollections.length > 1) {
      // Determine which one to keep (lowercase version)
      const collectionToKeep = farmersCollections.find(c => c.name === 'farmers');
      const collectionToRemove = farmersCollections.find(c => c.name !== 'farmers');
      
      console.log(`\nKeeping: ${collectionToKeep.name}`);
      console.log(`Removing: ${collectionToRemove.name}`);
      
      // Drop the duplicate collection
      console.log(`\nRemoving collection '${collectionToRemove.name}'...`);
      await db.collection(collectionToRemove.name).drop();
      console.log("✅ Duplicate collection removed successfully");
    } else {
      console.log("✅ No duplicate collections found");
    }
    
    await client.close();
  } catch (error) {
    console.log(`❌ Error: ${error.message}`);
    if (error.message.includes("bad auth") || error.message.includes("Authentication failed")) {
      console.log("\n🔧 Authentication failed. Please check:");
      console.log("   1. Admin username and password in .env file");
      console.log("   2. That the admin user exists in MongoDB Atlas");
      console.log("   3. That the admin user has proper privileges");
      console.log("   4. That your IP is whitelisted in MongoDB Atlas");
    }
    try {
      await client.close();
    } catch (closeError) {
      // Ignore
    }
  }
}

removeDuplicateCollection();
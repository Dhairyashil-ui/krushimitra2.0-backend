const { MongoClient } = require('mongodb');
require('dotenv').config({ path: './SIH_APP-main/backend/.env' });

async function testAdminConnection() {
  console.log("Testing Admin User Connection...");
  
  if (!process.env.DB_ADMIN_USER || !process.env.DB_ADMIN_PASS || !process.env.CLUSTER_HOST) {
    console.log("❌ Missing required environment variables");
    return;
  }
  
  const uri = `mongodb+srv://${process.env.DB_ADMIN_USER}:${process.env.DB_ADMIN_PASS}@${process.env.CLUSTER_HOST}/?retryWrites=true&w=majority`;
  console.log("Connection URI:", uri.replace(process.env.DB_ADMIN_PASS, '****'));
  
  const client = new MongoClient(uri, {
    serverSelectionTimeoutMS: 5000,
    connectTimeoutMS: 5000
  });
  
  try {
    console.log("Connecting to MongoDB...");
    await client.connect();
    console.log("✅ Successfully connected to MongoDB");
    
    // Test database access
    console.log("Testing database access...");
    const db = client.db("KrushiMitraDB");
    await db.command({ ping: 1 });
    console.log("✅ Successfully accessed KrushiMitraDB");
    
    // Close connection
    await client.close();
    console.log("Connection closed");
  } catch (error) {
    console.log("❌ Connection failed");
    console.log("Error name:", error.name);
    console.log("Error message:", error.message);
    console.log("Error code:", error.code);
    
    // Close connection if it was opened
    try {
      await client.close();
    } catch (closeError) {
      // Ignore close errors
    }
  }
}

async function testReaderConnection() {
  console.log("\nTesting Reader User Connection...");
  
  if (!process.env.DB_READER_USER || !process.env.DB_READER_PASS || !process.env.CLUSTER_HOST) {
    console.log("❌ Missing required environment variables");
    return;
  }
  
  const uri = `mongodb+srv://${process.env.DB_READER_USER}:${process.env.DB_READER_PASS}@${process.env.CLUSTER_HOST}/?retryWrites=true&w=majority`;
  console.log("Connection URI:", uri.replace(process.env.DB_READER_PASS, '****'));
  
  const client = new MongoClient(uri, {
    serverSelectionTimeoutMS: 5000,
    connectTimeoutMS: 5000
  });
  
  try {
    console.log("Connecting to MongoDB...");
    await client.connect();
    console.log("✅ Successfully connected to MongoDB");
    
    // Test database access
    console.log("Testing database access...");
    const db = client.db("KrushiMitraDB");
    await db.command({ ping: 1 });
    console.log("✅ Successfully accessed KrushiMitraDB");
    
    // Close connection
    await client.close();
    console.log("Connection closed");
  } catch (error) {
    console.log("❌ Connection failed");
    console.log("Error name:", error.name);
    console.log("Error message:", error.message);
    console.log("Error code:", error.code);
    
    // Close connection if it was opened
    try {
      await client.close();
    } catch (closeError) {
      // Ignore close errors
    }
  }
}

async function runAllTests() {
  console.log("=== MongoDB Connection Tests ===\n");
  
  await testAdminConnection();
  await testReaderConnection();
  
  console.log("\n=== Test Summary ===");
  console.log("If you see authentication errors, please verify:");
  console.log("1. Your MongoDB Atlas cluster is active");
  console.log("2. Your database users exist with correct passwords");
  console.log("3. Your IP address is whitelisted in MongoDB Atlas Network Access");
  console.log("4. The cluster host is correct");
}

runAllTests();
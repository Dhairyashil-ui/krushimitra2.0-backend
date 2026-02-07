const { MongoClient } = require('mongodb');
require('dotenv').config();

async function testConnection() {
  console.log("Testing database connection...");
  
  // Check if environment variables are set
  if (!process.env.DB_USER || !process.env.DB_PASS || !process.env.CLUSTER_HOST) {
    console.log("ERROR: Missing environment variables");
    console.log("DB_USER:", process.env.DB_USER);
    console.log("DB_PASS:", process.env.DB_PASS ? "[SET]" : "[NOT SET]");
    console.log("CLUSTER_HOST:", process.env.CLUSTER_HOST);
    return;
  }
  
  // MongoDB connection string
  const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  console.log("Connection string:", uri.replace(process.env.DB_PASS, '****').replace(process.env.DB_USER, '****'));
  
  const client = new MongoClient(uri);
  
  try {
    // Connect to the database
    console.log("Connecting to database...");
    await client.connect();
    
    // Send a ping to confirm a successful connection
    await client.db("admin").command({ ping: 1 });
    console.log("SUCCESS: Connected to MongoDB database");
    
    // List databases
    const databases = await client.db().admin().listDatabases();
    console.log("Available databases:", databases.databases.map(db => db.name));
    
    // Close the connection
    await client.close();
    console.log("Connection closed");
  } catch (error) {
    console.log("ERROR: Failed to connect to database");
    console.log("Error message:", error.message);
    console.log("Error code:", error.code);
  }
}

testConnection();
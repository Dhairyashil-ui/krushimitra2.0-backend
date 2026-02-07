const { MongoClient } = require('mongodb');
require('dotenv').config({ path: './SIH_APP-main/backend/.env' });

async function testConnection() {
  console.log("Testing database connection...");
  
  // Check if environment variables are set
  const requiredEnvVars = [
    'DB_READER_USER', 'DB_READER_PASS',
    'DB_WRITER_USER', 'DB_WRITER_PASS',
    'DB_ADMIN_USER', 'DB_ADMIN_PASS',
    'CLUSTER_HOST'
  ];
  
  const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);
  if (missingEnvVars.length > 0) {
    console.log("ERROR: Missing environment variables:", missingEnvVars);
    return;
  }
  
  // Use admin user for testing
  const uri = `mongodb+srv://${process.env.DB_ADMIN_USER}:${process.env.DB_ADMIN_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  console.log("Connection string:", uri.replace(process.env.DB_ADMIN_PASS, '****').replace(process.env.DB_ADMIN_USER, '****'));
  
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
    
    // Check if KrushiMitraDB exists
    const dbExists = databases.databases.some(db => db.name === "KrushiMitraDB");
    if (dbExists) {
      console.log("SUCCESS: KrushiMitraDB database found");
      
      // List collections in KrushiMitraDB
      const db = client.db("KrushiMitraDB");
      const collections = await db.listCollections().toArray();
      console.log("Collections in KrushiMitraDB:", collections.map(c => c.name));
    } else {
      console.log("WARNING: KrushiMitraDB database not found");
    }
    
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
const { MongoClient } = require('mongodb');

// Check if environment variables are provided
const dbUser = process.env.DB_USER || 'your_username';
const dbPass = process.env.DB_PASS || 'your_password';
const clusterHost = process.env.CLUSTER_HOST || 'your_cluster_host';

console.log('Attempting to connect with:');
console.log('- DB_USER:', dbUser !== 'your_username' ? 'Set' : 'Not set (using placeholder)');
console.log('- DB_PASS:', dbPass !== 'your_password' ? 'Set' : 'Not set (using placeholder)');
console.log('- CLUSTER_HOST:', clusterHost !== 'your_cluster_host' ? 'Set' : 'Not set (using placeholder)');

// MongoDB connection string with placeholders
const uri = `mongodb+srv://${dbUser}:${dbPass}@${clusterHost}/KrushiMitraDB?retryWrites=true&w=majority`;

console.log('\nConnection string (with placeholders if env vars not set):');
console.log(uri);

// If we're using placeholder values, return an error
if (dbUser === 'your_username' || dbPass === 'your_password' || clusterHost === 'your_cluster_host') {
  console.log(JSON.stringify({
    status: "ERROR_CONNECT",
    message: "Missing environment variables: DB_USER, DB_PASS, and/or CLUSTER_HOST must be set"
  }, null, 2));
  process.exit(1);
}

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(uri);

async function testConnection() {
  try {
    console.log('\nAttempting to connect to MongoDB...');
    // Connect the client to the server
    await client.connect();
    
    // Send a ping to confirm a successful connection
    await client.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment. You successfully connected to MongoDB!");
    
    // Get server info
    const adminDb = client.db("admin");
    const buildInfo = await adminDb.command({ buildInfo: 1 });
    
    // Get collections
    const db = client.db("KrushiMitraDB");
    const collections = await db.listCollections().toArray();
    const collectionNames = collections.map(collection => collection.name);
    
    console.log(JSON.stringify({
      status: "CONNECTED",
      serverInfo: {
        version: buildInfo.version,
        host: buildInfo.version
      },
      collections: collectionNames
    }, null, 2));
    
  } catch (error) {
    console.error("Connection failed:", error.message);
    console.log(JSON.stringify({
      status: "ERROR_CONNECT",
      message: error.message
    }, null, 2));
  } finally {
    // Ensures that the client will close when you finish/error
    await client.close();
  }
}

testConnection();
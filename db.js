const { MongoClient } = require('mongodb');
const { logDBOperation, logDBError } = require('./logger');

require('dotenv').config();

// Support both MONGODB_URI (cloud deployment) and individual credentials (local development)
let readerUri, writerUri, adminUri;

if (process.env.MONGODB_URI) {
  // Cloud deployment: use single connection string for all operations
  console.log('Using MONGODB_URI for database connection');
  readerUri = process.env.MONGODB_URI;
  writerUri = process.env.MONGODB_URI;
  adminUri = process.env.MONGODB_URI;
} else {
  // Local development: use individual credentials
  const requiredEnvVars = [
    'DB_READER_USER', 'DB_READER_PASS',
    'DB_WRITER_USER', 'DB_WRITER_PASS',
    'DB_ADMIN_USER', 'DB_ADMIN_PASS',
    'CLUSTER_HOST'
  ];

  const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);
  if (missingEnvVars.length > 0) {
    console.log(JSON.stringify({
      status: "ERROR_CONNECT",
      message: `Missing environment variables: Either set MONGODB_URI or set ${missingEnvVars.join(', ')} in .env file`
    }, null, 2));
    process.exit(1);
  }

  readerUri = `mongodb+srv://${process.env.DB_READER_USER}:${process.env.DB_READER_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  writerUri = `mongodb+srv://${process.env.DB_WRITER_USER}:${process.env.DB_WRITER_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  adminUri = `mongodb+srv://${process.env.DB_ADMIN_USER}:${process.env.DB_ADMIN_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
}

// Create MongoClients for different user types
const readerClient = new MongoClient(readerUri);
const writerClient = new MongoClient(writerUri);
const adminClient = new MongoClient(adminUri);

/**
 * Connect to database with appropriate user based on operation type
 * @param {string} operationType - 'read', 'write', or 'admin'
 * @param {object} context - Context information for logging (farmerId, etc.)
 * @returns {MongoClient} Connected MongoClient instance
 */
async function connectToDatabase(operationType = 'read', context = {}) {
  const startTime = Date.now();
  try {
    let client;
    let userDescription;
    
    switch (operationType.toLowerCase()) {
      case 'write':
        client = writerClient;
        userDescription = 'writer';
        break;
      case 'admin':
        client = adminClient;
        userDescription = 'admin';
        break;
      case 'read':
      default:
        client = readerClient;
        userDescription = 'reader';
        break;
    }
    
    // Connect the client to the server
    await client.connect();
    
    // Send a ping to confirm a successful connection
    await client.db("admin").command({ ping: 1 });
    
    // Log successful connection
    const duration = Date.now() - startTime;
    logDBOperation('connect', {
      user: userDescription,
      operationType,
      durationMs: duration,
      status: 'success',
      ...context
    });
    
    console.log(`Pinged your deployment. You successfully connected to MongoDB as ${userDescription} user!`);
    
    return client;
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('connect', error, {
      operationType,
      durationMs: duration,
      ...context
    });
    
    console.error(`Error connecting to MongoDB as ${operationType} user:`, error);
    throw error;
  }
}

async function getDatabaseInfo(context = {}) {
  const startTime = Date.now();
  try {
    const client = await connectToDatabase('admin', context);
    const db = client.db("KrushiMitraDB");
    
    // Get server info
    const adminDb = client.db("admin");
    const serverStatus = await adminDb.command({ serverStatus: 1 });
    const buildInfo = await adminDb.command({ buildInfo: 1 });
    
    // Get collections
    const collections = await db.listCollections().toArray();
    const collectionNames = collections.map(collection => collection.name);
    
    // Log operation
    const duration = Date.now() - startTime;
    logDBOperation('getDatabaseInfo', {
      durationMs: duration,
      status: 'success',
      collectionsCount: collectionNames.length,
      ...context
    });
    
    return {
      status: "CONNECTED",
      serverInfo: {
        version: buildInfo.version,
        host: serverStatus.host
      },
      collections: collectionNames
    };
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('getDatabaseInfo', error, {
      durationMs: duration,
      ...context
    });
    
    return {
      status: "ERROR_CONNECT",
      message: error.message
    };
  } finally {
    // Ensures that the client will close when you finish/error
    try {
      await adminClient.close();
    } catch (closeError) {
      // Ignore close errors
    }
  }
}

async function ensureCollections(context = {}) {
  const requiredCollections = [
    'farmers', 
    'activities', 
    'mandiprices', 
    'schemes', 
    'aiinteractions', 
    'crop_health', 
    'alerts'
  ];
  
  const startTime = Date.now();
  try {
    const client = await connectToDatabase('admin', context);
    const db = client.db("KrushiMitraDB");
    
    // Get existing collections
    const collections = await db.listCollections().toArray();
    const existingCollectionNames = collections.map(collection => collection.name);
    
    // Find missing collections
    const missingCollections = requiredCollections.filter(
      collection => !existingCollectionNames.includes(collection)
    );
    
    // Create missing collections
    const createdCollections = [];
    for (const collectionName of missingCollections) {
      await db.createCollection(collectionName);
      createdCollections.push(collectionName);
    }
    
    // Log operation
    const duration = Date.now() - startTime;
    logDBOperation('ensureCollections', {
      durationMs: duration,
      status: 'success',
      createdCollections,
      existingCollections: existingCollectionNames,
      collectionsCount: existingCollectionNames.concat(createdCollections).length,
      ...context
    });
    
    // Return status
    return {
      status: "COLLECTIONS_OK",
      collectionsCreated: createdCollections,
      collectionsPresent: existingCollectionNames.concat(createdCollections)
    };
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('ensureCollections', error, {
      durationMs: duration,
      ...context
    });
    
    return {
      status: "ERROR_CONNECT",
      message: error.message
    };
  } finally {
    // Ensures that the client will close when you finish/error
    try {
      await adminClient.close();
    } catch (closeError) {
      // Ignore close errors
    }
  }
}

async function createIndexes(context = {}) {
  const startTime = Date.now();
  try {
    const client = await connectToDatabase('admin', context);
    const db = client.db("KrushiMitraDB");
    
    const indexesCreated = [];
    
    // 1. Unique index on farmers.phone
    const farmersCollection = db.collection('farmers');
    await farmersCollection.createIndex({ phone: 1 }, { unique: true });
    indexesCreated.push('farmers.phone_unique');
    
    // 2. Index on activities.farmerId and activities.date (compound)
    const activitiesCollection = db.collection('activities');
    await activitiesCollection.createIndex({ farmerId: 1, date: 1 });
    indexesCreated.push('activities.farmerId_date');
    
    // 3. Index on mandiprices for (crop, location, date) descending by date
    const mandipricesCollection = db.collection('mandiprices');
    await mandipricesCollection.createIndex({ crop: 1, location: 1, date: -1 });
    indexesCreated.push('mandiprices.crop_location_date_desc');
    
    // 4. Index on aiinteractions.farmerId and aiinteractions.timestamp
    const aiinteractionsCollection = db.collection('aiinteractions');
    await aiinteractionsCollection.createIndex({ farmerId: 1, timestamp: 1 });
    indexesCreated.push('aiinteractions.farmerId_timestamp');
    
    // 5. Index on alerts.farmerId and alerts.status
    const alertsCollection = db.collection('alerts');
    await alertsCollection.createIndex({ farmerId: 1, status: 1 });
    indexesCreated.push('alerts.farmerId_status');
    
    // Log operation
    const duration = Date.now() - startTime;
    logDBOperation('createIndexes', {
      durationMs: duration,
      status: 'success',
      indexesCreated,
      indexesCount: indexesCreated.length,
      ...context
    });
    
    return {
      status: "INDEXES_CREATED",
      indexesCreated: indexesCreated
    };
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('createIndexes', error, {
      durationMs: duration,
      ...context
    });
    
    return {
      status: "ERROR_CONNECT",
      message: error.message
    };
  } finally {
    // Ensures that the client will close when you finish/error
    try {
      await adminClient.close();
    } catch (closeError) {
      // Ignore close errors
    }
  }
}

module.exports = { connectToDatabase, getDatabaseInfo, ensureCollections, createIndexes };
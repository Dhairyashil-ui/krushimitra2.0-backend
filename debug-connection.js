const { MongoClient } = require('mongodb');
const dotenv = require('dotenv');
dotenv.config();

console.log('=== Environment Variables Check ===');
console.log('DB_USER:', process.env.DB_USER || 'NOT SET');
console.log('DB_PASS:', process.env.DB_PASS ? '*** (SET)' : 'NOT SET');
console.log('CLUSTER_HOST:', process.env.CLUSTER_HOST || 'NOT SET');

if (!process.env.DB_USER || !process.env.DB_PASS || !process.env.CLUSTER_HOST) {
    console.log('ERROR: Missing required environment variables');
    process.exit(1);
}

console.log('\n=== Connection URI ===');
const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority&serverSelectionTimeoutMS=5000&connectTimeoutMS=5000`;
console.log('URI:', uri.replace(process.env.DB_PASS, '***'));

console.log('\n=== Attempting Connection ===');
const client = new MongoClient(uri);

async function testConnection() {
    try {
        console.log('Connecting to MongoDB...');
        await client.connect();
        console.log('SUCCESS: Connected to MongoDB');
        
        console.log('Sending ping...');
        await client.db("admin").command({ ping: 1 });
        console.log('SUCCESS: Ping successful');
        
        console.log('Listing databases...');
        const databases = await client.db().admin().listDatabases();
        console.log('Available databases:', databases.databases.map(db => db.name));
        
        await client.close();
        console.log('Connection closed');
    } catch (error) {
        console.log('ERROR: Connection failed');
        console.log('Error name:', error.name);
        console.log('Error message:', error.message);
        console.log('Error code:', error.code);
        if (error.errInfo) {
            console.log('Error info:', JSON.stringify(error.errInfo, null, 2));
        }
        await client.close();
    }
}

testConnection();
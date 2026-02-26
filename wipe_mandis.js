const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });
const { MongoClient } = require('mongodb');

async function wipeDatabase() {
    try {
        console.log('Connecting to MongoDB using constructed URI...');
        const uri = `mongodb+srv://${process.env.DB_ADMIN_USER}:${process.env.DB_ADMIN_PASS}@${process.env.CLUSTER_HOST}/`;
        const client = new MongoClient(uri);
        await client.connect();

        console.log('Connected! Deleting all records in mandi_prices...');
        const db = client.db('KrushiMitraDB');
        const collection = db.collection('mandi_prices');

        const result = await collection.deleteMany({});
        console.log(`Successfully deleted ${result.deletedCount} old records!`);

        await client.close();
        console.log('Done.');
    } catch (e) {
        console.error('Error:', e.message);
    }
}

wipeDatabase();

const { MongoClient } = require('mongodb');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const host = process.env.CLUSTER_HOST || '';
const user = process.env.DB_ADMIN_USER || '';
const pass = process.env.DB_ADMIN_PASS || '';
const uri = `mongodb+srv://${user}:${pass}@${host}/?retryWrites=true&w=majority`;

async function testQuery() {
    const client = new MongoClient(uri);
    try {
        await client.connect();
        const db = client.db('KrushiMitraDB');
        const location = "Main Market Yard (Gultekdi)";

        console.log(`Testing Query for Location: ${location}`);
        console.log(`Using regex: /${location}/i`);

        const prices = await db.collection('mandi_prices')
            .find({ mandi: { $regex: new RegExp(location, 'i') } })
            .sort({ date: -1 })
            .toArray();

        console.log(`Found ${prices.length} records!`);
        if (prices.length > 0) {
            console.log("Sample Result:");
            console.log(prices[0]);
        }

    } catch (error) {
        console.error("Test failed:", error);
    } finally {
        await client.close();
    }
}

testQuery();

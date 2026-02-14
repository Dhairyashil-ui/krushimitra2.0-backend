const { MongoClient } = require('mongodb');
const { getActiveSchemes, getCropHealthHistory, getLatestMandiPrices } = require('./ai-query-helper');
require('dotenv').config({ path: '../.env' });

async function testRag() {
    const uri = process.env.MONGODB_URI;
    if (!uri) {
        // creating fallback if .env not loaded correctly or empty
        console.log("⚠️ No MONGODB_URI in .env, trying default local");
    }

    // If env loading fails, use a hardcoded fallback purely for this test if reasonable, or just fail.
    // Actually, let's try to load .env from current dir too
    require('dotenv').config();

    const finalUri = process.env.MONGODB_URI; // || "mongodb://localhost:27017";
    if (!finalUri) {
        console.error("❌ MONGODB_URI missing.");
        return;
    }

    const client = new MongoClient(finalUri);

    try {
        await client.connect();
        const db = client.db("KrushiMitraDB");

        // 1. Setup Test Data
        const schemes = db.collection('schemes');
        const health = db.collection('crop_health');

        // Clean old test data
        await schemes.deleteMany({ title: "Test Scheme" });
        await health.deleteMany({ plant: "Test Plant" });

        // Insert dummy scheme
        await schemes.insertOne({
            title: "Test Scheme",
            description: "Free fertilizer for testing",
            startDate: new Date("2020-01-01"),
            endDate: new Date("2030-01-01"),
            location: "all"
        });

        // Insert dummy health record
        await health.insertOne({
            farmerId: "test-farmer-id",
            plant: "Test Plant",
            disease: "Test Rust",
            timestamp: new Date()
        });

        console.log("✅ Test Data Inserted");

        // 2. Test Retrieval
        console.log("🔍 Testing Retrieval...");

        const retrievedSchemes = await getActiveSchemes(schemes, "Pune");
        console.log(`Schemes Found: ${retrievedSchemes.length}`);
        if (retrievedSchemes.some(s => s.title === "Test Scheme")) {
            console.log("✅ Scheme Retrieval Passed");
        } else {
            console.error("❌ Scheme Retrieval Failed");
        }

        const retrievedHealth = await getCropHealthHistory(health, "test-farmer-id");
        console.log(`Health Records Found: ${retrievedHealth.length}`);
        if (retrievedHealth.some(h => h.disease === "Test Rust")) {
            console.log("✅ Disease Retrieval Passed");
        } else {
            console.error("❌ Disease Retrieval Failed");
        }

        console.log("🎉 Verification Complete");

    } catch (e) {
        console.error(e);
    } finally {
        await client.close();
    }
}

testRag();

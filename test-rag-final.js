
const { MongoClient } = require('mongodb');
const { getActiveSchemes, getCropHealthHistory, getLatestMandiPrices } = require('./ai-query-helper');
const path = require('path');
const fs = require('fs');
require('dotenv').config({ path: '../.env' }); // try parent

async function testRag() {
    let uri = process.env.MONGODB_URI;
    if (!uri) {
        console.log("⚠️ No MONGODB_URI in ../.env, checking current dir .env...");
        require('dotenv').config(); // try current
        uri = process.env.MONGODB_URI;
    }

    if (!uri) {
        console.log("⚠️ Still no MONGODB_URI. Assuming local default for TEST: mongodb://127.0.0.1:27017");
        uri = "mongodb://127.0.0.1:27017";
    }

    console.log(`Using Mongo URI: ${uri.replace(/\/\/.*@/, '//***@')}`); // Mask creds

    const client = new MongoClient(uri);

    try {
        await client.connect();
        console.log("✅ Connected to MongoDB");
        const db = client.db("KrushiMitraDB");

        // 1. Setup Test Data
        const schemes = db.collection('schemes');
        const health = db.collection('crop_health');
        const mandiprices = db.collection('mandiprices');

        // Clean old test data
        await schemes.deleteMany({ title: "Test Scheme" });
        await health.deleteMany({ plant: "Test Plant" });
        await mandiprices.deleteMany({ crop: "Test Crop" });

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

        // Insert dummy mandi price
        await mandiprices.insertOne({
            crop: "Test Crop",
            location: "Pune",
            price: 100,
            date: new Date()
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

        const retrievedPrices = await getLatestMandiPrices(mandiprices, "Test Crop", "Pune");
        console.log(`Prices Found: ${retrievedPrices.length}`);
        if (retrievedPrices.some(p => p.price === 100)) {
            console.log("✅ Mandi Price Retrieval Passed");
        } else {
            console.error("❌ Mandi Price Retrieval Failed");
        }

        console.log("🎉 Verification Complete");

    } catch (e) {
        console.error("❌ Test Failed:", e.message);
    } finally {
        await client.close();
    }
}

testRag();

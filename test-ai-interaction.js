const { connectToDatabase } = require('./db');

async function testAIInteraction() {
  try {
    console.log('Testing AI interaction saving to MongoDB...');
    
    // Connect to database
    const client = await connectToDatabase('write');
    const db = client.db("KrushiMitraDB");
    const aiinteractionsCollection = db.collection('aiinteractions');
    
    // Create a sample AI interaction
    const sampleInteraction = {
      farmerId: "test_farmer_123",
      query: "What crops should I plant this season?",
      response: "Based on your location and current weather patterns, I recommend planting wheat or barley.",
      context: {},
      timestamp: new Date()
    };
    
    // Save to database
    const result = await aiinteractionsCollection.insertOne(sampleInteraction);
    console.log('AI interaction saved successfully:', result.insertedId);
    
    // Retrieve the saved interaction
    const savedInteraction = await aiinteractionsCollection.findOne({ _id: result.insertedId });
    console.log('Retrieved AI interaction:', savedInteraction);
    
    // Close connection
    await client.close();
    
    console.log('Test completed successfully!');
  } catch (error) {
    console.error('Test failed:', error);
  }
}

testAIInteraction();
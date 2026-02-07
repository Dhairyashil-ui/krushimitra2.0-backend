// Test the AI interactions API endpoint

async function testAPIEndpoint() {
  try {
    console.log('Testing AI interactions API endpoint...');
    
    const testData = {
      farmerId: "test_farmer_api_456",
      query: "How often should I water my crops?",
      response: "For most crops, watering every 2-3 days is sufficient. Adjust based on soil type and weather conditions.",
      context: {
        cropType: "wheat",
        soilType: "black soil"
      }
    };
    
    const response = await fetch('http://localhost:3000/ai/interactions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer mock-token-12345'
      },
      body: JSON.stringify(testData)
    });
    
    const result = await response.json();
    console.log('API response:', result);
    
    if (response.ok) {
      console.log('Test completed successfully!');
    } else {
      console.error('Test failed:', result);
    }
  } catch (error) {
    console.error('Test failed with error:', error);
  }
}

testAPIEndpoint();
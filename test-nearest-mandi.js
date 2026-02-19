const fetch = require('node-fetch');

async function testNearestMandi() {
    const baseUrl = 'http://localhost:3001'; // Adjust if port is different
    const puneLat = 18.5204;
    const puneLon = 73.8567; // Pune City Center

    console.log(`Testing Nearest Mandi calculation for user at ${puneLat}, ${puneLon}...`);

    try {
        const response = await fetch(`${baseUrl}/mandis/nearest?lat=${puneLat}&lon=${puneLon}`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        console.log('Response Status:', result.status);
        console.log(`Found ${result.data.length} nearest mandis:`);

        result.data.forEach((m, i) => {
            console.log(`${i + 1}. ${m.name} (${m.distanceKm} km)`);
        });

        // Verification
        if (result.data.length === 5 && result.data[0].distanceKm < 10) {
            console.log('✅ Test Passed: Returned 5 nearby mandis.');
        } else {
            console.error('❌ Test Failed: Unexpected result format or distance.');
        }

    } catch (error) {
        console.error('Test Failed:', error.message);
        if (error.code === 'ECONNREFUSED') {
            console.log('Make sure the server is running!');
        }
    }
}

testNearestMandi();

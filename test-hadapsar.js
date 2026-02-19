const fetch = require('node-fetch');

// Hadapsar Coordinates from user request
const USER_LAT = 18.5089;
const USER_LON = 73.9260;

async function testHadapsarExample() {
    const baseUrl = 'http://localhost:3001';

    console.log(`Testing with Hadapsar Coordinates: ${USER_LAT}, ${USER_LON}`);

    try {
        const response = await fetch(`${baseUrl}/mandis/nearest?lat=${USER_LAT}&lon=${USER_LON}`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        console.log('--- Nearest Mandis ---');
        result.data.forEach((m, i) => {
            console.log(`${i + 1}. ${m.name} – ${m.distanceKm} km`);
        });

        // Verification Logic based on User's Expected Output
        // 1. Hadapsar Bhaji Mandai – 1.1 km
        // 2. Manjari APMC Market – 2.9 km
        // 3. Mundhwa Vegetable Market – 3.4 km

        const first = result.data[0];
        if (first.name === "Hadapsar Bhaji Mandai" && Math.abs(first.distanceKm - 1.1) < 0.2) {
            console.log('\n✅ MATCH: Rank 1 is accurate');
        } else {
            console.log('\n❌ MISMATCH: Rank 1 expected Hadapsar (~1.1km)');
        }

    } catch (error) {
        console.error('Test Failed:', error.message);
        if (error.code === 'ECONNREFUSED') {
            console.log('Make sure the server is running!');
        }
    }
}

testHadapsarExample();

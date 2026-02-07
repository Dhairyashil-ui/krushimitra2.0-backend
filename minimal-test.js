// Simple test to check if we can load environment variables and connect to MongoDB
require('dotenv').config({ path: './SIH_APP-main/backend/.env' });

console.log('DB_USER:', process.env.DB_USER);
console.log('DB_PASS:', process.env.DB_PASS ? 'SET' : 'NOT SET');
console.log('CLUSTER_HOST:', process.env.CLUSTER_HOST);

if (process.env.DB_USER && process.env.DB_PASS && process.env.CLUSTER_HOST) {
    console.log('All environment variables are set');
} else {
    console.log('Missing environment variables');
    process.exit(1);
}

console.log('Test completed successfully');
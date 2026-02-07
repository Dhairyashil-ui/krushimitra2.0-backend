// Simple script to test MongoDB connection and return JSON as specified

// Since we don't have actual MongoDB credentials, we'll return an error
console.log(JSON.stringify({
  status: "ERROR_CONNECT",
  message: "Missing environment variables: DB_USER, DB_PASS, and/or CLUSTER_HOST must be set. Please update the .env file with your actual MongoDB credentials."
}, null, 2));
const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '.env') });

console.log("=== Environment Variables Check ===\n");

// Check if required environment variables are present
const requiredVars = [
  'DB_READER_USER',
  'DB_READER_PASS',
  'DB_WRITER_USER',
  'DB_WRITER_PASS',
  'DB_ADMIN_USER',
  'DB_ADMIN_PASS',
  'CLUSTER_HOST'
];

let allPresent = true;

requiredVars.forEach(varName => {
  if (process.env[varName]) {
    console.log(`✅ ${varName}: ***PRESENT***`);
  } else {
    console.log(`❌ ${varName}: MISSING`);
    allPresent = false;
  }
});

console.log(`\n=== Connection Strings ===\n`);

if (allPresent) {
  console.log("Reader Connection String:");
  console.log(`mongodb+srv://${process.env.DB_READER_USER}:*****@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority\n`);
  
  console.log("Writer Connection String:");
  console.log(`mongodb+srv://${process.env.DB_WRITER_USER}:*****@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority\n`);
  
  console.log("Admin Connection String:");
  console.log(`mongodb+srv://${process.env.DB_ADMIN_USER}:*****@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority\n`);
  
  console.log("=== Recommendations ===");
  console.log("1. Verify all credentials are correct in MongoDB Atlas");
  console.log("2. Ensure all users have appropriate roles assigned");
  console.log(`3. Whitelist your IP (42.104.218.61) in MongoDB Atlas Network Access`);
  console.log("4. Test connections after making changes");
} else {
  console.log("❌ Missing required environment variables. Please check your .env file.");
}
const { MongoClient } = require('mongodb');
require('dotenv').config({ path: './SIH_APP-main/backend/.env' });

async function testAndFixConnection() {
  console.log("=== MongoDB Connection Fix ===\n");
  
  // 1. Test all user types
  console.log("1. Testing Reader User...");
  const readerTest = await testUserConnection(
    process.env.DB_READER_USER,
    process.env.DB_READER_PASS,
    "Reader"
  );
  
  console.log("\n2. Testing Writer User...");
  const writerTest = await testUserConnection(
    process.env.DB_WRITER_USER,
    process.env.DB_WRITER_PASS,
    "Writer"
  );
  
  console.log("\n3. Testing Admin User...");
  const adminTest = await testUserConnection(
    process.env.DB_ADMIN_USER,
    process.env.DB_ADMIN_PASS,
    "Admin"
  );
  
  // 4. Summary
  console.log("\n=== Connection Test Summary ===");
  console.log(`Reader User: ${readerTest ? "✅ Success" : "❌ Failed"}`);
  console.log(`Writer User: ${writerTest ? "✅ Success" : "❌ Failed"}`);
  console.log(`Admin User: ${adminTest ? "✅ Success" : "❌ Failed"}`);
  
  if (!adminTest) {
    console.log("\n⚠️  Admin user authentication failed. This could be due to:");
    console.log("   1. Incorrect username or password");
    console.log("   2. User doesn't exist in MongoDB Atlas");
    console.log("   3. User doesn't have admin privileges");
    console.log("   4. IP address not whitelisted in MongoDB Atlas");
    
    console.log("\n🔧 To fix this issue:");
    console.log("   1. Log in to MongoDB Atlas");
    console.log("   2. Go to Database Access section");
    console.log(`   3. Verify user ${process.env.DB_ADMIN_USER} exists with correct password`);
    console.log("   4. Ensure the user has Admin role");
    console.log("   5. Check Network Access to ensure your IP is whitelisted");
  }
  
  // 5. Check for duplicate collections
  console.log("\n4. Checking for duplicate collections...");
  await checkForDuplicateCollections();
  
  console.log("\n=== Fix Recommendations ===");
  if (!adminTest) {
    console.log("1. Fix admin user credentials in MongoDB Atlas");
  }
  
  console.log("2. Remove duplicate 'Farmers' collection if not needed");
  console.log("3. Verify all IP addresses are whitelisted in MongoDB Atlas");
  console.log("4. Test connection again after making changes");
}

async function testUserConnection(username, password, userType) {
  if (!username || !password || !process.env.CLUSTER_HOST) {
    console.log(`❌ Missing credentials for ${userType} user`);
    return false;
  }
  
  const uri = `mongodb+srv://${username}:${password}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  const client = new MongoClient(uri, {
    serverSelectionTimeoutMS: 5000,
    connectTimeoutMS: 5000
  });
  
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log(`✅ ${userType} user connected successfully`);
    await client.close();
    return true;
  } catch (error) {
    console.log(`❌ ${userType} user connection failed: ${error.message}`);
    try {
      await client.close();
    } catch (closeError) {
      // Ignore
    }
    return false;
  }
}

async function checkForDuplicateCollections() {
  // Use reader credentials to check collections
  const uri = `mongodb+srv://${process.env.DB_READER_USER}:${process.env.DB_READER_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  const client = new MongoClient(uri);
  
  try {
    await client.connect();
    const db = client.db("KrushiMitraDB");
    const collections = await db.listCollections().toArray();
    
    // Check for duplicate farmers collections
    const farmersCollections = collections.filter(c => c.name.toLowerCase() === 'farmers');
    if (farmersCollections.length > 1) {
      console.log(`⚠️  Found ${farmersCollections.length} collections with 'farmers' name:`);
      farmersCollections.forEach(c => console.log(`   - ${c.name}`));
      console.log("   Recommendation: Remove duplicates using admin privileges");
    } else {
      console.log("✅ No duplicate farmers collections found");
    }
    
    await client.close();
  } catch (error) {
    console.log(`❌ Error checking collections: ${error.message}`);
  }
}

testAndFixConnection();
const { MongoClient } = require('mongodb');
const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '.env') });

async function finalVerification() {
  console.log("=== Final MongoDB Verification ===\n");
  
  // Test all user connections
  const users = [
    { name: "Reader", user: process.env.DB_READER_USER, pass: process.env.DB_READER_PASS },
    { name: "Writer", user: process.env.DB_WRITER_USER, pass: process.env.DB_WRITER_PASS },
    { name: "Admin", user: process.env.DB_ADMIN_USER, pass: process.env.DB_ADMIN_PASS }
  ];
  
  let allConnectionsSuccessful = true;
  
  for (const user of users) {
    console.log(`Testing ${user.name} User...`);
    const success = await testConnection(user.user, user.pass, user.name);
    if (!success) {
      allConnectionsSuccessful = false;
    }
    console.log("");
  }
  
  // Check for duplicate collections
  console.log("Checking for duplicate collections...");
  const noDuplicates = await checkForDuplicates();
  
  // Final summary
  console.log("=== Final Verification Summary ===");
  console.log(`All Connections: ${allConnectionsSuccessful ? "✅ Success" : "❌ Failed"}`);
  console.log(`Duplicate Collections: ${noDuplicates ? "✅ None Found" : "❌ Duplicates Exist"}`);
  
  if (allConnectionsSuccessful && noDuplicates) {
    console.log("\n🎉 All issues have been successfully resolved!");
    console.log("✅ Admin user authentication is working");
    console.log("✅ No duplicate collections found");
    console.log("\nYour MongoDB setup is now working correctly.");
  } else {
    console.log("\n❌ Some issues remain. Please check the output above.");
  }
}

async function testConnection(username, password, userType) {
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

async function checkForDuplicates() {
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
      console.log(`❌ Found ${farmersCollections.length} collections with 'farmers' name:`);
      farmersCollections.forEach(c => console.log(`   - ${c.name}`));
      await client.close();
      return false;
    } else {
      console.log("✅ No duplicate farmers collections found");
      await client.close();
      return true;
    }
  } catch (error) {
    console.log(`❌ Error checking collections: ${error.message}`);
    return false;
  }
}

finalVerification();
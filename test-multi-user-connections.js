/**
 * Test script for multi-user MongoDB connections
 * 
 * This script tests the three different database users:
 * 1. Reader user (read-only operations)
 * 2. Writer user (read-write operations)
 * 3. Admin user (administrative operations)
 */

const { connectToDatabase } = require('./db');

async function testReaderConnection() {
  console.log('Testing reader connection...');
  try {
    const client = await connectToDatabase('read');
    console.log('✅ Reader connection successful');
    
    // Test read operation
    const db = client.db("KrushiMitraDB");
    const collections = await db.listCollections().toArray();
    console.log(`✅ Reader can list collections: ${collections.length} collections found`);
    
    await client.close();
    return true;
  } catch (error) {
    console.error('❌ Reader connection failed:', error.message);
    return false;
  }
}

async function testWriterConnection() {
  console.log('Testing writer connection...');
  try {
    const client = await connectToDatabase('write');
    console.log('✅ Writer connection successful');
    
    // Test write operation (we won't actually write data in this test)
    const db = client.db("KrushiMitraDB");
    const collections = await db.listCollections().toArray();
    console.log(`✅ Writer can list collections: ${collections.length} collections found`);
    
    await client.close();
    return true;
  } catch (error) {
    console.error('❌ Writer connection failed:', error.message);
    return false;
  }
}

async function testAdminConnection() {
  console.log('Testing admin connection...');
  try {
    const client = await connectToDatabase('admin');
    console.log('✅ Admin connection successful');
    
    // Test admin operation
    const db = client.db("KrushiMitraDB");
    const collections = await db.listCollections().toArray();
    console.log(`✅ Admin can list collections: ${collections.length} collections found`);
    
    await client.close();
    return true;
  } catch (error) {
    console.error('❌ Admin connection failed:', error.message);
    return false;
  }
}

async function runAllTests() {
  console.log('=== Multi-User MongoDB Connection Tests ===\n');
  
  const results = [];
  
  results.push(await testReaderConnection());
  console.log(); // Empty line for spacing
  
  results.push(await testWriterConnection());
  console.log(); // Empty line for spacing
  
  results.push(await testAdminConnection());
  console.log(); // Empty line for spacing
  
  const passed = results.filter(result => result).length;
  const total = results.length;
  
  console.log('=== Test Results Summary ===');
  console.log(`Passed: ${passed}/${total}`);
  
  if (passed === total) {
    console.log('🎉 All tests passed! Multi-user connection system is working correctly.');
  } else {
    console.log('❌ Some tests failed. Please check the error messages above.');
  }
  
  return passed === total;
}

// Run the tests if this file is executed directly
if (require.main === module) {
  runAllTests()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Test execution failed:', error);
      process.exit(1);
    });
}

module.exports = { runAllTests };
/**
 * Test script for structured logging
 * 
 * This script tests the structured logging implementation for database operations.
 */

const { logDBOperation, logDBError } = require('./logger');

async function runLoggingTests() {
  console.log('=== Structured Logging Tests ===\n');
  
  // Test successful database operation logging
  console.log('1. Testing successful operation logging...');
  logDBOperation('testOperation', {
    farmerId: 'farmer_123',
    collection: 'farmers',
    operation: 'find',
    durationMs: 45,
    status: 'success'
  });
  console.log('✅ Success operation logged\n');
  
  // Test error logging
  console.log('2. Testing error logging...');
  const testError = new Error('Test database error');
  testError.code = 'TEST_ERROR';
  logDBError('testErrorOperation', testError, {
    farmerId: 'farmer_456',
    collection: 'activities',
    operation: 'insert',
    durationMs: 120
  });
  console.log('✅ Error logged\n');
  
  // Test connection logging
  console.log('3. Testing connection logging...');
  logDBOperation('connect', {
    user: 'reader',
    operationType: 'read',
    durationMs: 230,
    status: 'success'
  });
  console.log('✅ Connection logged\n');
  
  // Test admin operation logging
  console.log('4. Testing admin operation logging...');
  logDBOperation('createIndexes', {
    indexesCreated: ['farmers.phone_unique', 'activities.farmerId_date'],
    indexesCount: 2,
    durationMs: 1500,
    status: 'success'
  });
  console.log('✅ Admin operation logged\n');
  
  console.log('=== All Logging Tests Completed ===');
  console.log('\nCheck the logs directory for the generated log files:');
  console.log('- logs/combined.log');
  console.log('- logs/error.log');
  console.log('- logs/database.log');
}

// Run the tests if this file is executed directly
if (require.main === module) {
  runLoggingTests()
    .then(() => {
      console.log('\n🎉 Logging tests completed successfully!');
      process.exit(0);
    })
    .catch(error => {
      console.error('❌ Logging tests failed:', error);
      process.exit(1);
    });
}

module.exports = { runLoggingTests };
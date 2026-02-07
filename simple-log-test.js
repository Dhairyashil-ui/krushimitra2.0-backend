const { logDBOperation, logDBError } = require('./logger');

// Test logging to file
logDBOperation('testFileLogging', {
  farmerId: 'test_farmer_123',
  operation: 'find',
  collection: 'farmers',
  durationMs: 50,
  status: 'success'
});

const testError = new Error('Test file error');
testError.code = 'TEST_FILE_ERROR';
logDBError('testFileError', testError, {
  farmerId: 'test_farmer_456',
  operation: 'insert',
  collection: 'activities',
  durationMs: 100
});

console.log('Log test completed. Check the logs directory for generated files.');
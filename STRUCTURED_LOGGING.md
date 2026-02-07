# Structured Logging Implementation

This document explains the structured logging implementation for database operations in the KrushiMitra backend.

## Overview

The structured logging system uses Winston to log all database operations with consistent formatting, including context information such as farmerId, operation type, and timestamps. Logs are written to both the console and files for easy monitoring and debugging.

## Implementation Details

### Logger Configuration

The logging system is configured in [logger.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/logger.js) with the following features:

1. **Multiple Transports**:
   - Console output (with color coding)
   - File output (`combined.log` for all logs)
   - File output (`error.log` for error logs)
   - File output (`database.log` for database-specific logs)

2. **Log Rotation**:
   - Maximum file size: 5MB
   - Maximum files: 5 rotations

3. **Structured Format**:
   - JSON format for easy parsing
   - Consistent timestamp format
   - Error stack traces included

### Database Operation Logging

Database operations are logged using two main functions:

1. **`logDBOperation(operation, context)`**:
   - Logs successful database operations
   - Includes operation type, duration, and context information

2. **`logDBError(operation, error, context)`**:
   - Logs database errors
   - Includes error message, stack trace, and context information

### Context Information

All database logs include the following context information:

- **Operation**: The type of database operation (e.g., 'findFarmer', 'insertActivity')
- **FarmerId**: The farmer identifier (when applicable)
- **Collection**: The database collection being accessed
- **Duration**: Operation duration in milliseconds
- **Status**: Operation status ('success', 'not_found', etc.)
- **Timestamp**: ISO formatted timestamp

### Log File Structure

#### Combined Log (`combined.log`)
Contains all application logs in JSON format:
```json
{
  "timestamp": "2023-06-15 10:30:45",
  "level": "info",
  "message": "Database operation executed",
  "service": "krushimitra-db",
  "operation": "findFarmer",
  "farmerId": "farmer_1234567890",
  "durationMs": 45,
  "status": "success"
}
```

#### Error Log (`error.log`)
Contains only error logs with stack traces:
```json
{
  "timestamp": "2023-06-15 10:32:15",
  "level": "error",
  "message": "Database operation failed",
  "service": "krushimitra-db",
  "operation": "insertActivity",
  "farmerId": "farmer_1234567890",
  "error": "Connection timeout",
  "stack": "Error: Connection timeout\n at ...",
  "durationMs": 5000
}
```

#### Database Log (`database.log`)
Contains only database-specific operations:
```json
{
  "timestamp": "2023-06-15 10:35:22",
  "level": "info",
  "message": "Database operation executed",
  "service": "krushimitra-db",
  "operation": "createIndexes",
  "indexesCreated": ["farmers.phone_unique", "activities.farmerId_date"],
  "durationMs": 1250,
  "status": "success"
}
```

## Usage Examples

### Logging a Successful Operation
```javascript
const startTime = Date.now();
// ... perform database operation ...
const duration = Date.now() - startTime;

logDBOperation('findFarmer', {
  farmerId: 'farmer_1234567890',
  collection: 'farmers',
  operation: 'find',
  durationMs: duration,
  status: 'success'
});
```

### Logging an Error
```javascript
const startTime = Date.now();
try {
  // ... perform database operation ...
} catch (error) {
  const duration = Date.now() - startTime;
  
  logDBError('findFarmer', error, {
    farmerId: 'farmer_1234567890',
    collection: 'farmers',
    operation: 'find',
    durationMs: duration
  });
}
```

## Testing

A test script is available to verify the logging implementation:

```bash
npm run test:logging
```

This script will generate sample log entries and output them to the console and log files.

## Benefits

1. **Consistent Format**: All logs follow the same structure for easy parsing
2. **Rich Context**: Logs include relevant context information for troubleshooting
3. **Performance Monitoring**: Operation durations help identify performance issues
4. **Error Tracking**: Detailed error information with stack traces
5. **Multiple Destinations**: Logs are written to console and files for flexibility
6. **Log Rotation**: Prevents log files from growing too large

## Log Analysis

The structured JSON format makes it easy to analyze logs using tools like:

- Log aggregation platforms (ELK Stack, Splunk)
- Custom analysis scripts
- Monitoring dashboards

Example analysis query to find slow operations:
```javascript
// Find all operations that took longer than 1000ms
const slowOps = logs.filter(log => log.durationMs > 1000);
```

This structured logging implementation provides comprehensive visibility into database operations while maintaining performance and scalability.
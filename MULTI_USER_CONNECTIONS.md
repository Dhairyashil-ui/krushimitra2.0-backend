# Multi-User MongoDB Connection System

This document explains how the KrushiMitra backend implements multiple MongoDB users with least privileges for different operations.

## Overview

The system uses three distinct MongoDB users, each with specific permissions:
1. **Reader User** - Read-only access for queries
2. **Writer User** - Read-write access for data modifications
3. **Admin User** - Full administrative access for schema changes

## Implementation Details

### Environment Variables

The system requires the following environment variables in [.env](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/.env):

```env
# Read-only user for queries
DB_READER_USER=krushimitra_reader
DB_READER_PASS=your_reader_password

# Read-write user for modifications
DB_WRITER_USER=krushimitra_writer
DB_WRITER_PASS=your_writer_password

# Admin user for administrative tasks
DB_ADMIN_USER=krushimitra_admin
DB_ADMIN_PASS=your_admin_password

CLUSTER_HOST=krushimitradb.lw8eigi.mongodb.net
```

### Connection Logic

The [db.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/db.js) file implements the connection logic with the `connectToDatabase()` function that accepts an `operationType` parameter:

```javascript
async function connectToDatabase(operationType = 'read') {
  // Returns appropriate client based on operation type
}
```

#### Operation Types

1. **'read'** (default) - Uses the reader user for:
   - GET requests
   - Data queries
   - Reports generation

2. **'write'** - Uses the writer user for:
   - POST/PUT/DELETE requests
   - Data modifications
   - Insertions and updates

3. **'admin'** - Uses the admin user for:
   - Schema changes
   - Index creation
   - Collection management
   - User administration

### Usage Examples

```javascript
// For read operations
const client = await connectToDatabase('read');

// For write operations
const client = await connectToDatabase('write');

// For admin operations
const client = await connectToDatabase('admin');
```

## Benefits

1. **Security**: Each operation uses the minimum required privileges
2. **Auditability**: Database operations can be traced to specific user types
3. **Scalability**: Different connection pools for different operation types
4. **Compliance**: Follows the principle of least privilege

## Future Enhancements

1. Connection pooling for each user type
2. Automatic retry logic for failed connections
3. Enhanced error handling and logging
4. Metrics collection for connection performance

This implementation ensures that the KrushiMitra application follows security best practices while maintaining optimal performance for different types of database operations.
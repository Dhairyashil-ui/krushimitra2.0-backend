# Scripts Update Guide

This document explains how to update existing scripts to use the new multi-user MongoDB connection system.

## Overview

All existing scripts that connect to the database need to be updated to use the new `connectToDatabase(operationType)` function with the appropriate operation type.

## Updating Existing Scripts

### 1. Collection Creation Scripts

Scripts like [ensure-collections.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/ensure-collections.js) and [create-indexes.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/create-indexes.js) should use the 'admin' operation type:

**Before:**
```javascript
const { connectToDatabase } = require('./db');
// ...
const client = await connectToDatabase();
```

**After:**
```javascript
const { connectToDatabase } = require('./db');
// ...
const client = await connectToDatabase('admin');
```

### 2. Data Insertion Scripts

Scripts like [insert-samples.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/insert-samples.js) should use the 'write' operation type:

**Before:**
```javascript
const { connectToDatabase } = require('./db');
// ...
const client = await connectToDatabase();
```

**After:**
```javascript
const { connectToDatabase } = require('./db');
// ...
const client = await connectToDatabase('write');
```

### 3. Data Query Scripts

Scripts like [run-queries.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/run-queries.js) should use the 'read' operation type:

**Before:**
```javascript
const { connectToDatabase } = require('./db');
// ...
const client = await connectToDatabase();
```

**After:**
```javascript
const { connectToDatabase } = require('./db');
// ...
const client = await connectToDatabase('read');
```

### 4. Data Update Scripts

Scripts like [update-data.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/update-data.js) should use the 'write' operation type:

**Before:**
```javascript
const { connectToDatabase } = require('./db');
// ...
const client = await connectToDatabase();
```

**After:**
```javascript
const { connectToDatabase } = require('./db');
// ...
const client = await connectToDatabase('write');
```

## Example Updates

### Update [ensure-collections.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/ensure-collections.js):

```javascript
const { ensureCollections } = require('./db');

async function main() {
  try {
    const result = await ensureCollections();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
```

Note that [ensureCollections()](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/db.js#L72-L117) in [db.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/db.js) already uses the 'admin' operation type internally.

### Update [create-indexes.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/create-indexes.js):

```javascript
const { createIndexes } = require('./db');

async function main() {
  try {
    const result = await createIndexes();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
```

Note that [createIndexes()](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/db.js#L119-L173) in [db.js](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/db.js) already uses the 'admin' operation type internally.

## Testing the Updates

After updating your scripts, test them to ensure they work correctly:

```bash
# Test collection creation
node ensure-collections.js

# Test index creation
node create-indexes.js

# Test data insertion
node insert-samples.js

# Test data queries
node run-queries.js

# Test data updates
node update-data.js
```

## Benefits of the Update

1. **Enhanced Security**: Each script now uses the minimum required privileges
2. **Better Auditing**: Database operations can be traced to specific user types
3. **Improved Performance**: Connection pooling can be optimized for each operation type
4. **Compliance**: Follows the principle of least privilege

## Troubleshooting

If you encounter connection errors after updating:

1. Verify that all three MongoDB Atlas users have been created
2. Check that the environment variables in [.env](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/.env) are correctly set
3. Ensure each user has the appropriate permissions for their role
4. Check the MongoDB Atlas logs for authentication errors

By following this guide, all your scripts will be updated to use the secure multi-user connection system.
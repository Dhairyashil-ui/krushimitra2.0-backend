# Test Plan Execution

This document describes the test plan for verifying the KrushiMitra backend functionality.

## Test Cases

### TC1: Connect to DB
- **Objective**: Verify that the application can connect to the MongoDB database
- **Expected Result**: Connection successful with status "CONNECTED"
- **Implementation**: Uses the MongoDB connection string with credentials from environment variables

### TC2: Create collections if missing
- **Objective**: Verify that all required collections are present in the database
- **Expected Result**: All collections (farmers, activities, mandiprices, schemes, aiinteractions, crop_health, alerts) are present
- **Implementation**: Lists existing collections and checks against required list

### TC3: Insert sample farmer and retrieve by phone
- **Objective**: Verify that farmers can be inserted and retrieved by phone number
- **Expected Result**: Sample farmer can be retrieved using their phone number
- **Implementation**: Inserts a test farmer document and then queries by phone number

### TC4: Log activity and verify it appears in activities query
- **Objective**: Verify that activities can be logged for farmers and retrieved
- **Expected Result**: Logged activity appears in query results for the farmer
- **Implementation**: Inserts an activity record and then queries activities for the farmer

### TC5: Insert mandi price and verify it's returned as latest for crop+location
- **Objective**: Verify that mandi prices can be inserted and retrieved as the latest for a crop and location
- **Expected Result**: Inserted mandi price is returned as the latest for the specified crop and location
- **Implementation**: Inserts a mandi price record and then queries for the latest price by crop and location

### TC6: Golden chance logic test
- **Objective**: Verify that the golden chance logic works correctly and creates alerts
- **Expected Result**: When a price is >10% above the 7-day average, an alert is created
- **Implementation**: 
  1. Inserts historical price data to establish a baseline
  2. Inserts a current price that is >10% above the average
  3. Verifies that an alert is created for the farmer

## Running the Test Plan

### Prerequisites
- Node.js installed
- MongoDB Atlas cluster set up
- Valid credentials in the `.env` file

### Execution
```bash
cd backend
npm run test:plan
```

## Expected Results

With valid MongoDB credentials, all test cases should pass:

1. **TC1**: PASS - Database connection established
2. **TC2**: PASS - All required collections present
3. **TC3**: PASS - Farmer inserted and retrieved by phone
4. **TC4**: PASS - Activity logged and appears in query
5. **TC5**: PASS - Mandi price inserted and returned as latest
6. **TC6**: PASS - Golden chance detected and alert created

## Current Status

The current test execution is failing because the required MongoDB credentials are not set in the `.env` file. To run the tests successfully:

1. Update the `.env` file with valid MongoDB credentials:
   ```
   DB_USER=your_actual_username
   DB_PASS=your_actual_password
   CLUSTER_HOST=your_actual_cluster_host
   ```

2. Ensure the MongoDB Atlas cluster is accessible from your network

3. Run the test plan again

## Test Results Files

- [test-results.json](test-results.json) - Actual results with missing credentials
- [test-results-detailed.json](test-results-detailed.json) - Expected results with valid credentials
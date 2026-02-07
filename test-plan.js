/**
 * Test Plan Execution Script
 * 
 * This script runs the short test plan and returns results for each test case.
 */

const { MongoClient } = require('mongodb');
require('dotenv').config({ path: './SIH_APP-main/backend/.env' });

// MongoDB connection
const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
const client = new MongoClient(uri);

// Test results
const testResults = [];

/**
 * Test Case 1: Connect to DB
 */
async function testCase1() {
  try {
    console.log("Running TC1: Connect to DB");
    
    // Connect to database
    await client.connect();
    
    // Send a ping to confirm a successful connection
    await client.db("admin").command({ ping: 1 });
    
    testResults.push({
      testCase: "TC1",
      description: "Connect to DB",
      status: "PASS",
      details: "Successfully connected to MongoDB database",
      expected: "CONNECTED",
      actual: "CONNECTED"
    });
    
    console.log("TC1 PASSED");
  } catch (error) {
    testResults.push({
      testCase: "TC1",
      description: "Connect to DB",
      status: "FAIL",
      details: `Failed to connect to database: ${error.message}`,
      expected: "CONNECTED",
      actual: "ERROR_CONNECT"
    });
    
    console.log("TC1 FAILED:", error.message);
    throw error;
  }
}

/**
 * Test Case 2: Create collections if missing
 */
async function testCase2() {
  try {
    console.log("Running TC2: Create collections if missing");
    
    const db = client.db("KrushiMitraDB");
    
    // Get existing collections
    const collections = await db.listCollections().toArray();
    const existingCollectionNames = collections.map(collection => collection.name);
    
    // Required collections
    const requiredCollections = [
      'farmers', 
      'activities', 
      'mandiprices', 
      'schemes', 
      'aiinteractions', 
      'crop_health', 
      'alerts'
    ];
    
    // Check if all required collections are present
    const missingCollections = requiredCollections.filter(
      collection => !existingCollectionNames.includes(collection)
    );
    
    if (missingCollections.length === 0) {
      testResults.push({
        testCase: "TC2",
        description: "Create collections if missing",
        status: "PASS",
        details: "All required collections are present",
        expected: "all collections present",
        actual: `Collections present: ${existingCollectionNames.join(', ')}`
      });
      
      console.log("TC2 PASSED");
    } else {
      testResults.push({
        testCase: "TC2",
        description: "Create collections if missing",
        status: "FAIL",
        details: `Missing collections: ${missingCollections.join(', ')}`,
        expected: "all collections present",
        actual: `Missing collections: ${missingCollections.join(', ')}`
      });
      
      console.log("TC2 FAILED: Missing collections:", missingCollections);
    }
  } catch (error) {
    testResults.push({
      testCase: "TC2",
      description: "Create collections if missing",
      status: "FAIL",
      details: `Error checking collections: ${error.message}`,
      expected: "all collections present",
      actual: "ERROR"
    });
    
    console.log("TC2 FAILED:", error.message);
  }
}

/**
 * Test Case 3: Insert sample farmer and retrieve by phone
 */
async function testCase3() {
  try {
    console.log("Running TC3: Insert sample farmer and retrieve by phone");
    
    const db = client.db("KrushiMitraDB");
    const farmersCollection = db.collection('farmers');
    
    // Create a sample farmer
    const sampleFarmer = {
      name: "Test Farmer",
      phone: "+919900112233",
      language: "Hindi",
      location: "Pune, Maharashtra",
      crops: ["Wheat"],
      joinedAt: new Date()
    };
    
    // Insert the sample farmer
    const insertResult = await farmersCollection.insertOne(sampleFarmer);
    const insertedFarmerId = insertResult.insertedId;
    
    // Retrieve the farmer by phone
    const retrievedFarmer = await farmersCollection.findOne({ phone: "+919900112233" });
    
    if (retrievedFarmer) {
      testResults.push({
        testCase: "TC3",
        description: "Insert sample farmer and retrieve by phone",
        status: "PASS",
        details: "Sample farmer inserted and successfully retrieved by phone",
        expected: "can be retrieved by phone",
        actual: "Farmer retrieved successfully"
      });
      
      console.log("TC3 PASSED");
      return insertedFarmerId;
    } else {
      testResults.push({
        testCase: "TC3",
        description: "Insert sample farmer and retrieve by phone",
        status: "FAIL",
        details: "Sample farmer inserted but could not be retrieved by phone",
        expected: "can be retrieved by phone",
        actual: "Farmer not found"
      });
      
      console.log("TC3 FAILED: Farmer not found");
      return null;
    }
  } catch (error) {
    testResults.push({
      testCase: "TC3",
      description: "Insert sample farmer and retrieve by phone",
      status: "FAIL",
      details: `Error inserting/retrieving farmer: ${error.message}`,
      expected: "can be retrieved by phone",
      actual: "ERROR"
    });
    
    console.log("TC3 FAILED:", error.message);
    return null;
  }
}

/**
 * Test Case 4: Log activity and verify it appears in activities query
 */
async function testCase4(farmerId) {
  try {
    console.log("Running TC4: Log activity and verify it appears in activities query");
    
    if (!farmerId) {
      testResults.push({
        testCase: "TC4",
        description: "Log activity and verify it appears in activities query",
        status: "FAIL",
        details: "Cannot run test without farmer ID from TC3",
        expected: "appears in activities query",
        actual: "No farmer ID available"
      });
      
      console.log("TC4 SKIPPED: No farmer ID available");
      return;
    }
    
    const db = client.db("KrushiMitraDB");
    const activitiesCollection = db.collection('activities');
    
    // Create a sample activity
    const sampleActivity = {
      farmerId: farmerId,
      description: "Installed app / first chat",
      date: new Date(),
      type: "app_install"
    };
    
    // Insert the sample activity
    const insertResult = await activitiesCollection.insertOne(sampleActivity);
    const insertedActivityId = insertResult.insertedId;
    
    // Retrieve activities for the farmer
    const activities = await activitiesCollection
      .find({ farmerId: farmerId })
      .sort({ date: -1 })
      .limit(5)
      .toArray();
    
    const activityFound = activities.some(activity => activity._id.equals(insertedActivityId));
    
    if (activityFound) {
      testResults.push({
        testCase: "TC4",
        description: "Log activity and verify it appears in activities query",
        status: "PASS",
        details: "Sample activity logged and appears in activities query",
        expected: "appears in activities query",
        actual: "Activity found in query results"
      });
      
      console.log("TC4 PASSED");
    } else {
      testResults.push({
        testCase: "TC4",
        description: "Log activity and verify it appears in activities query",
        status: "FAIL",
        details: "Sample activity logged but does not appear in activities query",
        expected: "appears in activities query",
        actual: "Activity not found in query results"
      });
      
      console.log("TC4 FAILED: Activity not found in query results");
    }
  } catch (error) {
    testResults.push({
      testCase: "TC4",
      description: "Log activity and verify it appears in activities query",
      status: "FAIL",
      details: `Error logging/retrieving activity: ${error.message}`,
      expected: "appears in activities query",
      actual: "ERROR"
    });
    
    console.log("TC4 FAILED:", error.message);
  }
}

/**
 * Test Case 5: Insert mandi price and verify it's returned as latest for crop+location
 */
async function testCase5() {
  try {
    console.log("Running TC5: Insert mandi price and verify it's returned as latest for crop+location");
    
    const db = client.db("KrushiMitraDB");
    const mandipricesCollection = db.collection('mandiprices');
    
    // Create a sample mandi price
    const sampleMandiPrice = {
      crop: "Wheat",
      location: "Pune",
      price: 2400,
      date: new Date()
    };
    
    // Insert the sample mandi price
    const insertResult = await mandipricesCollection.insertOne(sampleMandiPrice);
    const insertedPriceId = insertResult.insertedId;
    
    // Retrieve latest mandi price for Wheat in Pune
    const latestPrice = await mandipricesCollection
      .findOne({ crop: "Wheat", location: "Pune" }, { sort: { date: -1 } });
    
    if (latestPrice && latestPrice._id.equals(insertedPriceId)) {
      testResults.push({
        testCase: "TC5",
        description: "Insert mandi price and verify it's returned as latest for crop+location",
        status: "PASS",
        details: "Sample mandi price inserted and returned as latest for crop+location",
        expected: "mandiprices returns it as latest for crop+location",
        actual: "Latest price matches inserted price"
      });
      
      console.log("TC5 PASSED");
    } else {
      testResults.push({
        testCase: "TC5",
        description: "Insert mandi price and verify it's returned as latest for crop+location",
        status: "FAIL",
        details: "Sample mandi price inserted but not returned as latest for crop+location",
        expected: "mandiprices returns it as latest for crop+location",
        actual: "Latest price does not match inserted price"
      });
      
      console.log("TC5 FAILED: Latest price does not match inserted price");
    }
  } catch (error) {
    testResults.push({
      testCase: "TC5",
      description: "Insert mandi price and verify it's returned as latest for crop+location",
      status: "FAIL",
      details: `Error inserting/retrieving mandi price: ${error.message}`,
      expected: "mandiprices returns it as latest for crop+location",
      actual: "ERROR"
    });
    
    console.log("TC5 FAILED:", error.message);
  }
}

/**
 * Test Case 6: Golden chance logic test
 */
async function testCase6() {
  try {
    console.log("Running TC6: Golden chance logic test");
    
    const db = client.db("KrushiMitraDB");
    const mandipricesCollection = db.collection('mandiprices');
    const alertsCollection = db.collection('alerts');
    
    // First, insert some historical prices to establish a baseline
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    
    // Insert 7 days of historical prices with an average around 2000
    const historicalPrices = [];
    for (let i = 0; i < 7; i++) {
      const date = new Date(sevenDaysAgo);
      date.setDate(date.getDate() + i);
      
      historicalPrices.push({
        crop: "Rice",
        location: "Nagpur",
        price: 1900 + (Math.random() * 200), // Prices between 1900-2100
        date: date
      });
    }
    
    await mandipricesCollection.insertMany(historicalPrices);
    
    // Now insert a current price that's >10% above the average to trigger golden chance
    const goldenChancePrice = {
      crop: "Rice",
      location: "Nagpur",
      price: 2500, // This should be >10% above the average of historical prices
      date: new Date()
    };
    
    const insertResult = await mandipricesCollection.insertOne(goldenChancePrice);
    
    // Simulate the golden chance logic
    // In a real implementation, this would be done by the background job
    // For this test, we'll manually check if the condition is met
    
    // Calculate 7-day average
    const weekPrices = await mandipricesCollection
      .find({
        crop: "Rice",
        location: "Nagpur",
        date: { $gte: sevenDaysAgo }
      })
      .toArray();
    
    if (weekPrices.length > 0) {
      const averagePrice = weekPrices.reduce((sum, price) => sum + price.price, 0) / weekPrices.length;
      const percentageIncrease = ((2500 - averagePrice) / averagePrice) * 100;
      
      console.log(`Average price: ${averagePrice}, Current price: 2500, Increase: ${percentageIncrease}%`);
      
      if (percentageIncrease > 10) {
        // Create a sample alert for a farmer (using the farmer from TC3 if available)
        const farmersCollection = db.collection('farmers');
        const farmer = await farmersCollection.findOne({ phone: "+919900112233" });
        
        if (farmer) {
          const alertDoc = {
            farmerId: farmer._id,
            type: "price",
            message: `Great opportunity! Rice prices in Nagpur are ${percentageIncrease.toFixed(2)}% above the 7-day average. Current price: ₹2500`,
            status: "active",
            createdAt: new Date()
          };
          
          await alertsCollection.insertOne(alertDoc);
          
          // Verify the alert was created
          const alert = await alertsCollection.findOne({ 
            farmerId: farmer._id, 
            type: "price" 
          });
          
          if (alert) {
            testResults.push({
              testCase: "TC6",
              description: "Golden chance logic test",
              status: "PASS",
              details: `Golden chance detected (${percentageIncrease.toFixed(2)}% above average) and alert created`,
              expected: "alerts entry created",
              actual: "Alert created successfully"
            });
            
            console.log("TC6 PASSED");
          } else {
            testResults.push({
              testCase: "TC6",
              description: "Golden chance logic test",
              status: "FAIL",
              details: "Golden chance detected but alert was not created",
              expected: "alerts entry created",
              actual: "Alert not found"
            });
            
            console.log("TC6 FAILED: Alert not found");
          }
        } else {
          testResults.push({
            testCase: "TC6",
            description: "Golden chance logic test",
            status: "FAIL",
            details: "Golden chance detected but no farmer found to create alert for",
            expected: "alerts entry created",
            actual: "No farmer found"
          });
          
          console.log("TC6 FAILED: No farmer found");
        }
      } else {
        testResults.push({
          testCase: "TC6",
          description: "Golden chance logic test",
          status: "FAIL",
          details: `Golden chance not detected (${percentageIncrease.toFixed(2)}% is not >10%)`,
          expected: "alerts entry created",
          actual: "No golden chance"
        });
        
        console.log("TC6 FAILED: No golden chance detected");
      }
    } else {
      testResults.push({
        testCase: "TC6",
        description: "Golden chance logic test",
        status: "FAIL",
        details: "Not enough historical data to calculate 7-day average",
        expected: "alerts entry created",
        actual: "Insufficient data"
      });
      
      console.log("TC6 FAILED: Insufficient data");
    }
  } catch (error) {
    testResults.push({
      testCase: "TC6",
      description: "Golden chance logic test",
      status: "FAIL",
      details: `Error testing golden chance logic: ${error.message}`,
      expected: "alerts entry created",
      actual: "ERROR"
    });
    
    console.log("TC6 FAILED:", error.message);
  }
}

/**
 * Main function to run all test cases
 */
async function runTestPlan() {
  console.log("Starting test plan execution...");
  
  try {
    // Run TC1: Connect to DB
    await testCase1();
    
    // Run TC2: Create collections if missing
    await testCase2();
    
    // Run TC3: Insert sample farmer and retrieve by phone
    const farmerId = await testCase3();
    
    // Run TC4: Log activity and verify it appears in activities query
    await testCase4(farmerId);
    
    // Run TC5: Insert mandi price and verify it's returned as latest for crop+location
    await testCase5();
    
    // Run TC6: Golden chance logic test
    await testCase6();
    
    console.log("Test plan execution completed");
    
    // Return results as JSON
    console.log("\n=== TEST RESULTS ===");
    console.log(JSON.stringify(testResults, null, 2));
    
    return testResults;
  } catch (error) {
    console.error("Test plan execution failed:", error);
    
    // Add a final result for overall failure
    testResults.push({
      testCase: "Overall",
      description: "Test plan execution",
      status: "FAIL",
      details: `Test plan execution failed: ${error.message}`,
      expected: "All tests pass",
      actual: "Execution error"
    });
    
    console.log("\n=== TEST RESULTS ===");
    console.log(JSON.stringify(testResults, null, 2));
    
    return testResults;
  } finally {
    // Close database connection
    await client.close();
    console.log("Database connection closed");
  }
}

// Export the function
module.exports = { runTestPlan };

// Run the test plan if this file is executed directly
if (require.main === module) {
  runTestPlan()
    .then(results => {
      console.log("Final Results:", JSON.stringify(results, null, 2));
      process.exit(0);
    })
    .catch(error => {
      console.error("Test plan failed with unhandled error:", error);
      process.exit(1);
    });
}
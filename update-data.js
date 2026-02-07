const { MongoClient } = require('mongodb');
require('dotenv').config();

// Check if environment variables are set
if (!process.env.DB_USER || !process.env.DB_PASS || !process.env.CLUSTER_HOST) {
  console.log(JSON.stringify({
    status: "ERROR_CONNECT",
    message: "Missing environment variables: DB_USER, DB_PASS, and/or CLUSTER_HOST must be set in .env file"
  }, null, 2));
  process.exit(1);
}

// MongoDB connection string with placeholders
const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;

async function updateData() {
  const client = new MongoClient(uri);
  
  try {
    // Connect to the database
    await client.connect();
    const db = client.db("KrushiMitraDB");
    
    // 1. Update the test farmer language from Hindi to Marathi
    const farmersCollection = db.collection('farmers');
    const updateResult = await farmersCollection.updateOne(
      { phone: "+919900112233" },
      { $set: { language: "Marathi" } }
    );
    
    if (updateResult.modifiedCount > 0) {
      console.log(JSON.stringify({
        status: "UPDATED_LANGUAGE",
        modifiedCount: updateResult.modifiedCount
      }, null, 2));
    }
    
    // 2. For mandi automation test
    const mandipricesCollection = db.collection('mandiprices');
    
    // Get the latest price for Wheat in Pune
    const latestPriceDoc = await mandipricesCollection
      .findOne({ crop: "Wheat", location: "Pune" }, { sort: { date: -1 } });
    
    if (latestPriceDoc) {
      // Calculate 7-day average
      const sevenDaysAgo = new Date();
      sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
      
      const weekPrices = await mandipricesCollection
        .find({
          crop: "Wheat",
          location: "Pune",
          date: { $gte: sevenDaysAgo }
        })
        .toArray();
      
      if (weekPrices.length > 0) {
        const averagePrice = weekPrices.reduce((sum, price) => sum + price.price, 0) / weekPrices.length;
        const percentageIncrease = ((latestPriceDoc.price - averagePrice) / averagePrice) * 100;
        
        if (percentageIncrease > 10) {
          // Set isGoldenChance=true for that document
          await mandipricesCollection.updateOne(
            { _id: latestPriceDoc._id },
            { $set: { isGoldenChance: true } }
          );
          
          // Insert an alerts document for the test farmer
          const farmer = await farmersCollection.findOne({ phone: "+919900112233" });
          
          if (farmer) {
            const alertsCollection = db.collection('alerts');
            await alertsCollection.insertOne({
              farmerId: farmer._id,
              type: "price",
              message: `Great opportunity! Wheat prices in Pune are ${percentageIncrease.toFixed(2)}% above the 7-day average. Current price: ₹${latestPriceDoc.price}`,
              status: "active",
              createdAt: new Date()
            });
          }
          
          console.log(JSON.stringify({
            status: "GOLDEN_CHANCE_FOUND",
            message: `Wheat prices are ${percentageIncrease.toFixed(2)}% above the 7-day average`,
            latestPrice: latestPriceDoc.price,
            averagePrice: averagePrice
          }, null, 2));
        } else {
          console.log(JSON.stringify({
            status: "NO_GOLDEN_CHANCE",
            message: `Wheat prices are only ${percentageIncrease.toFixed(2)}% above the 7-day average (below 10% threshold)`,
            latestPrice: latestPriceDoc.price,
            averagePrice: averagePrice
          }, null, 2));
        }
      } else {
        console.log(JSON.stringify({
          status: "NO_GOLDEN_CHANCE",
          message: "Not enough historical data to calculate 7-day average"
        }, null, 2));
      }
    } else {
      console.log(JSON.stringify({
        status: "NO_GOLDEN_CHANCE",
        message: "No wheat prices found for Pune"
      }, null, 2));
    }
    
    // 3. Mark the sample alert status as sent
    const alertsCollection = db.collection('alerts');
    const alertUpdateResult = await alertsCollection.updateOne(
      { status: "active" },
      { $set: { status: "sent" } }
    );
    
    if (alertUpdateResult.modifiedCount > 0) {
      console.log(JSON.stringify({
        status: "ALERT_SENT",
        modifiedCount: alertUpdateResult.modifiedCount
      }, null, 2));
    }
    
  } catch (error) {
    console.log(JSON.stringify({
      status: "ERROR",
      message: error.message
    }, null, 2));
  } finally {
    // Ensures that the client will close when you finish/error
    try {
      await client.close();
    } catch (closeError) {
      // Ignore close errors
    }
  }
}

// Run the function
updateData();
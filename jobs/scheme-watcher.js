/**
 * Scheme Watcher Job
 * 
 * This background job runs daily to check official feeds for new agricultural schemes,
 * insert them into the database, and create alerts for eligible farmers.
 */

const { MongoClient } = require('mongodb');
require('dotenv').config();

// MongoDB connection
const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
const client = new MongoClient(uri);

// Job execution statistics
let stats = {
  schemesFetched: 0,
  newSchemesDetected: 0,
  schemesInserted: 0,
  alertsCreated: 0,
  farmersNotified: 0
};

// Sources for scheme data
const sources = [
  {
    name: "Ministry of Agriculture",
    url: "https://mog.gov.in/api/schemes",
    enabled: true
  },
  {
    name: "State Agricultural Department",
    url: "https://agri.state.gov.in/api/schemes",
    enabled: true
  }
];

/**
 * Fetch official scheme data from a source
 * @param {Object} source - The data source configuration
 * @returns {Promise<Array>} Array of scheme data
 */
async function fetchOfficialSchemes(source) {
  console.log(`Fetching schemes from ${source.name}...`);
  
  // In a real implementation, this would make HTTP requests to official APIs
  // For demo purposes, we'll return mock data
  
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  // Mock data
  const mockSchemes = [
    {
      title: "PM Kisan Samman Nidhi",
      description: "Financial assistance to small and marginal farmer families",
      eligibility: "Small and marginal farmer families with cultivable land up to 2 hectares",
      startDate: new Date('2023-04-01'),
      endDate: new Date('2024-03-31'),
      location: "all",
      benefits: "₹6000 per year transferred directly to farmer's bank account in 3 equal installments",
      applicationProcess: "Register through Common Service Centers or online portal",
      source: source.name
    },
    {
      title: "Pradhan Mantri Fasal Bima Yojana",
      description: "Crop insurance scheme to provide insurance coverage and financial support to farmers",
      eligibility: "All farmers including sharecroppers and tenant farmers growing notified crops",
      startDate: new Date('2023-01-01'),
      endDate: new Date('2023-12-31'),
      location: "Maharashtra",
      benefits: "Low premium rates and comprehensive insurance coverage",
      applicationProcess: "Contact local agriculture department or insurance provider",
      source: source.name
    }
  ];
  
  stats.schemesFetched += mockSchemes.length;
  console.log(`Fetched ${mockSchemes.length} schemes from ${source.name}`);
  
  return mockSchemes;
}

/**
 * Check if a scheme already exists in the database
 * @param {Object} scheme - Scheme object to check
 * @returns {Promise<boolean>} True if scheme exists
 */
async function schemeExists(scheme) {
  try {
    const db = client.db("KrushiMitraDB");
    const collection = db.collection("schemes");
    
    const existing = await collection.findOne({
      title: scheme.title,
      startDate: scheme.startDate
    });
    
    return !!existing;
  } catch (error) {
    console.error("Error checking if scheme exists:", error);
    throw error;
  }
}

/**
 * Insert new schemes into the database
 * @param {Array} schemes - Array of scheme objects to insert
 * @returns {Promise<Array>} Array of inserted scheme objects
 */
async function insertNewSchemes(schemes) {
  try {
    const db = client.db("KrushiMitraDB");
    const collection = db.collection("schemes");
    
    const newSchemes = [];
    
    for (const scheme of schemes) {
      const exists = await schemeExists(scheme);
      
      if (!exists) {
        scheme.createdAt = new Date();
        const result = await collection.insertOne(scheme);
        scheme._id = result.insertedId;
        newSchemes.push(scheme);
        stats.newSchemesDetected++;
        stats.schemesInserted++;
        console.log(`Inserted new scheme: ${scheme.title}`);
      }
    }
    
    return newSchemes;
  } catch (error) {
    console.error("Error inserting new schemes:", error);
    throw error;
  }
}

/**
 * Find farmers eligible for a scheme based on location and eligibility criteria
 * @param {Object} scheme - Scheme object
 * @returns {Promise<Array>} Array of eligible farmer objects
 */
async function findEligibleFarmers(scheme) {
  try {
    const db = client.db("KrushiMitraDB");
    const collection = db.collection("farmers");
    
    let query = {};
    
    // Location-based filtering
    if (scheme.location !== "all") {
      query.location = { $regex: new RegExp(scheme.location, 'i') };
    }
    
    // Additional eligibility criteria could be implemented here
    // For example, checking land size, crop types, etc.
    
    const farmers = await collection.find(query).toArray();
    return farmers;
  } catch (error) {
    console.error("Error finding eligible farmers:", error);
    throw error;
  }
}

/**
 * Create scheme alerts for eligible farmers
 * @param {Array} farmers - Array of farmer objects
 * @param {Object} scheme - Scheme object
 * @returns {Promise<void>}
 */
async function createSchemeAlerts(farmers, scheme) {
  try {
    const db = client.db("KrushiMitraDB");
    const collection = db.collection("alerts");
    
    // Create alerts for each farmer
    for (const farmer of farmers) {
      const alert = {
        farmerId: farmer._id,
        type: "scheme",
        message: `New scheme available: ${scheme.title} - ${scheme.benefits}`,
        status: "active",
        createdAt: new Date(),
        metadata: {
          schemeId: scheme._id,
          title: scheme.title,
          benefits: scheme.benefits,
          eligibility: scheme.eligibility,
          applicationProcess: scheme.applicationProcess
        }
      };
      
      await collection.insertOne(alert);
      stats.alertsCreated++;
    }
    
    stats.farmersNotified += farmers.length;
    console.log(`Created ${farmers.length} scheme alerts for "${scheme.title}"`);
  } catch (error) {
    console.error("Error creating scheme alerts:", error);
    throw error;
  }
}

/**
 * Main function to run the scheme watcher job
 * @returns {Promise<Object>} Job execution report
 */
async function runSchemeWatcher() {
  const startTime = new Date();
  console.log("Starting Scheme Watcher job...");
  
  // Reset statistics
  stats = {
    schemesFetched: 0,
    newSchemesDetected: 0,
    schemesInserted: 0,
    alertsCreated: 0,
    farmersNotified: 0
  };
  
  try {
    // Connect to database
    await client.connect();
    console.log("Connected to database");
    
    // Process each data source
    for (const source of sources) {
      if (!source.enabled) {
        console.log(`Skipping disabled source: ${source.name}`);
        continue;
      }
      
      try {
        // Fetch schemes from official source
        const schemes = await fetchOfficialSchemes(source);
        
        // Insert new schemes into database
        const newSchemes = await insertNewSchemes(schemes);
        
        // Create alerts for each new scheme
        for (const scheme of newSchemes) {
          // Find eligible farmers
          const farmers = await findEligibleFarmers(scheme);
          
          if (farmers.length > 0) {
            // Create alerts for farmers
            await createSchemeAlerts(farmers, scheme);
          }
        }
      } catch (sourceError) {
        console.error(`Error processing source ${source.name}:`, sourceError);
        // Continue with other sources even if one fails
      }
    }
    
    const endTime = new Date();
    const durationSeconds = (endTime - startTime) / 1000;
    
    // Create success report
    const report = {
      jobName: "Scheme Watcher",
      status: "SUCCESS",
      startedAt: startTime,
      completedAt: endTime,
      durationSeconds: durationSeconds,
      statistics: { ...stats },
      details: {
        sources: sources.map(source => ({
          name: source.name,
          status: "SUCCESS"
        }))
      }
    };
    
    console.log("Scheme Watcher job completed successfully");
    return report;
  } catch (error) {
    const endTime = new Date();
    const durationSeconds = (endTime - startTime) / 1000;
    
    // Create failure report
    const report = {
      jobName: "Scheme Watcher",
      status: "FAILURE",
      startedAt: startTime,
      completedAt: endTime,
      durationSeconds: durationSeconds,
      error: {
        code: "JOB_EXECUTION_ERROR",
        message: error.message,
        details: error.stack
      },
      statistics: { ...stats }
    };
    
    console.error("Scheme Watcher job failed:", error);
    return report;
  } finally {
    // Close database connection
    await client.close();
    console.log("Database connection closed");
  }
}

// Export the job function
module.exports = { runSchemeWatcher };

// Run the job if this file is executed directly
if (require.main === module) {
  runSchemeWatcher()
    .then(report => {
      console.log("Job Report:", JSON.stringify(report, null, 2));
      process.exit(0);
    })
    .catch(error => {
      console.error("Job failed with unhandled error:", error);
      process.exit(1);
    });
}
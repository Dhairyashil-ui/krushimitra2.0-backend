/**
 * Background Job Scheduler
 * 
 * This module schedules and runs the background jobs for the KrushiMitra application.
 * It uses node-cron to schedule jobs at specified intervals.
 */

const cron = require('node-cron');
const { runMandiPriceFetcher } = require('./mandi-price-fetcher');
const { runSchemeWatcher } = require('./scheme-watcher');

// Job execution history
const jobHistory = [];

/**
 * Log job execution result
 * @param {Object} report - Job execution report
 */
function logJobResult(report) {
  const logEntry = {
    timestamp: new Date(),
    jobName: report.jobName,
    status: report.status,
    durationSeconds: report.durationSeconds,
    statistics: report.statistics
  };
  
  jobHistory.push(logEntry);
  
  // Keep only the last 100 job executions
  if (jobHistory.length > 100) {
    jobHistory.shift();
  }
  
  console.log(`[${report.jobName}] ${report.status} - Duration: ${report.durationSeconds}s`);
  
  if (report.statistics) {
    console.log(`[${report.jobName}] Statistics:`, JSON.stringify(report.statistics));
  }
  
  if (report.error) {
    console.error(`[${report.jobName}] Error:`, report.error.message);
  }
}

/**
 * Schedule the Mandi Price Fetcher job
 * Runs every hour (at minute 0)
 */
function scheduleMandiPriceFetcher() {
  console.log("Scheduling Mandi Price Fetcher job (runs every hour)");
  
  cron.schedule('0 * * * *', async () => {
    try {
      console.log("Starting scheduled Mandi Price Fetcher job");
      const report = await runMandiPriceFetcher();
      logJobResult(report);
    } catch (error) {
      console.error("Scheduled Mandi Price Fetcher job failed:", error);
      logJobResult({
        jobName: "Mandi Price Fetcher",
        status: "FAILURE",
        durationSeconds: 0,
        error: {
          message: error.message
        }
      });
    }
  });
}

/**
 * Schedule the Scheme Watcher job
 * Runs daily at 6:00 AM
 */
function scheduleSchemeWatcher() {
  console.log("Scheduling Scheme Watcher job (runs daily at 6:00 AM)");
  
  cron.schedule('0 6 * * *', async () => {
    try {
      console.log("Starting scheduled Scheme Watcher job");
      const report = await runSchemeWatcher();
      logJobResult(report);
    } catch (error) {
      console.error("Scheduled Scheme Watcher job failed:", error);
      logJobResult({
        jobName: "Scheme Watcher",
        status: "FAILURE",
        durationSeconds: 0,
        error: {
          message: error.message
        }
      });
    }
  });
}

/**
 * Start all scheduled jobs
 */
function startAllJobs() {
  console.log("Starting background job scheduler");
  
  // Schedule all jobs
  scheduleMandiPriceFetcher();
  scheduleSchemeWatcher();
  
  console.log("All background jobs scheduled");
}

/**
 * Get job execution history
 * @returns {Array} Array of job execution logs
 */
function getJobHistory() {
  return [...jobHistory];
}

/**
 * Run all jobs once (for testing purposes)
 */
async function runAllJobsOnce() {
  console.log("Running all background jobs once (for testing)");
  
  try {
    console.log("Running Mandi Price Fetcher job");
    const mandiReport = await runMandiPriceFetcher();
    logJobResult(mandiReport);
    
    console.log("Running Scheme Watcher job");
    const schemeReport = await runSchemeWatcher();
    logJobResult(schemeReport);
    
    console.log("All jobs completed");
  } catch (error) {
    console.error("Error running jobs:", error);
  }
}

// Export functions
module.exports = {
  startAllJobs,
  getJobHistory,
  runAllJobsOnce
};

// Start jobs if this file is executed directly
if (require.main === module) {
  // Check if we should run jobs once or start the scheduler
  const runOnce = process.argv.includes('--run-once');
  
  if (runOnce) {
    runAllJobsOnce().then(() => {
      console.log("All jobs executed once, exiting");
      process.exit(0);
    }).catch(error => {
      console.error("Error running jobs:", error);
      process.exit(1);
    });
  } else {
    startAllJobs();
  }
}
const winston = require('winston');
const path = require('path');

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, '..', 'logs');
const fs = require('fs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir);
}

// Create the logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'krushimitra-backend' },
  transports: [
    // Write all logs with level `info` and below to `combined.log`
    new winston.transports.File({ 
      filename: path.join(logsDir, 'combined.log'),
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
    
    // Write all logs with level `error` and below to `error.log`
    new winston.transports.File({ 
      filename: path.join(logsDir, 'error.log'),
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
    
    // Write to console
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Database operation logger with structured logging
const dbLogger = {
  // Log database connection attempts
  logConnection: (operationType, user, status, details = {}) => {
    logger.info('Database connection attempt', {
      operationType,
      user,
      status,
      timestamp: new Date().toISOString(),
      ...details
    });
  },

  // Log database queries
  logQuery: (operation, collection, farmerId = null, status, duration, details = {}) => {
    logger.info('Database query executed', {
      operation,
      collection,
      farmerId,
      status,
      durationMs: duration,
      timestamp: new Date().toISOString(),
      ...details
    });
  },

  // Log database modifications
  logModification: (operation, collection, farmerId = null, documentId = null, status, duration, details = {}) => {
    logger.info('Database modification executed', {
      operation,
      collection,
      farmerId,
      documentId,
      status,
      durationMs: duration,
      timestamp: new Date().toISOString(),
      ...details
    });
  },

  // Log database errors
  logError: (operation, collection, farmerId = null, error, details = {}) => {
    logger.error('Database operation failed', {
      operation,
      collection,
      farmerId,
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString(),
      ...details
    });
  },

  // Log database admin operations
  logAdminOperation: (operation, user, status, duration, details = {}) => {
    logger.info('Database admin operation executed', {
      operation,
      user,
      status,
      durationMs: duration,
      timestamp: new Date().toISOString(),
      ...details
    });
  }
};

module.exports = { logger, dbLogger };
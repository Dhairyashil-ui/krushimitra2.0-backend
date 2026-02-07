const winston = require('winston');
const path = require('path');

// Create logs directory if it doesn't exist
const fs = require('fs');
const logsDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir);
}

// Define custom log format
const logFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.printf(({ timestamp, level, message, stack, ...meta }) => {
    let logEntry = {
      timestamp,
      level,
      message
    };

    // Add stack trace if it exists (for errors)
    if (stack) {
      logEntry.stack = stack;
    }

    // Add metadata if it exists
    if (Object.keys(meta).length > 0) {
      logEntry = { ...logEntry, ...meta };
    }

    return JSON.stringify(logEntry);
  })
);

// Create the logger
const logger = winston.createLogger({
  level: 'info',
  format: logFormat,
  defaultMeta: { service: 'krushimitra-backend' },
  transports: [
    // Write all logs with level `info` and below to `combined.log`
    new winston.transports.File({ 
      filename: path.join(logsDir, 'combined.log'),
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
    
    // Write all error logs to `error.log`
    new winston.transports.File({ 
      filename: path.join(logsDir, 'error.log'),
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    })
  ]
});

// If we're not in production, also log to the console
if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.combine(
      winston.format.colorize(),
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
      winston.format.printf(({ timestamp, level, message, ...meta }) => {
        let logMessage = `${timestamp} [${level}]: ${message}`;
        
        // Add metadata if it exists
        if (Object.keys(meta).length > 0) {
          logMessage += ` ${JSON.stringify(meta)}`;
        }
        
        return logMessage;
      })
    )
  }));
}

// Database operation logger
const dbLogger = winston.createLogger({
  level: 'info',
  format: logFormat,
  defaultMeta: { service: 'krushimitra-db' },
  transports: [
    // Write all database logs to `database.log`
    new winston.transports.File({ 
      filename: path.join(logsDir, 'database.log'),
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
    
    // Also log to combined log
    new winston.transports.File({ 
      filename: path.join(logsDir, 'combined.log'),
      maxsize: 5242880, // 5MB
      maxFiles: 5
    })
  ]
});

// If we're not in production, also log database operations to the console
if (process.env.NODE_ENV !== 'production') {
  dbLogger.add(new winston.transports.Console({
    format: winston.format.combine(
      winston.format.colorize(),
      winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
      winston.format.printf(({ timestamp, level, message, ...meta }) => {
        let logMessage = `${timestamp} [DB ${level}]: ${message}`;
        
        // Add metadata if it exists
        if (Object.keys(meta).length > 0) {
          logMessage += ` ${JSON.stringify(meta)}`;
        }
        
        return logMessage;
      })
    )
  }));
}

// Helper function to log database operations
function logDBOperation(operation, context = {}) {
  const logData = {
    operation,
    ...context,
    timestamp: new Date().toISOString()
  };
  
  dbLogger.info('Database operation executed', logData);
}

// Helper function to log database errors
function logDBError(operation, error, context = {}) {
  const logData = {
    operation,
    error: error.message,
    stack: error.stack,
    ...context,
    timestamp: new Date().toISOString()
  };
  
  dbLogger.error('Database operation failed', logData);
}

module.exports = {
  logger,
  dbLogger,
  logDBOperation,
  logDBError
};
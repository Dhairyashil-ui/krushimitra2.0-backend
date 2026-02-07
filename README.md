# KrushiMitra Backend API

This is the backend API for the KrushiMitra agricultural application. It provides RESTful endpoints for farmer profile management, activity tracking, mandi price information, and AI interactions.

## Table of Contents
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Environment Variables](#environment-variables)
- [Database Setup](#database-setup)
- [Multi-User Connection System](#multi-user-connection-system)
- [Structured Logging](#structured-logging)
- [IP Access Restriction](#ip-access-restriction)
- [Running the Server](#running-the-server)
- [Background Jobs](#background-jobs)
- [Testing](#testing)
- [AI Database Query Templates](#ai-database-query-templates)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites
- Node.js (v14 or higher)
- MongoDB Atlas account
- Firebase project for authentication
- ElevenLabs API key for text-to-speech
- Ollama with llama3 model for AI chat

### Installation
```bash
cd backend
npm install
```

## API Documentation

See [API_CONTRACT.md](API_CONTRACT.md) for detailed API documentation including endpoints, request/response formats, and examples.

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
# Read-only user for queries
DB_READER_USER=your_reader_username
DB_READER_PASS=your_reader_password

# Read-write user for modifications
DB_WRITER_USER=your_writer_username
DB_WRITER_PASS=your_writer_password

# Admin user for administrative tasks
DB_ADMIN_USER=your_admin_username
DB_ADMIN_PASS=your_admin_password

CLUSTER_HOST=your_mongodb_cluster_host
FIREBASE_SERVICE_ACCOUNT_KEY=path_to_your_firebase_service_account_key.json
PORT=3000
```

## Database Setup

### MongoDB Atlas Users

The application uses three MongoDB Atlas users with least privileges:

1. **Reader User** - Read-only access for queries
2. **Writer User** - Read-write access for data modifications
3. **Admin User** - Full administrative access for schema changes

See [MONGODB_ATLAS_USERS_SETUP.md](MONGODB_ATLAS_USERS_SETUP.md) for detailed instructions on creating these users.

### Database Initialization

1. Ensure MongoDB Atlas cluster is set up
2. Create a database named `KrushiMitraDB`
3. The application will automatically create required collections and indexes

## Multi-User Connection System

The backend implements a multi-user connection system that uses different MongoDB users based on the operation type:

- **Read Operations** use the reader user (least privileges)
- **Write Operations** use the writer user (read-write privileges)
- **Admin Operations** use the admin user (full privileges)

See [MULTI_USER_CONNECTIONS.md](MULTI_USER_CONNECTIONS.md) for detailed documentation on how this system works.

## Structured Logging

All database operations are logged with structured logging using Winston, including:

- Operation type and duration
- Farmer ID context (when applicable)
- Success/failure status
- Error details with stack traces
- Timestamps for all operations

Logs are written to:
- Console (with color coding)
- `logs/combined.log` (all logs)
- `logs/error.log` (error logs only)
- `logs/database.log` (database operations only)

See [STRUCTURED_LOGGING.md](STRUCTURED_LOGGING.md) for detailed documentation on the logging implementation.

## IP Access Restriction

For enhanced security, restrict MongoDB Atlas access to only authorized IP addresses:

- Remove the default `0.0.0.0/0` entry
- Add your development machine IP address
- Add your backend server IP address

See [MONGODB_IP_RESTRICTION_GUIDE.md](MONGODB_IP_RESTRICTION_GUIDE.md) for detailed instructions on configuring IP access restrictions.

## Running the Server

### Development Mode
```bash
npm run dev
```

### Production Mode
```bash
npm start
```

The server will start on port 3000 (or the port specified in the PORT environment variable).

## Background Jobs

The application includes two background jobs that run on a schedule:

### Mandi Price Fetcher
- **Schedule**: Every hour (0 minutes past every hour)
- **Purpose**: Fetches external mandi price data, computes golden chances, and creates alerts for farmers
- **Implementation**: [jobs/mandi-price-fetcher.js](jobs/mandi-price-fetcher.js)

### Scheme Watcher
- **Schedule**: Daily at 6:00 AM
- **Purpose**: Checks official feeds for new agricultural schemes and notifies eligible farmers
- **Implementation**: [jobs/scheme-watcher.js](jobs/scheme-watcher.js)

### Running Background Jobs

#### Start the job scheduler
```bash
npm run jobs
```

#### Run all jobs once (for testing)
```bash
npm run jobs:run-once
```

See [background-jobs.md](background-jobs.md) for detailed documentation on the job logic and database actions.

## Testing

### Test Plan Execution

The application includes a comprehensive test plan to verify all core functionality:

#### Test Cases:
1. **TC1**: Connect to DB → expected CONNECTED
2. **TC2**: Create collections if missing → all collections present
3. **TC3**: Insert sample farmer → can be retrieved by phone
4. **TC4**: Log activity → appears in activities query
5. **TC5**: Insert mandi price → mandiprices returns it as latest for crop+location
6. **TC6**: Golden chance logic test → alerts entry created

#### Running the Test Plan:
```bash
npm run test:plan
```

#### Testing Structured Logging:
```bash
npm run test:logging
```

See [TEST_PLAN.md](TEST_PLAN.md) for detailed documentation on the test plan and expected results.

## AI Database Query Templates

The AI system uses natural language-like query templates to interact with the database. These templates are defined in [ai-db-queries.json](ai-db-queries.json) and helper functions are available in [ai-query-helper.js](ai-query-helper.js).

## Farmer-Friendly AI Implementation

The application uses a specially trained LLaMA 3 model that provides farmer-friendly responses in multiple languages:

- Simple language explanations
- Culturally relevant advice
- Practical, actionable recommendations
- Support for small-scale farming practices

The AI prompting system is implemented in [farmer-llm-prompt.js](farmer-llm-prompt.js) which provides:

1. **Persona-based prompting** - The AI adopts a friendly farming expert persona
2. **Context-aware responses** - Personalized advice based on farmer profile
3. **Language-specific guidelines** - Culturally appropriate communication
4. **Practical focus** - Emphasis on low-cost, implementable solutions

For detailed information on training the AI to be farmer-friendly, see [FARMER_FRIENDLY_AI_TRAINING.md](../FARMER_FRIENDLY_AI_TRAINING.md).

### Available Query Templates:
1. **Get farmer by phone**: `FIND farmers WHERE phone = {phone}`
2. **Save activity**: `INSERT into activities { farmerId, activityType, description, date, aiSuggestions }`
3. **Get latest mandi prices**: `FIND mandiprices WHERE crop = {crop} AND location = {location} SORT date DESC LIMIT 5`
4. **Check golden chance**: `COMPUTE 7-day-average price and compare current price; if > 10% then true`
5. **Fetch farmer memory for chat**: `FIND aiinteractions WHERE farmerId = {farmerId} SORT timestamp DESC LIMIT 10`

See [ai-db-queries.json](ai-db-queries.json) for the complete list of templates with examples.

## API Endpoints

All endpoints require authentication via Firebase ID token in the Authorization header:
```
Authorization: Bearer <firebase_id_token>
```

### Farmer Profile Management
- `POST /farmers` - Create or update farmer profile
- `GET /farmers/:phone` - Fetch profile by phone

### Authentication
- `POST /auth/verify` - Verify Firebase ID token and get/create farmer record

### Activity Tracking
- `POST /activities` - Log activity for a farmer
- `GET /activities/:farmerId` - Fetch activity log

### Mandi Prices
- `POST /mandiprices/update` - Ingest mandi price data (bulk)
- `GET /mandiprices` - Get latest prices (with optional filtering)

### AI Chat
- `POST /ai/chat` - Send user query to AI

### Health Check
- `GET /health` - Check if the server is running

## Troubleshooting

### MongoDB Connection Issues

If you're experiencing MongoDB connection issues, refer to our comprehensive troubleshooting guide:

- [MONGODB_FIX_GUIDE.md](MONGODB_FIX_GUIDE.md) - Complete guide to fixing MongoDB connection issues
- [FIX_SUMMARY.md](FIX_SUMMARY.md) - Summary of identified issues and fix steps

### Common Scripts for Issue Resolution

1. Check your public IP for MongoDB Atlas whitelisting:
   ```bash
   node check-public-ip.js
   ```

2. Verify environment variables are loaded correctly:
   ```bash
   node verify-env-config.js
   ```

3. Test all MongoDB connections:
   ```bash
   node fix-mongodb-connection.js
   ```

4. Remove duplicate collections:
   ```bash
   node remove-duplicate-collection.js
   ```

### Ollama and TTS Service Issues

If you're experiencing issues with the Ollama or TTS services, please refer to our [Troubleshooting Guide](../TROUBLESHOOTING.md) for detailed solutions.

## Error Handling

All errors follow a consistent format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message"
  }
}
```

## Rate Limiting

- 1000 requests per hour per IP address
- 100 requests per minute per IP address

## CORS

Allowed origins:
- `https://app.krushimitra.example.com`

Allowed methods:
- GET, POST, PUT, DELETE, OPTIONS

Allowed headers:
- Authorization, Content-Type, X-Requested-With
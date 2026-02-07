# REST API Contract for KrushiMitra Application

## Overview
This document defines the REST API endpoints for the KrushiMitra agricultural application backend. The API provides functionality for farmer profile management, activity tracking, mandi price information, and AI interactions.

## Base URL
```
https://api.krushimitra.example.com/v1
```

## Authentication
All endpoints require authentication via Firebase ID token passed in the `Authorization` header:
```
Authorization: Bearer <firebase_id_token>
```

## Error Response Format
All error responses follow this format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message"
  }
}
```

---

## 1. Farmer Profile Management

### POST /farmers
Create or update a farmer profile.

**Request Body:**
```json
{
  "name": "string",
  "phone": "string",
  "language": "string",
  "location": "string",
  "crops": ["string"],
  "landSize": "number",
  "soilType": "string"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "_id": "string",
    "name": "string",
    "phone": "string",
    "language": "string",
    "location": "string",
    "crops": ["string"],
    "landSize": "number",
    "soilType": "string",
    "joinedAt": "date",
    "updatedAt": "date"
  }
}
```

**Example Request:**
```json
{
  "name": "Rajesh Kumar",
  "phone": "+919876543210",
  "language": "Hindi",
  "location": "Pune, Maharashtra",
  "crops": ["Wheat", "Soybean"],
  "landSize": 5.5,
  "soilType": "Black soil"
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "_id": "60f7b3b3e3b3a2a8c8f3b3a2",
    "name": "Rajesh Kumar",
    "phone": "+919876543210",
    "language": "Hindi",
    "location": "Pune, Maharashtra",
    "crops": ["Wheat", "Soybean"],
    "landSize": 5.5,
    "soilType": "Black soil",
    "joinedAt": "2023-07-21T10:30:00Z",
    "updatedAt": "2023-07-21T10:30:00Z"
  }
}
```

### GET /farmers/:phone
Fetch a farmer profile by phone number.

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "_id": "string",
    "name": "string",
    "phone": "string",
    "language": "string",
    "location": "string",
    "crops": ["string"],
    "landSize": "number",
    "soilType": "string",
    "joinedAt": "date",
    "updatedAt": "date"
  }
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "_id": "60f7b3b3e3b3a2a8c8f3b3a2",
    "name": "Rajesh Kumar",
    "phone": "+919876543210",
    "language": "Hindi",
    "location": "Pune, Maharashtra",
    "crops": ["Wheat", "Soybean"],
    "landSize": 5.5,
    "soilType": "Black soil",
    "joinedAt": "2023-07-21T10:30:00Z",
    "updatedAt": "2023-07-21T10:30:00Z"
  }
}
```

**Error Responses:**
- 404 Not Found: Farmer not found

---

## 2. Authentication

### POST /auth/verify
Accept Firebase ID token, verify it, and return farmer record or create a new one.

**Request Body:**
```json
{
  "idToken": "string"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "farmer": {
      "_id": "string",
      "name": "string",
      "phone": "string",
      "language": "string",
      "location": "string",
      "crops": ["string"],
      "landSize": "number",
      "soilType": "string",
      "joinedAt": "date",
      "updatedAt": "date"
    },
    "isNewUser": "boolean"
  }
}
```

**Example Request:**
```json
{
  "idToken": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjFmNzU1..."
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "farmer": {
      "_id": "60f7b3b3e3b3a2a8c8f3b3a2",
      "name": "Rajesh Kumar",
      "phone": "+919876543210",
      "language": "Hindi",
      "location": "Pune, Maharashtra",
      "crops": ["Wheat", "Soybean"],
      "landSize": 5.5,
      "soilType": "Black soil",
      "joinedAt": "2023-07-21T10:30:00Z",
      "updatedAt": "2023-07-21T10:30:00Z"
    },
    "isNewUser": false
  }
}
```

**Error Responses:**
- 400 Bad Request: Invalid ID token
- 401 Unauthorized: Token verification failed

---

## 3. Activity Tracking

### POST /activities
Log an activity for a farmer.

**Request Body:**
```json
{
  "farmerId": "string",
  "description": "string",
  "type": "string",
  "details": "object"
}
```

**Response (201 Created):**
```json
{
  "status": "success",
  "data": {
    "_id": "string",
    "farmerId": "string",
    "description": "string",
    "type": "string",
    "details": "object",
    "date": "date"
  }
}
```

**Example Request:**
```json
{
  "farmerId": "60f7b3b3e3b3a2a8c8f3b3a2",
  "description": "Applied fertilizer to wheat field",
  "type": "fertilization",
  "details": {
    "product": "NPK 20-20-20",
    "quantity": "50kg",
    "field": "North field"
  }
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "_id": "60f7b3b3e3b3a2a8c8f3b3a3",
    "farmerId": "60f7b3b3e3b3a2a8c8f3b3a2",
    "description": "Applied fertilizer to wheat field",
    "type": "fertilization",
    "details": {
      "product": "NPK 20-20-20",
      "quantity": "50kg",
      "field": "North field"
    },
    "date": "2023-07-21T14:45:00Z"
  }
}
```

### GET /activities/:farmerId
Fetch activity log for a farmer.

**Query Parameters:**
- `limit` (optional): Number of activities to return (default: 10)
- `offset` (optional): Number of activities to skip (default: 0)
- `type` (optional): Filter by activity type

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {
      "_id": "string",
      "farmerId": "string",
      "description": "string",
      "type": "string",
      "details": "object",
      "date": "date"
    }
  ],
  "pagination": {
    "limit": "number",
    "offset": "number",
    "total": "number"
  }
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": [
    {
      "_id": "60f7b3b3e3b3a2a8c8f3b3a3",
      "farmerId": "60f7b3b3e3b3a2a8c8f3b3a2",
      "description": "Applied fertilizer to wheat field",
      "type": "fertilization",
      "details": {
        "product": "NPK 20-20-20",
        "quantity": "50kg",
        "field": "North field"
      },
      "date": "2023-07-21T14:45:00Z"
    }
  ],
  "pagination": {
    "limit": 10,
    "offset": 0,
    "total": 1
  }
}
```

**Error Responses:**
- 404 Not Found: Farmer not found

---

## 4. Mandi Prices

### POST /mandiprices/update
Ingest mandi price data (bulk).

**Request Body:**
```json
{
  "prices": [
    {
      "crop": "string",
      "location": "string",
      "price": "number",
      "date": "date"
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "inserted": "number",
    "updated": "number"
  }
}
```

**Example Request:**
```json
{
  "prices": [
    {
      "crop": "Wheat",
      "location": "Pune",
      "price": 2400,
      "date": "2023-07-21T00:00:00Z"
    },
    {
      "crop": "Soybean",
      "location": "Pune",
      "price": 5200,
      "date": "2023-07-21T00:00:00Z"
    }
  ]
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "inserted": 2,
    "updated": 0
  }
}
```

### GET /mandiprices
Get latest prices with optional filtering.

**Query Parameters:**
- `crop` (optional): Filter by crop name
- `location` (optional): Filter by location

**Response (200 OK):**
```json
{
  "status": "success",
  "data": [
    {
      "_id": "string",
      "crop": "string",
      "location": "string",
      "price": "number",
      "date": "date"
    }
  ]
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": [
    {
      "_id": "60f7b3b3e3b3a2a8c8f3b3a4",
      "crop": "Wheat",
      "location": "Pune",
      "price": 2400,
      "date": "2023-07-21T00:00:00Z"
    },
    {
      "_id": "60f7b3b3e3b3a2a8c8f3b3a5",
      "crop": "Soybean",
      "location": "Pune",
      "price": 5200,
      "date": "2023-07-21T00:00:00Z"
    }
  ]
}
```

---

## 5. AI Chat

### POST /ai/chat
Send user query to AI; the backend queries DB for farmer memory and returns AI response and any automations triggered.

**Request Body:**
```json
{
  "farmerId": "string",
  "query": "string",
  "context": "object"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "response": "string",
    "automations": [
      {
        "type": "string",
        "triggered": "boolean",
        "details": "object"
      }
    ],
    "relatedData": "object"
  }
}
```

**Example Request:**
```json
{
  "farmerId": "60f7b3b3e3b3a2a8c8f3b3a2",
  "query": "What should I plant this season?",
  "context": {
    "location": "Pune, Maharashtra",
    "currentSeason": "Kharif"
  }
}
```

**Example Response:**
```json
{
  "status": "success",
  "data": {
    "response": "Based on your location in Pune and the current Kharif season, I recommend planting rice or maize. The soil conditions and rainfall patterns are favorable for these crops.",
    "automations": [
      {
        "type": "mandi_alert",
        "triggered": true,
        "details": {
          "crop": "Rice",
          "message": "Rice prices in Pune are currently favorable for planting"
        }
      }
    ],
    "relatedData": {
      "weatherForecast": {
        "nextWeekRainfall": "70mm",
        "temperatureRange": "28-35°C"
      }
    }
  }
}
```

## Status Codes
- 200 OK: Successful GET, PUT, PATCH, or DELETE request
- 201 Created: Successful POST request
- 400 Bad Request: Invalid request data
- 401 Unauthorized: Authentication failed
- 403 Forbidden: Access denied
- 404 Not Found: Resource not found
- 500 Internal Server Error: Server error

## Rate Limiting
- 1000 requests per hour per IP address
- 100 requests per minute per IP address

## CORS
- Allowed origins: `https://app.krushimitra.example.com`
- Allowed methods: GET, POST, PUT, DELETE, OPTIONS
- Allowed headers: Authorization, Content-Type, X-Requested-With
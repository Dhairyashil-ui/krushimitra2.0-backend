# Background Jobs for KrushiMitra Application

## 1. Mandi Price Fetcher

### Schedule
- Runs every hour (0 minutes past every hour)
- Cron expression: `0 * * * *`

### Logic Flow

1. **Fetch External Data**
   - Connect to external mandi price APIs or scrape government websites
   - Retrieve latest price data for all supported crops and locations
   - Parse and validate the data

2. **Insert into Database**
   - For each crop+location combination:
     - INSERT into mandiprices { crop, location, price, date }
   - Log any errors during insertion

3. **Compute Golden Chance**
   - For each crop+location in the new data:
     - FIND mandiprices WHERE crop = {crop} AND location = {location} SORT date DESC LIMIT 7
     - COMPUTE 7-day-average price
     - COMPARE current price with 7-day average
     - IF (current_price - average_price) / average_price > 0.10 THEN golden_chance = true

4. **Identify Affected Farmers**
   - FIND farmers WHERE crops CONTAINS {crop} AND location = {location} AND priceAlerts = true
   - Create a list of affected farmer IDs

5. **Insert Alerts**
   - For each affected farmer:
     - INSERT into alerts { farmerId, type: "price", message: "Price alert for {crop}", status: "active", createdAt: now, metadata: { crop, location, price, isGoldenChance } }
   - Log alert creation statistics

6. **Job Completion**
   - Record job execution details:
     - Timestamp
     - Number of prices fetched
     - Number of prices inserted
     - Number of golden chances detected
     - Number of alerts created
     - Any errors encountered

### Database Actions

1. **Read Operations:**
   - Read existing mandi prices to calculate 7-day averages
   - Read farmers who have opted for price alerts

2. **Write Operations:**
   - Insert new mandi price records
   - Insert alert records for farmers

### Example Alert Document Structure

```json
{
  "_id": "alert_1234567890",
  "farmerId": "farmer_9876543210",
  "type": "price",
  "message": "Great opportunity! Wheat prices in Pune are 12% above the 7-day average at ₹2400 per quintal",
  "status": "active",
  "createdAt": "2023-07-21T10:00:00Z",
  "metadata": {
    "crop": "Wheat",
    "location": "Pune",
    "price": 2400,
    "isGoldenChance": true,
    "previousAverage": 2142,
    "percentageIncrease": 12.04
  }
}
```

---

## 2. Scheme Watcher

### Schedule
- Runs daily at 6:00 AM (local time)
- Cron expression: `0 6 * * *`

### Logic Flow

1. **Fetch Official Feeds**
   - Connect to government agricultural scheme APIs
   - Scrape official websites for new schemes
   - Retrieve scheme data including:
     - Title
     - Description
     - Eligibility criteria
     - Start date
     - End date
     - Location (state/district/all)
     - Benefits
     - Application process

2. **Identify New Schemes**
   - For each scheme in the fetched data:
     - FIND schemes WHERE title = {title} AND startDate = {startDate}
     - IF no matching scheme found THEN mark as new scheme

3. **Validate and Enrich Data**
   - Validate scheme data format and required fields
   - Enrich with additional metadata if needed
   - Format dates consistently

4. **Insert New Schemes**
   - For each new scheme:
     - INSERT into schemes { title, description, eligibility, startDate, endDate, location, benefits, applicationProcess, createdAt, source }
   - Log insertion statistics

5. **Identify Eligible Farmers**
   - For each new scheme:
     - FIND farmers WHERE location MATCHES scheme.location AND (eligibility criteria)
     - Create a list of eligible farmer IDs

6. **Create Alerts**
   - For each eligible farmer:
     - INSERT into alerts { farmerId, type: "scheme", message: "New scheme available: {scheme.title}", status: "active", createdAt: now, metadata: { schemeId, title, benefits } }
   - Log alert creation statistics

7. **Job Completion**
   - Record job execution details:
     - Timestamp
     - Number of schemes fetched
     - Number of new schemes detected
     - Number of schemes inserted
     - Number of alerts created
     - Any errors encountered

### Database Actions

1. **Read Operations:**
   - Read existing schemes to identify new ones
   - Read farmers to find eligible recipients

2. **Write Operations:**
   - Insert new scheme records
   - Insert alert records for eligible farmers

### Example Alert Document Structure

```json
{
  "_id": "alert_0987654321",
  "farmerId": "farmer_1234567890",
  "type": "scheme",
  "message": "New scheme available: PM Kisan Samman Nidhi - Financial assistance for small and marginal farmers",
  "status": "active",
  "createdAt": "2023-07-21T06:00:00Z",
  "metadata": {
    "schemeId": "scheme_1122334455",
    "title": "PM Kisan Samman Nidhi",
    "benefits": "₹6000 per year transferred directly to farmer's bank account in 3 equal installments",
    "eligibility": "Small and marginal farmer families with cultivable land up to 2 hectares",
    "applicationProcess": "Register through Common Service Centers or online portal"
  }
}
```

---

## Job Success/Failure Reporting Format

### Success Report

```json
{
  "jobName": "Mandi Price Fetcher",
  "status": "SUCCESS",
  "startedAt": "2023-07-21T10:00:00Z",
  "completedAt": "2023-07-21T10:05:30Z",
  "durationSeconds": 330,
  "statistics": {
    "pricesFetched": 142,
    "pricesInserted": 142,
    "goldenChancesDetected": 7,
    "alertsCreated": 42,
    "farmersNotified": 38
  },
  "details": {
    "sources": [
      {
        "name": "Maharashtra Mandi Board",
        "pricesFetched": 75,
        "status": "SUCCESS"
      },
      {
        "name": "National Agriculture Market",
        "pricesFetched": 67,
        "status": "SUCCESS"
      }
    ]
  }
}
```

### Failure Report

```json
{
  "jobName": "Scheme Watcher",
  "status": "FAILURE",
  "startedAt": "2023-07-21T06:00:00Z",
  "completedAt": "2023-07-21T06:02:15Z",
  "durationSeconds": 135,
  "error": {
    "code": "EXTERNAL_API_ERROR",
    "message": "Failed to fetch data from Ministry of Agriculture API",
    "details": "Connection timeout after 30 seconds"
  },
  "statistics": {
    "schemesFetched": 0,
    "schemesInserted": 0,
    "alertsCreated": 0
  },
  "retry": {
    "attempt": 1,
    "maxAttempts": 3,
    "nextAttemptAt": "2023-07-21T06:15:00Z"
  }
}
```

### Partial Success Report

```json
{
  "jobName": "Mandi Price Fetcher",
  "status": "PARTIAL_SUCCESS",
  "startedAt": "2023-07-21T10:00:00Z",
  "completedAt": "2023-07-21T10:05:30Z",
  "durationSeconds": 330,
  "statistics": {
    "pricesFetched": 142,
    "pricesInserted": 128,
    "goldenChancesDetected": 6,
    "alertsCreated": 38,
    "farmersNotified": 35
  },
  "details": {
    "sources": [
      {
        "name": "Maharashtra Mandi Board",
        "pricesFetched": 75,
        "status": "SUCCESS"
      },
      {
        "name": "National Agriculture Market",
        "pricesFetched": 67,
        "status": "PARTIAL_SUCCESS",
        "error": "Failed to insert 14 prices due to database connection issues"
      }
    ]
  }
}
```
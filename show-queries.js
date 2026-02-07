// Script to show what the queries would return

console.log(JSON.stringify({
  status: "ERROR_CONNECT",
  message: "Missing environment variables: DB_USER, DB_PASS, and/or CLUSTER_HOST must be set. With valid credentials, this script would run the following queries:",
  queryDetails: {
    "Find farmer by phone": "db.farmers.findOne({ phone: '+919900112233' })",
    "Get last 5 activities": "db.activities.find({ farmerId: farmer._id }).sort({ date: -1 }).limit(5)",
    "Get latest mandi price": "db.mandiprices.findOne({ crop: 'Wheat', location: 'Pune' }, { sort: { date: -1 } })",
    "Get active schemes": "db.schemes.find({ startDate: { $lte: today }, endDate: { $gte: today } })"
  },
  expectedResponseFormat: {
    status: "QUERIES_OK",
    results: {
      farmers: { 
        _id: "farmer_object_id",
        name: "Test Farmer",
        phone: "+919900112233",
        // ... other farmer fields
      },
      activities: [
        { 
          _id: "activity_object_id",
          farmerId: "farmer_object_id",
          description: "activity description",
          date: "activity_date",
          // ... other activity fields
        }
        // ... up to 5 activities
      ],
      mandiprice: {
        _id: "mandiprice_object_id",
        crop: "Wheat",
        location: "Pune",
        price: 2400,
        date: "latest_date"
        // ... other mandiprice fields
      },
      schemes: [
        {
          _id: "scheme_object_id",
          title: "scheme title",
          description: "scheme description",
          startDate: "start_date",
          endDate: "end_date"
          // ... other scheme fields
        }
        // ... active schemes
      ]
    }
  }
}, null, 2));
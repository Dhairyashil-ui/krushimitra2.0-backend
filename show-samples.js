// Script to show what sample documents would be inserted

console.log(JSON.stringify({
  status: "ERROR_CONNECT",
  message: "Missing environment variables: DB_USER, DB_PASS, and/or CLUSTER_HOST must be set. With valid credentials, this script would insert the following sample documents:",
  sampleDocuments: {
    farmers: { 
      name: "Test Farmer", 
      phone: "+919900112233", 
      language: "Hindi", 
      location: "Pune, Maharashtra", 
      crops: ["Wheat"], 
      joinedAt: "current_date" 
    },
    activities: { 
      farmerId: "farmer_object_id", 
      description: "Installed app / first chat", 
      date: "current_date", 
      type: "app_install" 
    },
    mandiprices: { 
      crop: "Wheat", 
      location: "Pune", 
      price: 2400, 
      date: "today" 
    },
    schemes: { 
      title: "sample scheme title", 
      description: "sample scheme description", 
      startDate: "today", 
      endDate: "today + 30 days" 
    },
    aiinteractions: { 
      farmerId: "farmer_object_id", 
      question: "sample question", 
      response: "sample response", 
      timestamp: "current_date" 
    },
    crop_health: { 
      farmerId: "farmer_object_id", 
      detectedIssue: "sample issue", 
      solution: "sample solution", 
      confidence: "0.0-1.0", 
      imageUrl: "sample_url", 
      diagnosedAt: "current_date" 
    },
    alerts: { 
      farmerId: "farmer_object_id", 
      type: "weather", 
      message: "sample alert message", 
      status: "active", 
      createdAt: "current_date" 
    }
  }
}, null, 2));
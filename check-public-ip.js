const https = require('https');

function getPublicIP() {
  console.log("=== Getting Public IP Address ===\n");
  
  https.get('https://api.ipify.org', (res) => {
    let data = '';
    
    res.on('data', (chunk) => {
      data += chunk;
    });
    
    res.on('end', () => {
      console.log(`Your Public IP Address: ${data}`);
      console.log(`\nTo whitelist this IP in MongoDB Atlas:`);
      console.log(`1. Go to MongoDB Atlas Dashboard`);
      console.log(`2. Navigate to Network Access`);
      console.log(`3. Click "Add IP Address"`);
      console.log(`4. Enter this IP: ${data}`);
      console.log(`5. Click "Confirm"`);
    });
  }).on('error', (err) => {
    console.log(`Error getting IP: ${err.message}`);
    console.log(`\nAlternative method:`);
    console.log(`1. Visit https://www.whatismyip.com/ in your browser`);
    console.log(`2. Copy your public IP address`);
    console.log(`3. Add it to MongoDB Atlas Network Access`);
  });
}

getPublicIP();
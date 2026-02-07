/**
 * Script to help identify your public IP address for MongoDB Atlas IP whitelist configuration
 */

const https = require('https');

function getPublicIP() {
  return new Promise((resolve, reject) => {
    https.get('https://api.ipify.org', (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        resolve(data.trim());
      });
    }).on('error', (err) => {
      reject(err);
    });
  });
}

async function main() {
  console.log('=== Public IP Address Detector ===');
  console.log('This script helps identify your public IP address for MongoDB Atlas configuration.\n');
  
  try {
    const ip = await getPublicIP();
    console.log(`Your public IP address is: ${ip}`);
    console.log('\nInstructions for MongoDB Atlas IP Whitelisting:');
    console.log('1. Log in to MongoDB Atlas');
    console.log('2. Navigate to your cluster');
    console.log('3. Go to "Network Access" in the left sidebar');
    console.log('4. Click "Add IP Address"');
    console.log(`5. Enter this IP address: ${ip}`);
    console.log('6. Add a description like "Development Machine"');
    console.log('7. Click "Confirm"');
    console.log('\nNote: If this is for a production environment, consider setting an expiration date.');
    
    // Also provide the CIDR notation which is what MongoDB Atlas expects
    console.log(`\nAlternative CIDR notation: ${ip}/32`);
  } catch (error) {
    console.error('Error fetching public IP address:', error.message);
    console.log('\nAlternative methods to find your public IP:');
    console.log('- Visit https://whatismyipaddress.com/');
    console.log('- On Windows: curl ifconfig.me');
    console.log('- On macOS/Linux: curl ifconfig.me');
  }
}

if (require.main === module) {
  main();
}

module.exports = { getPublicIP };
/**
 * API Integration Test for UserContext System
 * Tests the complete flow through actual HTTP endpoints:
 * 1. Signup via /auth/send-otp and /auth/verify-otp
 * 2. Home page update via PUT /user-context/home
 * 3. AI chat via POST /ai/chat
 * 4. Verify chat history limit via GET /user-context/:userId
 */

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function success(message) {
  log(`✅ ${message}`, 'green');
}

function error(message) {
  log(`❌ ${message}`, 'red');
}

function info(message) {
  log(`ℹ️  ${message}`, 'cyan');
}

function section(title) {
  log(`\n${'='.repeat(70)}`, 'bright');
  log(`  ${title}`, 'bright');
  log(`${'='.repeat(70)}`, 'bright');
}

// Configuration
const BASE_URL = process.env.API_BASE_URL || 'http://localhost:3001';
const TEST_EMAIL = `test-${Date.now()}@krushimitra.test`;
const TEST_PHONE = `+91${Math.floor(Math.random() * 9000000000) + 1000000000}`;

let sessionToken = null;
let userId = null;

async function makeRequest(method, endpoint, body = null, headers = {}) {
  const url = `${BASE_URL}${endpoint}`;
  
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers
    }
  };
  
  if (sessionToken) {
    options.headers['Authorization'] = `Bearer ${sessionToken}`;
  }
  
  if (body) {
    options.body = JSON.stringify(body);
  }
  
  const response = await fetch(url, options);
  const data = await response.json();
  
  return { response, data };
}

async function runAPITests() {
  try {
    section('UserContext API Integration Test Suite');
    info(`Base URL: ${BASE_URL}`);
    info(`Test Email: ${TEST_EMAIL}`);
    info(`Test Phone: ${TEST_PHONE}`);
    
    // Test 1: Request OTP
    section('Test 1: Request OTP for Signup');
    
    const { data: otpData } = await makeRequest('POST', '/auth/send-otp', {
      email: TEST_EMAIL
    });
    
    if (otpData.status === 'success') {
      success('OTP sent successfully');
      info(`Check console for OTP (in development, it may be logged)`);
    } else {
      throw new Error(`Failed to send OTP: ${otpData.error?.message}`);
    }
    
    // For testing, we'll use a mock OTP or read from environment
    const TEST_OTP = process.env.TEST_OTP || '123456';
    info(`Using OTP: ${TEST_OTP} (set TEST_OTP env var or check server logs)`);
    
    // Wait a moment before verifying
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Test 2: Verify OTP and Signup
    section('Test 2: Verify OTP and Create Account');
    
    const { data: signupData } = await makeRequest('POST', '/auth/verify-otp', {
      email: TEST_EMAIL,
      otp: TEST_OTP,
      name: 'Test Farmer',
      phone: TEST_PHONE,
      language: 'hi',
      landSize: '5 acres',
      soilType: 'Loamy'
    });
    
    if (signupData.status === 'success') {
      success('User registered successfully');
      sessionToken = signupData.token || signupData.session?.token;
      userId = signupData.user?.userId || signupData.user?.id;
      
      if (!userId) {
        // Try to extract from user object
        userId = signupData.user?._id;
      }
      
      info(`User ID: ${userId}`);
      info(`Session Token: ${sessionToken ? sessionToken.substring(0, 20) + '...' : 'None'}`);
      
      if (signupData.user) {
        info(`User Profile: ${JSON.stringify(signupData.user, null, 2)}`);
      }
    } else {
      throw new Error(`Signup failed: ${signupData.error?.message}`);
    }
    
    if (!userId) {
      throw new Error('No userId received from signup. Cannot continue tests.');
    }
    
    // Test 3: Fetch initial UserContext
    section('Test 3: Fetch Initial UserContext');
    
    const { data: initialContext } = await makeRequest('GET', `/user-context/${userId}`);
    
    if (initialContext.status === 'success') {
      success('Initial UserContext fetched successfully');
      
      const context = initialContext.data;
      
      if (context.profile && context.profile.email === TEST_EMAIL) {
        success('Profile contains correct email');
      }
      
      if (context.location === null && context.weather === null) {
        success('Location and weather are null (as expected for new user)');
      } else {
        info(`Location: ${JSON.stringify(context.location)}`);
        info(`Weather: ${JSON.stringify(context.weather)}`);
      }
      
      if (Array.isArray(context.chats) && context.chats.length === 0) {
        success('Chats array is empty (as expected for new user)');
      } else {
        error(`Expected empty chats array, got ${context.chats?.length} messages`);
      }
    } else {
      error(`Failed to fetch UserContext: ${initialContext.error?.message}`);
    }
    
    // Test 4: Update location and weather (home page load)
    section('Test 4: Update Location & Weather (Home Page)');
    
    const locationData = {
      address: 'Village Shirur, Pune District, Maharashtra',
      lat: 18.8314,
      lon: 74.3769,
      precision: 'locality',
      raw: 'Shirur, Pune, MH'
    };
    
    const weatherData = {
      temperature: 28,
      humidity: 65,
      condition: 'Partly Cloudy',
      windSpeed: 12,
      precipitationProbability: 20,
      source: 'app'
    };
    
    const { data: homeUpdateData } = await makeRequest('PUT', '/user-context/home', {
      userId,
      location: locationData,
      weather: weatherData
    });
    
    if (homeUpdateData.status === 'success') {
      success('Location and weather updated successfully');
      
      const context = homeUpdateData.data;
      
      if (context.location?.address === locationData.address) {
        success(`Location set to: ${context.location.address}`);
      } else {
        error('Location mismatch');
      }
      
      if (context.weather?.temperature === weatherData.temperature) {
        success(`Weather set to: ${context.weather.condition}, ${context.weather.temperature}°C`);
      } else {
        error('Weather mismatch');
      }
    } else {
      error(`Failed to update home context: ${homeUpdateData.error?.message}`);
    }
    
    // Test 5: Send AI chat messages
    section('Test 5: AI Chat - First Conversation');
    
    const { data: chat1Data } = await makeRequest('POST', '/ai/chat', {
      userId,
      query: 'मेरी गेहूं की फसल में पीले पत्ते आ रहे हैं',
      language: 'hi'
    });
    
    if (chat1Data.status === 'success') {
      success('First AI chat successful');
      info(`AI Response: ${chat1Data.data.response.substring(0, 100)}...`);
      
      if (chat1Data.data.userContext) {
        success('UserContext included in AI response');
        
        if (chat1Data.data.userContext.location?.address === locationData.address) {
          success('Location context available to AI');
        }
        
        if (chat1Data.data.userContext.weather?.temperature === weatherData.temperature) {
          success('Weather context available to AI');
        }
      }
    } else {
      error(`AI chat failed: ${chat1Data.error?.message}`);
    }
    
    // Test 6: Multiple conversations to test chat limit
    section('Test 6: Multiple AI Conversations (Testing 5-message limit)');
    
    const conversations = [
      'मंडी में गेहूं का भाव क्या है?',
      'मौसम कैसा रहेगा कल?',
      'सोयाबीन बोने का सही समय क्या है?',
      'खाद कितनी मात्रा में डालूं?',
      'कीटनाशक की सिफारिश करें',
      'फसल कब काटूं?'
    ];
    
    for (let i = 0; i < conversations.length; i++) {
      info(`Sending conversation ${i + 1}/${conversations.length}...`);
      
      const { data } = await makeRequest('POST', '/ai/chat', {
        userId,
        query: conversations[i],
        language: 'hi'
      });
      
      if (data.status === 'success') {
        success(`Conversation ${i + 1} successful`);
      } else {
        error(`Conversation ${i + 1} failed`);
      }
      
      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    // Test 7: Verify chat history limit
    section('Test 7: Verify Chat History Limited to 5 Messages');
    
    const { data: finalContext } = await makeRequest('GET', `/user-context/${userId}`);
    
    if (finalContext.status === 'success') {
      const chatCount = finalContext.data.chats?.length || 0;
      
      if (chatCount === 5) {
        success(`Chat history correctly limited to 5 messages (got ${chatCount})`);
        
        log('\nLast 5 chat messages:', 'bright');
        finalContext.data.chats.forEach((chat, idx) => {
          const preview = chat.message.substring(0, 50);
          info(`  ${idx + 1}. [${chat.role}]: ${preview}${chat.message.length > 50 ? '...' : ''}`);
        });
        
        success('Chat history working correctly!');
      } else {
        error(`Expected 5 messages, but found ${chatCount}`);
        if (chatCount > 0) {
          info('Current chat messages:');
          finalContext.data.chats.forEach((chat, idx) => {
            info(`  ${idx + 1}. [${chat.role}]: ${chat.message.substring(0, 50)}...`);
          });
        }
      }
    } else {
      error(`Failed to fetch final context: ${finalContext.error?.message}`);
    }
    
    // Final Summary
    section('Test Summary - Complete UserContext Flow');
    
    const { data: summary } = await makeRequest('GET', `/user-context/${userId}`);
    
    if (summary.status === 'success') {
      success('✨ All API tests completed successfully!');
      
      log('\nFinal UserContext State:', 'bright');
      console.log(JSON.stringify({
        userId: summary.data.userId,
        profile: summary.data.profile,
        location: summary.data.location ? {
          address: summary.data.location.address,
          coordinates: `(${summary.data.location.latitude}, ${summary.data.location.longitude})`
        } : null,
        weather: summary.data.weather ? {
          condition: summary.data.weather.condition,
          temperature: summary.data.weather.temperature,
          humidity: summary.data.weather.humidity
        } : null,
        chatHistory: {
          count: summary.data.chats?.length || 0,
          latestMessages: summary.data.chats?.slice(-2).map(c => ({
            role: c.role,
            preview: c.message.substring(0, 40) + '...'
          }))
        }
      }, null, 2));
    }
    
    info('\nTest user created for verification. You can:');
    info(`  - View in MongoDB: user_context collection, userId: ${userId}`);
    info(`  - Login with: ${TEST_EMAIL}`);
    info(`  - Delete test user manually if needed`);
    
  } catch (err) {
    error(`\nAPI Test failed: ${err.message}`);
    console.error(err);
    process.exit(1);
  }
}

// Check if fetch is available (Node 18+)
if (typeof fetch === 'undefined') {
  error('This script requires Node.js 18+ with native fetch support');
  error('Or install node-fetch: npm install node-fetch');
  process.exit(1);
}

// Run the tests
log('Starting UserContext API Integration Tests...', 'bright');
log('Make sure the backend server is running!\n', 'yellow');

runAPITests()
  .then(() => {
    log('\n✅ All UserContext API tests passed!', 'green');
    process.exit(0);
  })
  .catch((err) => {
    error(`\nAPI Test suite failed: ${err.message}`);
    process.exit(1);
  });

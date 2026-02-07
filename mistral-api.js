const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Path to your Python AI script
const PYTHON_SCRIPT_PATH = path.join(__dirname, '../mistral_service.py');

/**
 * Execute Python script with input and return response
 */
function executePythonScript(input, action = 'advice') {
  return new Promise((resolve, reject) => {
    const python = spawn('python', [PYTHON_SCRIPT_PATH]);
    let output = '';
    let error = '';

    // Send input to Python script
    const requestData = JSON.stringify({ action, input });
    python.stdin.write(requestData);
    python.stdin.end();

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      error += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}: ${error}`));
      } else {
        try {
          const result = JSON.parse(output.trim());
          resolve(result);
        } catch (parseError) {
          resolve({ response: output.trim() });
        }
      }
    });

    python.on('error', (err) => {
      reject(new Error(`Failed to start Python process: ${err.message}`));
    });
  });
}

// Health check endpoint
app.get('/api/mistral/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'Mistral AI API',
    timestamp: new Date().toISOString()
  });
});

// Chat endpoint for general conversation
app.post('/api/mistral/chat', async (req, res) => {
  try {
    const { query, language = 'hi' } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    console.log(`[${new Date().toISOString()}] Chat request: ${query.substring(0, 100)}...`);

    const result = await executePythonScript(query, 'chat');
    
    res.json({
      response: result.response || result,
      language: language,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Chat endpoint error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
});

// Farming advice endpoint
app.post('/api/mistral/advice', async (req, res) => {
  try {
    const { query, language = 'hi' } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    console.log(`[${new Date().toISOString()}] Advice request: ${query.substring(0, 100)}...`);

    const result = await executePythonScript(query, 'advice');
    
    res.json({
      response: result.response || result,
      language: language,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Advice endpoint error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
});

// Crop diagnosis endpoint
app.post('/api/mistral/diagnose', async (req, res) => {
  try {
    const { symptoms, crop, language = 'hi' } = req.body;

    if (!symptoms || !crop) {
      return res.status(400).json({ error: 'Symptoms and crop are required' });
    }

    const query = `Crop: ${crop}, Symptoms: ${symptoms}`;
    console.log(`[${new Date().toISOString()}] Diagnosis request: ${query}`);

    const result = await executePythonScript(query, 'diagnose');
    
    res.json({
      response: result.response || result,
      crop: crop,
      symptoms: symptoms,
      language: language,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Diagnosis endpoint error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
});

// Streaming endpoint for real-time responses
app.post('/api/mistral/stream', async (req, res) => {
  try {
    const { query, language = 'hi' } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    res.writeHead(200, {
      'Content-Type': 'text/plain',
      'Transfer-Encoding': 'chunked',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Methods': 'POST, OPTIONS'
    });

    console.log(`[${new Date().toISOString()}] Stream request: ${query.substring(0, 100)}...`);

    const python = spawn('python', [PYTHON_SCRIPT_PATH]);
    
    // Send input to Python script
    const requestData = JSON.stringify({ action: 'chat', input: query });
    python.stdin.write(requestData);
    python.stdin.end();

    python.stdout.on('data', (data) => {
      res.write(data.toString());
    });

    python.stderr.on('data', (data) => {
      console.error('Python error:', data.toString());
    });

    python.on('close', (code) => {
      res.end();
    });

    python.on('error', (err) => {
      console.error('Python process error:', err);
      res.end();
    });

  } catch (error) {
    console.error('Stream endpoint error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`\n🚀 Mistral AI API Server running on port ${PORT}`);
  console.log(`📋 Health check: http://localhost:${PORT}/api/mistral/health`);
  console.log(`💬 Chat endpoint: http://localhost:${PORT}/api/mistral/chat`);
  console.log(`🌾 Advice endpoint: http://localhost:${PORT}/api/mistral/advice`);
  console.log(`🔍 Diagnosis endpoint: http://localhost:${PORT}/api/mistral/diagnose`);
  console.log(`📡 Stream endpoint: http://localhost:${PORT}/api/mistral/stream`);
  console.log(`\n✅ Ready to serve Mistral AI requests!\n`);
});

module.exports = app;
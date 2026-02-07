// Load environment variables
require('dotenv').config({ path: require('path').join(__dirname, '..', '.env') });

const express = require('express');
const cookieParser = require('cookie-parser');
let cors;
try {
  cors = require('cors');
} catch (e) {
  console.warn('cors package not found; using manual CORS headers. Install it with npm i cors for enhanced handling.');
}
const { connectToDatabase } = require('./db');
const { logger, logDBOperation, logDBError } = require('./logger');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const twilio = require('twilio'); // Twilio SDK
const { generateSpeech } = require('./tts');
const sgMail = require('@sendgrid/mail');
const { ObjectId } = require('mongodb');
const { OAuth2Client } = require('google-auth-library'); // Google OAuth verification
const Groq = require('groq-sdk');
const multer = require('multer');
const { identifyPlant } = require('./plantnet_client');
const { spawn } = require('child_process');




const {
  initUserContextCollection,
  ensureUserContext,
  updateLocationAndWeather,
  appendChatMessage,
  fetchUserContext
} = require('./user-context');

// Initialize Google OAuth client
const googleClient = new OAuth2Client(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  `${process.env.FRONTEND_URL || 'http://localhost:3000'}` // Default Redirect URI
);

// Initialize Groq AI Client
// Initialize Groq AI Client (Conditional)
let groq = null;
if (process.env.GROQ_API_KEY) {
  try {
    groq = new Groq({
      apiKey: process.env.GROQ_API_KEY
    });
    console.log("✅ Groq AI Client initialized");
  } catch (err) {
    console.error("⚠️ Failed to initialize Groq:", err.message);
  }
} else {
  console.warn("⚠️ GROQ_API_KEY is missing. AI treatment recommendations will be disabled.");
}



// Ensure the working directory is the backend folder even if started from project root
// This prevents relative path lookups (e.g. accidental attempts to access `./health`) from resolving against the root.
try {
  if (process.cwd() !== __dirname) {
    process.chdir(__dirname);
  }
} catch (e) {
  // If changing directory fails, log but continue; all path-sensitive code uses __dirname.
  console.error('Failed to set working directory to backend:', e.message);
}

const app = express();
const SESSION_COOKIE_NAME = 'session_token';
const SESSION_DEFAULT_DAYS = Number(process.env.SESSION_TTL_DAYS || 30);
const SESSION_SECURE = process.env.NODE_ENV === 'production';

// Global CORS (first middleware): either use cors package or manual implementation
if (cors) {
  app.use(
    cors({
      origin: '*',
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'ngrok-skip-browser-warning'],
      exposedHeaders: ['Content-Type'],
    })
  );
} else {
  app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, ngrok-skip-browser-warning');
    if (req.method === 'OPTIONS') {
      return res.sendStatus(204);
    }
    next();
  });
}

// Explicit preflight handler (ensures Authorization header allowed before auth middleware)
app.options('*', (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, ngrok-skip-browser-warning');
  return res.sendStatus(204);
});

const PORT = process.env.PORT || 3001;

app.use(express.json());
app.use(cookieParser());

// Database collections
let farmersCollection;
let activitiesCollection;
let mandipricesCollection;
let aiinteractionsCollection;
let usersCollection; // Add this line
let weatherDataCollection;
let sessionsCollection;
let userMemoriesCollection;
let otpCollection;
let userContextCollection;

const DEFAULT_MEMORY_SLICE = Number(process.env.AI_MEMORY_SLICE || 10);
const MAX_MEMORY_ENTRIES = Number(process.env.AI_MEMORY_LIMIT || 200);
const OTP_TTL_MINUTES = Number(process.env.OTP_TTL_MINUTES || 10);
const OTP_MAX_ATTEMPTS = Number(process.env.OTP_MAX_ATTEMPTS || 3);

async function persistOtpRecord(email, otp) {
  if (!otpCollection) {
    throw new Error('OTP collection not initialized');
  }
  const now = new Date();
  const expiresAt = new Date(now.getTime() + OTP_TTL_MINUTES * 60 * 1000);
  const otpHash = hashToken(`${email}:${otp}`);
  await otpCollection.updateOne(
    { email },
    {
      $set: {
        otpHash,
        expiresAt,
        attempts: 0,
        createdAt: now,
        lastAttemptAt: null
      }
    },
    { upsert: true }
  );
}

async function fetchOtpRecord(email) {
  if (!otpCollection) {
    return null;
  }
  return otpCollection.findOne({ email });
}

async function deleteOtpRecord(email) {
  if (otpCollection) {
    await otpCollection.deleteOne({ email });
  }
}

// Initialize database collections
async function initializeCollections() {
  const startTime = Date.now();
  try {
    const client = await connectToDatabase('admin');
    const db = client.db("KrushiMitraDB");

    farmersCollection = db.collection('farmers');
    activitiesCollection = db.collection('activities');
    mandipricesCollection = db.collection('mandiprices');
    aiinteractionsCollection = db.collection('aiinteractions');
    usersCollection = db.collection('users'); // Add this line
    sessionsCollection = db.collection('sessions');
    weatherDataCollection = db.collection('weather_data');
    userMemoriesCollection = db.collection('user_memories');
    otpCollection = db.collection('otp_codes');
    userContextCollection = await initUserContextCollection(db);

    await userMemoriesCollection.createIndex({ userKey: 1 }, { unique: true });
    await aiinteractionsCollection.createIndex({ userId: 1, timestamp: -1 });
    await otpCollection.createIndex({ email: 1 }, { unique: true });
    await otpCollection.createIndex({ expiresAt: 1 }, { expireAfterSeconds: 0 });

    const duration = Date.now() - startTime;
    logDBOperation('initializeCollections', {
      durationMs: duration,
      status: 'success',
      collections: ['farmers', 'activities', 'mandiprices', 'aiinteractions', 'weather_data', 'sessions', 'user_memories', 'otp_codes', 'user_context']
    });

    logger.info('Database collections initialized', { durationMs: duration });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('initializeCollections', error, { durationMs: duration });
    logger.error('Error initializing database collections', {
      error: error.message,
      durationMs: duration
    });
  }
}

function normalizeUserKey(userId, fallback) {
  if (userId instanceof ObjectId) {
    return userId.toString();
  }
  if (typeof userId === 'string' && userId.trim().length > 0) {
    return userId.trim();
  }
  if (fallback) {
    return String(fallback);
  }
  return null;
}

function toObjectId(value) {
  if (!value) {
    return null;
  }
  if (value instanceof ObjectId) {
    return value;
  }
  if (typeof value === 'string' && ObjectId.isValid(value)) {
    return new ObjectId(value);
  }
  return null;
}

async function ensureUserMemoryDocument(userKey) {
  if (!userMemoriesCollection) {
    throw new Error('User memories collection not initialized');
  }
  const normalizedKey = normalizeUserKey(userKey);
  if (!normalizedKey) {
    return null;
  }
  const now = new Date();
  await userMemoriesCollection.updateOne(
    { userKey: normalizedKey },
    {
      $setOnInsert: { userKey: normalizedKey, entries: [], createdAt: now },
      $set: { updatedAt: now }
    },
    { upsert: true }
  );
  return normalizedKey;
}

async function getUserMemoryEntries(userKey, limit = DEFAULT_MEMORY_SLICE) {
  const normalizedKey = await ensureUserMemoryDocument(userKey);
  if (!normalizedKey) {
    return [];
  }
  const doc = await userMemoriesCollection.findOne(
    { userKey: normalizedKey },
    { projection: { entries: { $slice: -Math.abs(limit) } } }
  );
  return doc?.entries || [];
}

async function appendUserMemoryEntries(userKey, newEntries = []) {
  const normalizedKey = await ensureUserMemoryDocument(userKey);
  if (!normalizedKey || !Array.isArray(newEntries) || newEntries.length === 0) {
    return;
  }
  await userMemoriesCollection.updateOne(
    { userKey: normalizedKey },
    {
      $push: {
        entries: {
          $each: newEntries.map(entry => ({
            role: entry.role || 'assistant',
            content: entry.content,
            metadata: entry.metadata || {},
            timestamp: entry.timestamp || new Date()
          })),
          $slice: -Math.abs(MAX_MEMORY_ENTRIES)
        }
      },
      $set: { updatedAt: new Date() }
    }
  );
}

async function findUserDocument(identifier, farmerId) {
  if (!usersCollection) {
    return null;
  }

  const tried = new Set();
  const candidates = [];
  const maybeId = toObjectId(identifier);
  if (maybeId) {
    candidates.push({ _id: maybeId });
  }

  const pushPhoneQueries = (value) => {
    if (!value) {
      return;
    }
    const trimmed = value.trim();
    if (!trimmed) {
      return;
    }
    const digits = trimmed.replace(/\D/g, '');
    candidates.push({ phone: trimmed });
    if (digits && digits !== trimmed) {
      candidates.push({ phone: digits });
    }
  };

  if (typeof identifier === 'string') {
    const trimmed = identifier.trim();
    if (trimmed.includes('@')) {
      candidates.push({ email: trimmed.toLowerCase() });
    }
    pushPhoneQueries(trimmed);
  }

  if (typeof farmerId === 'string') {
    pushPhoneQueries(farmerId);
  }

  for (const query of candidates) {
    const key = JSON.stringify(query);
    if (tried.has(key)) {
      continue;
    }
    tried.add(key);
    try {
      const user = await usersCollection.findOne(query);
      if (user) {
        return user;
      }
    } catch (error) {
      logger.warn('User lookup failed', { query, error: error.message });
    }
  }

  return null;
}

function hashToken(token) {
  return crypto.createHash('sha256').update(token).digest('hex');
}

function extractSessionToken(req) {
  const authHeader = req.headers?.authorization || req.headers?.Authorization;
  if (authHeader && typeof authHeader === 'string') {
    const prefix = authHeader.trim().slice(0, 6).toLowerCase();
    if (prefix === 'bearer') {
      return authHeader.trim().slice(7);
    }
  }
  if (req.cookies && req.cookies[SESSION_COOKIE_NAME]) {
    return req.cookies[SESSION_COOKIE_NAME];
  }
  return null;
}

function setSessionCookie(res, token, expiresAt) {
  res.cookie(SESSION_COOKIE_NAME, token, {
    httpOnly: true,
    secure: SESSION_SECURE,
    sameSite: 'lax',
    expires: expiresAt,
    path: '/'
  });
}

function clearSessionCookie(res) {
  res.clearCookie(SESSION_COOKIE_NAME, { path: '/' });
}

async function createSession(userId, req) {
  if (!sessionsCollection) {
    throw new Error('Sessions collection not initialized');
  }

  const now = new Date();
  const expiresAt = new Date(now.getTime() + SESSION_DEFAULT_DAYS * 24 * 60 * 60 * 1000);
  const token = crypto.randomBytes(48).toString('hex');
  const tokenHash = hashToken(token);

  await sessionsCollection.insertOne({
    userId: typeof userId === 'string' ? new ObjectId(userId) : userId,
    tokenHash,
    userAgent: req.get('user-agent') || 'unknown',
    createdAt: now,
    lastUsedAt: now,
    expiresAt,
    revoked: false
  });

  return { token, expiresAt };
}

async function findActiveSession(token) {
  if (!token || !sessionsCollection) {
    return null;
  }

  const tokenHash = hashToken(token);
  const session = await sessionsCollection.findOne({ tokenHash, revoked: { $ne: true } });
  if (!session) {
    return null;
  }

  if (session.expiresAt && session.expiresAt < new Date()) {
    await sessionsCollection.updateOne(
      { _id: session._id },
      { $set: { revoked: true, revokedAt: new Date() } }
    );
    return null;
  }

  await sessionsCollection.updateOne(
    { _id: session._id },
    { $set: { lastUsedAt: new Date() } }
  );

  return session;
}

async function revokeSessionByToken(token) {
  if (!token || !sessionsCollection) {
    return false;
  }

  const tokenHash = hashToken(token);
  const result = await sessionsCollection.updateOne(
    { tokenHash, revoked: { $ne: true } },
    { $set: { revoked: true, revokedAt: new Date() } }
  );
  return result.modifiedCount > 0;
}

async function resolveSessionUser(token) {
  const session = await findActiveSession(token);
  if (!session) {
    return null;
  }

  const user = await usersCollection.findOne({ _id: new ObjectId(session.userId) });
  if (!user) {
    return null;
  }

  return { session, user };
}

function formatUserResponse(user) {
  if (!user) {
    return null;
  }

  return {
    id: user._id?.toString(),
    email: user.email,
    name: user.name,
    phone: user.phone || null,
    photo: user.photo || null,
    profile: user.profile || {},
    preferredLanguage: user.preferredLanguage || user.profile?.language || null,
    createdAt: user.createdAt,
    lastLogin: user.lastLogin
  };
}

function formatUserContextResponse(doc) {
  if (!doc) {
    return null;
  }
  const normalizedUserId = doc.userId instanceof ObjectId ? doc.userId.toString() : doc.userId;
  return {
    userId: normalizedUserId,
    userData: doc.userData || {},
    query: doc.query || []
  };
}

// Helper function to verify Firebase token (mock implementation)
async function verifyFirebaseToken(idToken) {
  // In a real implementation, this would call Firebase Admin SDK
  // For now, we'll just check if it's a non-empty string
  return idToken && typeof idToken === 'string' && idToken.length > 0;
}

// Middleware to authenticate requests
function setCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

// Authentication disabled: middleware now permits all requests.
async function authenticate(req, res, next) {
  setCorsHeaders(res); // keep CORS headers consistent
  req.userId = 'anonymous';
  return next();
}
// TTS route - supports GET with query params
app.get('/tts', async (req, res) => {
  try {
    const { text, lang = 'hi' } = req.query; // default to Hindi
    if (!text) {
      return res.status(400).json({
        error: { code: 'VALIDATION_ERROR', message: 'text query parameter is required' }
      });
    }

    const tmpPath = path.join(__dirname, `speech-${Date.now()}.mp3`);
    const saved = await generateSpeech(text, lang, { outputFile: tmpPath });

    if (!fs.existsSync(saved)) {
      return res.status(500).json({
        error: { code: 'TTS_ERROR', message: 'Speech file not found after generation' }
      });
    }

    // Set headers for proper audio playback including ngrok compatibility
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Disposition', 'inline; filename="speech.mp3"');
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Access-Control-Allow-Origin', '*');

    const stream = fs.createReadStream(saved);
    stream.pipe(res);
    stream.on('close', () => {
      fs.promises.unlink(saved).catch(() => { }); // cleanup temp file
    });
  } catch (error) {
    logger.error('TTS generation failed', { error: error.message });
    res.status(500).json({
      error: { code: 'TTS_ERROR', message: 'Failed to generate speech' }
    });
  }
});

// 1. Farmer Profile Management

// POST /farmers - Create or update farmer profile
app.post('/farmers', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { name, phone, language, location, crops, landSize, soilType } = req.body;

    // Validation
    if (!name || !phone) {
      const duration = Date.now() - startTime;
      logger.warn('Farmer profile validation failed - missing required fields', {
        farmerId: req.body.phone,
        missingFields: [!name ? 'name' : null, !phone ? 'phone' : null].filter(Boolean),
        durationMs: duration
      });

      return res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Name and phone are required'
        }
      });
    }

    const now = new Date();

    // Use MongoDB to find or create farmer
    const farmer = await farmersCollection.findOneAndUpdate(
      { phone: phone },
      {
        $set: {
          name,
          language,
          location,
          crops,
          landSize,
          soilType,
          updatedAt: now
        },
        $setOnInsert: {
          joinedAt: now
        }
      },
      {
        upsert: true,
        returnDocument: 'after'
      }
    );

    const duration = Date.now() - startTime;
    logDBOperation('upsertFarmer', {
      farmerId: phone,
      durationMs: duration,
      status: 'success'
    });

    logger.info('Farmer profile created/updated successfully', {
      farmerId: phone,
      durationMs: duration
    });

    res.status(200).json({
      status: 'success',
      data: farmer
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('upsertFarmer', error, {
      farmerId: req.body?.phone,
      durationMs: duration
    });
    logger.error('Error creating/updating farmer profile', {
      error: error.message,
      farmerId: req.body?.phone,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error creating/updating farmer profile'
      }
    });
  }
});

// GET /farmers/:phone - Fetch profile by phone
app.get('/farmers/:phone', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { phone } = req.params;

    // Use MongoDB to find farmer
    const farmer = await farmersCollection.findOne({ phone: phone });

    const duration = Date.now() - startTime;
    if (farmer) {
      logDBOperation('findFarmer', {
        farmerId: phone,
        durationMs: duration,
        status: 'success'
      });

      logger.info('Farmer profile retrieved successfully', {
        farmerId: phone,
        durationMs: duration
      });
    } else {
      logDBOperation('findFarmer', {
        farmerId: phone,
        durationMs: duration,
        status: 'not_found'
      });

      logger.warn('Farmer not found', {
        farmerId: phone,
        durationMs: duration
      });
    }

    if (!farmer) {
      return res.status(404).json({
        error: {
          code: 'FARMER_NOT_FOUND',
          message: 'Farmer not found'
        }
      });
    }

    res.status(200).json({
      status: 'success',
      data: farmer
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('findFarmer', error, {
      farmerId: req.params?.phone,
      durationMs: duration
    });
    logger.error('Error fetching farmer profile', {
      error: error.message,
      farmerId: req.params?.phone,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error fetching farmer profile'
      }
    });
  }
});

// 2. Authentication

// Google OAuth Authentication Endpoints

// POST /auth/google - Google OAuth login/signup
app.post('/auth/google', async (req, res) => {
  const startTime = Date.now();
  try {
    // Determine if we received a code (web flow) or idToken (mobile/other flow)
    const { code, idToken, redirect_uri } = req.body;

    if (!code && !idToken) {
      return res.status(400).json({
        error: { code: 'VALIDATION_ERROR', message: 'Auth code or ID token is required' }
      });
    }

    let payload;

    // SCENARIO 1: We received an Authorization Code (Web Flow)
    if (code) {
      try {
        // Exchange code for tokens (access_token, id_token, refresh_token)
        // We MUST pass the same redirect_uri that was used on the frontend
        // This is critical for security: backend must match frontend's redirect URI
        const { tokens } = await googleClient.getToken({
          code,
          redirect_uri, // Use the redirect_uri passed from frontend
          client_id: process.env.GOOGLE_CLIENT_ID,
          client_secret: process.env.GOOGLE_CLIENT_SECRET
        });

        // Determine the ID Token from the response
        const idTokenFromCode = tokens.id_token;
        if (!idTokenFromCode) {
          throw new Error('No ID token returned from code exchange');
        }

        // Verify the ID token
        const ticket = await googleClient.verifyIdToken({
          idToken: idTokenFromCode,
          audience: process.env.GOOGLE_CLIENT_ID,
        });
        payload = ticket.getPayload();
      } catch (exchangeError) {
        logger.error('Google code exchange failed', { error: exchangeError.message });
        return res.status(401).json({
          error: { code: 'INVALID_GRANT', message: 'Failed to exchange authorization code' }
        });
      }
    }
    // SCENARIO 2: We received an ID Token directly (Mobile/Implicit Flow)
    else {
      try {
        const ticket = await googleClient.verifyIdToken({
          idToken: idToken,
          audience: process.env.GOOGLE_CLIENT_ID,
        });
        payload = ticket.getPayload();
      } catch (verifyError) {
        logger.error('Google token verification failed', { error: verifyError.message });
        return res.status(401).json({
          error: { code: 'INVALID_TOKEN', message: 'Invalid Google ID token' }
        });
      }
    }

    const { sub: googleId, email, name, picture: photo } = payload;

    if (!email) {
      return res.status(400).json({
        error: { code: 'VALIDATION_ERROR', message: 'Email not found in Google token' }
      });
    }

    const now = new Date();

    // Check if user exists
    let existingUser = await usersCollection.findOne({ email });

    if (existingUser) {
      await usersCollection.updateOne(
        { email },
        {
          $set: {
            lastLogin: now,
            name,
            photo,
            googleId
          }
        }
      );
      existingUser = {
        ...existingUser,
        name,
        photo,
        googleId,
        lastLogin: now
      };

      await ensureUserContext(existingUser._id, {
        name: existingUser.name,
        email,
        phone: existingUser.phone || null,
        language: existingUser.preferredLanguage || existingUser.profile?.language || null
      });

      const session = await createSession(existingUser._id, req);
      setSessionCookie(res, session.token, session.expiresAt);

      logger.info('User logged in via Google', { userId: existingUser._id.toString(), email });

      return res.json({
        status: 'success',
        data: {
          user: formatUserResponse(existingUser),
          token: session.token,
          session: {
            token: session.token,
            expiresAt: session.expiresAt
          }
        }
      });
    }

    // Create new user
    const newUser = {
      googleId,
      email,
      name,
      photo,
      createdAt: now,
      lastLogin: now,
      profile: {}
    };

    const result = await usersCollection.insertOne(newUser);
    newUser._id = result.insertedId;

    await ensureUserContext(result.insertedId, {
      name: newUser.name,
      email: newUser.email,
      phone: newUser.phone || null,
      language: newUser.profile?.language || newUser.preferredLanguage || null
    });

    const session = await createSession(result.insertedId, req);
    setSessionCookie(res, session.token, session.expiresAt);

    const duration = Date.now() - startTime;
    logger.info('New user registered via Google', {
      userId: result.insertedId.toString(),
      email,
      durationMs: duration
    });

    res.status(201).json({
      status: 'success',
      data: {
        user: formatUserResponse(newUser),
        token: session.token,
        session: {
          token: session.token,
          expiresAt: session.expiresAt
        }
      }
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logger.error('Error in Google auth', { error: error.message, durationMs: duration });
    res.status(500).json({
      error: { code: 'SERVER_ERROR', message: 'Error processing Google authentication' }
    });
  }
});



// GET /auth/user/:userId - Get user profile
app.get('/auth/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const { ObjectId } = require('mongodb');

    const user = await usersCollection.findOne({ _id: new ObjectId(userId) });

    if (!user) {
      return res.status(404).json({
        error: { code: 'NOT_FOUND', message: 'User not found' }
      });
    }

    res.json({
      status: 'success',
      data: {
        user: {
          id: user._id.toString(),
          email: user.email,
          name: user.name,
          phone: user.phone || null,
          photo: user.photo,
          profile: user.profile || {},
          createdAt: user.createdAt,
          lastLogin: user.lastLogin
        }
      }
    });
  } catch (error) {
    logger.error('Error fetching user', { error: error.message });
    res.status(500).json({
      error: { code: 'SERVER_ERROR', message: 'Error fetching user profile' }
    });
  }
});

// PUT /auth/user/:userId - Update user profile
app.put('/auth/user/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const { profile } = req.body;
    const { ObjectId } = require('mongodb');

    await usersCollection.updateOne(
      { _id: new ObjectId(userId) },
      { $set: { profile, updatedAt: new Date() } }
    );

    res.json({ status: 'success', message: 'Profile updated' });
  } catch (error) {
    logger.error('Error updating user profile', { error: error.message });
    res.status(500).json({
      error: { code: 'SERVER_ERROR', message: 'Error updating profile' }
    });
  }
});

// GET /user-context/:userId - Fetch full UserContext (userData + query)
app.get('/user-context/:userId', authenticate, async (req, res) => {
  try {
    const { userId } = req.params;
    const contextDoc = await fetchUserContext(userId);
    if (!contextDoc) {
      return res.status(404).json({
        error: { code: 'USER_CONTEXT_NOT_FOUND', message: 'User context not found' }
      });
    }
    return res.json({
      status: 'success',
      data: formatUserContextResponse(contextDoc) // Returns: { userId, userData, query }
    });
  } catch (error) {
    logger.error('Error fetching user context', { error: error.message });
    res.status(500).json({
      error: { code: 'SERVER_ERROR', message: 'Error fetching user context' }
    });
  }
});

// PUT /user-context/home - Update userData.location and userData.weather
app.put('/user-context/home', authenticate, async (req, res) => {
  try {
    const { userId, location, weather, profile } = req.body || {};
    if (!userId) {
      return res.status(400).json({
        error: { code: 'VALIDATION_ERROR', message: 'userId is required' }
      });
    }

    // Updates only userData.location and userData.weather, preserves query
    const updatedDoc = await updateLocationAndWeather(userId, { profile, location, weather });
    if (!updatedDoc) {
      return res.status(404).json({
        error: { code: 'USER_CONTEXT_NOT_FOUND', message: 'User context not found' }
      });
    }
    return res.json({
      status: 'success',
      data: formatUserContextResponse(updatedDoc)
    });
  } catch (error) {
    logger.error('Error updating home context', { error: error.message });
    res.status(500).json({
      error: { code: 'SERVER_ERROR', message: 'Error updating user context' }
    });
  }
});

// Email/OTP System (SendGrid)

// In-memory OTP storage (in production, use Redis or database)

// Configure SendGrid if API key is available
const SENDGRID_API_KEY = process.env.SENDGRID_API_KEY;
const SENDGRID_FROM = process.env.SENDGRID_FROM;

if (SENDGRID_API_KEY) {
  try {
    sgMail.setApiKey(SENDGRID_API_KEY);
    logger.info('SendGrid configured for OTP emails', {
      fromConfigured: Boolean(SENDGRID_FROM)
    });
  } catch (e) {
    logger.error('Failed to configure SendGrid', { error: e?.message || String(e) });
  }
} else {
  logger.warn('SENDGRID_API_KEY not configured - OTP email sending disabled');
}

async function sendOtpEmail(to, otp) {
  if (!SENDGRID_API_KEY || !SENDGRID_FROM) {
    const reason = !SENDGRID_API_KEY ? 'SENDGRID_API_KEY missing' : 'SENDGRID_FROM missing';
    throw new Error(`Email not configured: ${reason}`);
  }

  const subject = 'KrushiMitra - Your OTP Code';
  const text = `Your KrushiMitra OTP is ${otp}. It is valid for 10 minutes.`;
  const html = `
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;">
      <div style="background-color: #4CAF50; padding: 20px; text-align: center; border-radius: 10px 10px 0 0;">
        <h1 style="color: white; margin: 0;">KrushiMitra</h1>
        <p style="color: #E8F5E9; margin: 5px 0;">AI-Powered Farming Assistant</p>
      </div>
      <div style="background-color: white; padding: 30px; border-radius: 0 0 10px 10px;">
        <h2 style="color: #2E7D32; margin-top: 0;">Your OTP Code</h2>
        <p style="color: #666; font-size: 16px;">Your One-Time Password (OTP) for KrushiMitra login/signup is:</p>
        <div style="background-color: #F1F8E9; padding: 20px; margin: 20px 0; text-align: center; border-radius: 8px; border-left: 4px solid #4CAF50;">
          <h1 style="color: #2E7D32; margin: 0; font-size: 36px; letter-spacing: 8px;">${otp}</h1>
        </div>
        <p style="color: #666; font-size: 14px;">This OTP is valid for 10 minutes.</p>
        <p style="color: #666; font-size: 14px;">If you didn't request this OTP, please ignore this email.</p>
        <hr style="border: none; border-top: 1px solid #E0E0E0; margin: 20px 0;">
        <p style="color: #999; font-size: 12px; text-align: center;">© 2025 KrushiMitra. All rights reserved.</p>
      </div>
    </div>
  `;

  const msg = {
    to,
    from: SENDGRID_FROM,
    subject,
    text,
    html
  };

  try {
    console.log('📨 Sending email to:', to);
    console.log('SG API Key exists:', Boolean(SENDGRID_API_KEY));
    console.log('From:', SENDGRID_FROM);
    await sgMail.send(msg);
    console.log('✅ SendGrid accepted request for:', to);
    return true;
  } catch (err) {
    const sgError = {
      message: err?.message || String(err),
      code: err?.code,
      responseStatus: err?.response?.statusCode,
      responseBody: err?.response?.body
    };
    console.error('❌ SENDGRID EMAIL ERROR:', err?.response?.body || err);
    logger.error('SendGrid send failed', sgError);
    throw new Error(sgError.message || 'Failed to send OTP email');
  }
}

// GET /auth/email-status - SendGrid diagnostics
app.get('/auth/email-status', (req, res) => {
  res.json({
    status: 'success',
    data: {
      provider: 'SendGrid',
      hasApiKey: Boolean(SENDGRID_API_KEY),
      hasFrom: Boolean(SENDGRID_FROM),
      configured: Boolean(SENDGRID_API_KEY && SENDGRID_FROM)
    }
  });
});

// POST /auth/send-otp - Send OTP to email
app.post('/auth/send-otp', async (req, res) => {
  try {
    console.log('🔍 /auth/send-otp hit');
    console.log('Body:', req.body);
    console.log('SENDGRID_API_KEY:', process.env.SENDGRID_API_KEY ? 'OK' : 'MISSING');
    console.log('SENDGRID_FROM:', process.env.SENDGRID_FROM);
    const { email } = req.body;

    if (!email || !email.includes('@')) {
      return res.status(400).json({
        error: { code: 'VALIDATION_ERROR', message: 'Valid email is required' }
      });
    }

    // Check SendGrid configuration
    if (!SENDGRID_API_KEY || !SENDGRID_FROM) {
      logger.error('Email service not configured', {
        hasApiKey: Boolean(SENDGRID_API_KEY),
        hasFrom: Boolean(SENDGRID_FROM)
      });
      return res.status(503).json({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Email service is not configured. Please contact administrator.'
        }
      });
    }

    // Generate 6-digit OTP
    const otp = Math.floor(100000 + Math.random() * 900000).toString();

    // Persist OTP with expiration
    await persistOtpRecord(email, otp);

    // Send OTP email via SendGrid
    await sendOtpEmail(email, otp);

    logger.info('OTP sent successfully', { email });

    res.json({
      status: 'success',
      message: 'OTP sent to your email'
    });

  } catch (error) {
    console.error('🔥 OTP ERROR:', error);
    logger.error('Error sending OTP', {
      error: error.message,
      stack: error.stack,
      code: error.code
    });
    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error sending OTP. Please try again.',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      }
    });
  }
});

// POST /auth/verify-otp - Verify OTP and login/signup
app.post('/auth/verify-otp', async (req, res) => {
  try {
    const { email, otp, name, landSize, soilType, phone, language, validateOnly } = req.body;
    const sanitizedPhone = typeof phone === 'string' ? phone.trim() : '';
    const sanitizedName = typeof name === 'string' ? name.trim() : '';
    const preferredLanguage = typeof language === 'string' && language.trim().length > 0 ? language.trim() : null;

    if (!email || !otp) {
      return res.status(400).json({
        error: { code: 'VALIDATION_ERROR', message: 'Email and OTP are required' }
      });
    }

    // Check if OTP exists
    const otpRecord = await fetchOtpRecord(email);

    if (!otpRecord) {
      return res.status(400).json({
        error: { code: 'OTP_NOT_FOUND', message: 'OTP not found. Please request a new one.' }
      });
    }

    const nowTs = new Date();
    if (otpRecord.expiresAt && nowTs > otpRecord.expiresAt) {
      await deleteOtpRecord(email);
      return res.status(400).json({
        error: { code: 'OTP_EXPIRED', message: 'OTP has expired. Please request a new one.' }
      });
    }

    const attempts = otpRecord.attempts || 0;
    if (attempts >= OTP_MAX_ATTEMPTS) {
      await deleteOtpRecord(email);
      return res.status(400).json({
        error: { code: 'TOO_MANY_ATTEMPTS', message: 'Too many failed attempts. Please request a new OTP.' }
      });
    }

    const providedHash = hashToken(`${email}:${otp}`);
    if (otpRecord.otpHash !== providedHash) {
      const nextAttempts = attempts + 1;
      if (nextAttempts >= OTP_MAX_ATTEMPTS) {
        await deleteOtpRecord(email);
        return res.status(400).json({
          error: { code: 'TOO_MANY_ATTEMPTS', message: 'Too many failed attempts. Please request a new OTP.' }
        });
      }
      await otpCollection.updateOne(
        { email },
        { $set: { attempts: nextAttempts, lastAttemptAt: nowTs } }
      );
      return res.status(400).json({
        error: { code: 'INVALID_OTP', message: `Invalid OTP. ${Math.max(OTP_MAX_ATTEMPTS - nextAttempts, 0)} attempts remaining.` }
      });
    }

    let existingUser = await usersCollection.findOne({ email });

    if (validateOnly) {
      await otpCollection.updateOne(
        { email },
        { $set: { lastAttemptAt: nowTs } }
      );
      return res.json({
        status: 'success',
        data: {
          valid: true,
          existingUser: Boolean(existingUser),
          requiresPhone: !existingUser
        }
      });
    }

    // OTP verified - clear it
    await deleteOtpRecord(email);

    // Check if user exists
    if (existingUser) {
      const now = new Date();
      const updateFields = { lastLogin: now };
      if (sanitizedPhone) {
        updateFields.phone = sanitizedPhone;
        updateFields['profile.phone'] = sanitizedPhone;
      }
      if (preferredLanguage) {
        updateFields.preferredLanguage = preferredLanguage;
        updateFields['profile.language'] = preferredLanguage;
      }
      if (sanitizedName) {
        updateFields.name = sanitizedName;
        updateFields['profile.name'] = sanitizedName;
      }
      if (landSize) {
        updateFields['profile.landSize'] = landSize?.toString() || '';
      }
      if (soilType) {
        updateFields['profile.soilType'] = soilType;
      }
      await usersCollection.updateOne(
        { email },
        { $set: updateFields }
      );
      existingUser = {
        ...existingUser,
        ...('phone' in updateFields ? { phone: sanitizedPhone } : {}),
        ...(sanitizedName ? { name: sanitizedName } : {}),
        lastLogin: now,
        profile: {
          ...(existingUser.profile || {}),
          ...(sanitizedPhone ? { phone: sanitizedPhone } : {}),
          ...(sanitizedName ? { name: sanitizedName } : {}),
          ...(preferredLanguage ? { language: preferredLanguage } : {}),
          ...(landSize ? { landSize: landSize?.toString() || '' } : {}),
          ...(soilType ? { soilType } : {})
        },
        preferredLanguage: preferredLanguage || existingUser.preferredLanguage || existingUser.profile?.language
      };

      try {
        await ensureUserContext(existingUser._id, {
          name: existingUser.name || sanitizedName || email.split('@')[0],
          email,
          phone: sanitizedPhone || existingUser.phone || null,
          language: existingUser.preferredLanguage || preferredLanguage || existingUser.profile?.language || null
        });

        await ensureUserMemoryDocument(existingUser._id?.toString() || email);
      } catch (contextError) {
        logger.warn('User context init skipped for existing user', {
          userId: existingUser._id?.toString(),
          error: contextError.message
        });
      }

      const session = await createSession(existingUser._id, req);
      setSessionCookie(res, session.token, session.expiresAt);

      logger.info('User logged in via email OTP', { userId: existingUser._id.toString(), email });

      return res.json({
        status: 'success',
        message: 'Login successful',
        user: formatUserResponse(existingUser),
        token: session.token,
        session: {
          token: session.token,
          expiresAt: session.expiresAt
        }
      });
    } else {
      // New user - signup (fallback values for optional fields)
      const derivedName = sanitizedName || email.split('@')[0] || 'KrushiMitra Farmer';
      if (!sanitizedPhone) {
        return res.status(400).json({
          error: { code: 'VALIDATION_ERROR', message: 'Phone number is required for new users' }
        });
      }
      const now = new Date();
      const userLanguage = preferredLanguage || 'hi';
      const newUser = {
        email,
        name: derivedName,
        phone: sanitizedPhone,
        photo: null,
        profile: {
          name: derivedName,
          phone: sanitizedPhone,
          landSize: landSize?.toString() || '',
          soilType: soilType || '',
          language: userLanguage
        },
        preferredLanguage: userLanguage,
        createdAt: now,
        lastLogin: now
      };

      const result = await usersCollection.insertOne(newUser);
      newUser._id = result.insertedId;

      try {
        await ensureUserContext(result.insertedId, {
          name: newUser.name,
          email: newUser.email,
          phone: newUser.phone || null,
          language: newUser.profile?.language || newUser.preferredLanguage || null
        });

        await ensureUserMemoryDocument(newUser._id.toString());
      } catch (contextError) {
        logger.warn('User context init skipped for new user', {
          userId: result.insertedId.toString(),
          error: contextError.message
        });
      }

      const session = await createSession(result.insertedId, req);
      setSessionCookie(res, session.token, session.expiresAt);

      logger.info('New user registered via email OTP', { userId: result.insertedId.toString(), email });

      return res.json({
        status: 'success',
        message: 'Registration successful',
        user: formatUserResponse(newUser),
        token: session.token,
        session: {
          token: session.token,
          expiresAt: session.expiresAt
        }
      });
    }

  } catch (error) {
    logger.error('Error verifying OTP', { error: error.message });
    res.status(500).json({
      error: { code: 'SERVER_ERROR', message: 'Error verifying OTP' }
    });
  }
});

// GET /auth/session - Validate existing session token (cookie or header)
app.get('/auth/session', async (req, res) => {
  try {
    const token = extractSessionToken(req);
    if (!token) {
      return res.status(401).json({
        error: { code: 'SESSION_NOT_FOUND', message: 'No session token found' }
      });
    }

    const resolved = await resolveSessionUser(token);
    if (!resolved) {
      clearSessionCookie(res);
      return res.status(401).json({
        error: { code: 'SESSION_INVALID', message: 'Session expired or invalid' }
      });
    }

    return res.json({
      status: 'success',
      data: {
        user: formatUserResponse(resolved.user),
        session: {
          token,
          expiresAt: resolved.session.expiresAt
        }
      }
    });
  } catch (error) {
    logger.error('Error checking session', { error: error.message });
    res.status(500).json({
      error: { code: 'SERVER_ERROR', message: 'Unable to verify session' }
    });
  }
});

// POST /auth/logout - Clear session cookie and revoke token
app.post('/auth/logout', async (req, res) => {
  try {
    const token = extractSessionToken(req);
    if (token) {
      await revokeSessionByToken(token);
    }
    clearSessionCookie(res);

    res.json({
      status: 'success',
      message: 'Logged out successfully'
    });
  } catch (error) {
    logger.error('Error logging out', { error: error.message });
    res.status(500).json({
      error: { code: 'SERVER_ERROR', message: 'Unable to logout' }
    });
  }
});

// POST /auth/verify - Accept Firebase idToken, verify, return farmer record or create
app.post('/auth/verify', async (req, res) => {
  try {
    const { idToken } = req.body;

    if (!idToken) {
      logger.warn('Authentication failed - missing ID token');

      return res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'ID token is required'
        }
      });
    }

    const isValid = await verifyFirebaseToken(idToken);
    if (!isValid) {
      logger.warn('Authentication failed - invalid Firebase ID token');

      return res.status(401).json({
        error: {
          code: 'INVALID_TOKEN',
          message: 'Invalid Firebase ID token'
        }
      });
    }

    // For demo purposes, we'll create a mock farmer
    // In a real implementation, you'd extract user info from the token
    const mockFarmer = {
      name: 'Test Farmer',
      phone: '+919876543210',
      language: 'English',
      location: 'Pune, Maharashtra',
      crops: ['Wheat'],
      landSize: 5.0,
      soilType: 'Black soil',
      joinedAt: new Date(),
      updatedAt: new Date()
    };

    logger.info('User authenticated successfully', { farmerId: mockFarmer.phone });

    res.status(200).json({
      status: 'success',
      data: {
        farmer: mockFarmer,
        isNewUser: true
      }
    });
  } catch (error) {
    logger.error('Error verifying token', { error: error.message });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error verifying token'
      }
    });
  }
});

/* ==========================================================================
   CROP DISEASE PREDICTION FLOW
   ========================================================================== */

// Configure Multer for temporary file uploads
const upload = multer({ dest: 'uploads/' });

// POST /predict - Analyze plant image (Phase 1: Plant Check, Phase 2: ID, Phase 3: Disease)
app.post('/predict', upload.single('file'), async (req, res) => {
  const file = req.file;
  if (!file) {
    return res.status(400).json({ success: false, message: 'No file uploaded' });
  }

  const filePath = file.path;

  try {
    console.log(`📸 File uploaded: ${filePath} (${file.size} bytes)`);

    // 1. Plant Identification (PlantNet)
    console.log('🌿 Step 1: Identifying plant with PlantNet...');
    const organ = req.body.organ || 'leaf';
    const plantIdentity = await identifyPlant(filePath, organ);

    // If PlantNet fails or returns low confidence (optional check), we might still want to proceed 
    // or return early. For now, we proceed but log it.
    console.log(`✅ Plant Identified: ${plantIdentity.plant_common} (${plantIdentity.plant_scientific})`);

    // 2. Disease Analysis (Python: YOLOv8 + MobileNetV3)
    console.log('🔬 Step 2: Running AI Disease Analysis (YOLOv8 + MobileNetV3)...');

    // Pass plant name to Python script just in case it's needed for context/logging
    // We wrap the plant name in quotes to handle spaces
    const pythonArgs = ['scripts/crop_inference.py', filePath];
    if (plantIdentity.plant_common) {
      pythonArgs.push(plantIdentity.plant_common);
    }

    const pythonProcess = spawn('python', pythonArgs);

    let pythonData = '';
    let pythonError = '';

    const diseaseResult = await new Promise((resolve, reject) => {
      pythonProcess.stdout.on('data', (data) => {
        pythonData += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error(`Python script exited with code ${code}`);
          console.error(`Python Error: ${pythonError}`);
          // Resolve with error state rather than rejecting to prevent server crash
          resolve({
            success: false,
            disease: "Analysis Failed",
            confidence: 0,
            details: "Computer vision model failed. " + pythonError.split('\n')[0] // Return first line of error
          });
        } else {
          try {
            // Find the JSON part of the output (in case of other stdout logs)
            const jsonStart = pythonData.indexOf('{');
            const jsonEnd = pythonData.lastIndexOf('}');
            if (jsonStart !== -1 && jsonEnd !== -1) {
              const jsonStr = pythonData.substring(jsonStart, jsonEnd + 1);
              const result = JSON.parse(jsonStr);

              if (!result.success) {
                resolve({
                  success: false,
                  disease: "Error",
                  details: result.error || "Python script error"
                });
                return;
              }

              const analysis = result.disease_analysis;
              const leafInfo = result.leaf_detection;

              resolve({
                success: true,
                disease: analysis.disease,
                confidence: analysis.confidence,
                details: `Detected via ${analysis.model}. Leaf detected: ${leafInfo.detected ? 'Yes' : 'No'} (${leafInfo.objects} objects)`,
                // Pass through raw data if needed
                raw: result
              });
            } else {
              console.error('No JSON found in Python output:', pythonData);
              resolve({
                success: false,
                disease: "Analysis Error",
                confidence: 0,
                details: "Invalid output from AI model"
              });
            }
          } catch (e) {
            console.error('Failed to parse Python output:', pythonData);
            resolve({
              success: false,
              disease: "Parse Error",
              confidence: 0,
              details: "Could not parse model results."
            });
          }
        }
      });
    });

    // 3. AI Solution (LLM)
    console.log(`🧠 Step 3: Generating AI Solution for "${diseaseResult.disease}" on "${plantIdentity.plant_common}"...`);

    let aiSolution = {
      treatment: "Consult a local agricultural expert.",
      prevention: [],
      tips: []
    };

    if (diseaseResult.success && diseaseResult.disease !== 'Healthy' && diseaseResult.disease !== 'Analysis Failed' && diseaseResult.disease !== 'Error') {
      if (groq) {
        try {
          const prompt = `
                    Act as an agricultural expert. 
                    
                    Context:
                    - Crop: ${plantIdentity.plant_common} (${plantIdentity.plant_scientific})
                    - Detected Object/Condition: "${diseaseResult.disease}"
                    - Confidence: ${(diseaseResult.confidence * 100).toFixed(1)}%
                    
                    IMPORTANT: The disease detection model is currently running in a generic mode (ImageNet). 
                    - If the "Detected Object" is a pest (e.g., "slug", "beetle", "worm"), treat it as a pest infestation.
                    - If the "Detected Object" is unrelated (e.g., "pot", "velvet", "chain"), ignore it and provide GENERAL healthy care tips for this specific crop.
                    - If the "Detected Object" sounds like a disease (e.g., "rust", "mildew"), treat it as a disease.
                    
                    Provide a strict JSON response (no markdown) with practical advice for an Indian farmer:
                    {
                        "treatment": "Brief treatment plan if a pest/disease is identified, otherwise say 'No specific disease detected, follow general care'. (max 3 sentences)",
                        "prevention": ["List of 3 short prevention/maintenance tips"],
                        "tips": ["List of 2 general maintenance tips for this crop"]
                    }
                `;

          const completion = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.1-8b-instant",
            temperature: 0.3,
            response_format: { type: "json_object" }
          });

          const content = completion.choices[0]?.message?.content;
          if (content) {
            aiSolution = JSON.parse(content);
          }
        } catch (err) {
          console.error("LLM Generation failed:", err.message);
        }
      }
    } else if (diseaseResult.disease === 'Healthy') {
      aiSolution = {
        treatment: "Your crop looks healthy! Keep up the good work.",
        prevention: ["Continue regular watering", "Monitor for early signs of pests"],
        tips: ["Ensure proper sunlight", "Maintain soil moisture"]
      };
    }

    // Final Response Construction
    const finalResponse = {
      success: true,
      plant_identification: plantIdentity,
      disease_detection: diseaseResult,
      ai_solution: aiSolution,
      crop: plantIdentity.plant_common, // For backward compatibility
      message: "Analysis complete"
    };

    // Cleanup
    try {
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
    } catch (e) { }

    res.json(finalResponse);

  } catch (error) {
    console.error('Prediction Flow Error:', error);
    // Cleanup
    try {
      if (req.file?.path && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    } catch (e) { }

    res.status(500).json({
      success: false,
      message: "Internal Server Error during prediction pipeline",
      error: error.message
    });
  }
});

// 3. Activity Tracking

// ... (existing activity routes) ...

/* ==========================================================================
   JUDGE DEMO: ORB VOICE TRIGGER
   ========================================================================== */

// POST /demo/orb-trigger
// Triggered by Frontend Triple-Tap
// Initiates an outgoing call to the user via Twilio
app.post('/demo/orb-trigger', async (req, res) => {
  try {
    const { farmerId, phoneNumber } = req.body;

    // Demo Constraints: Simple validation
    if (!phoneNumber) {
      return res.status(400).json({ error: 'Phone number is required' });
    }

    console.log(`🔮 Demo Triggered! calling ${phoneNumber}...`);

    // HARDCODED DEMO DATA
    // In a real app, we would fetch location from farmerId, then find nearby Mandi, then get prices.
    const demoData = {
      location: "JSPM Tathawade College, Pune",
      prices: [
        { commodity: "Tomato", price: 18, unit: "kg" },
        { commodity: "Onion", price: 22, unit: "kg" },
        { commodity: "Potato", price: 16, unit: "kg" }
      ]
    };

    // TWILIO CALL LOGIC
    const accountSid = process.env.TWILIO_ACCOUNT_SID;
    const authToken = process.env.TWILIO_AUTH_TOKEN;
    const fromNumber = process.env.TWILIO_PHONE_NUMBER; // Must be a purchased Twilio number

    if (!accountSid || !authToken || !fromNumber) {
      console.warn('⚠️ Twilio credentials missing in .env');
      return res.status(503).json({
        error: 'Twilio not configured',
        demoData // Return data anyway so frontend can debug if needed
      });
    }





    const client = twilio(accountSid, authToken);

    // Make the call
    // logic: "When they pick up, hit /demo/orb-voice to get TwiML"
    // Use ngrok/public URL for 'url' parameter if running locally (Render/Heroku handles this automatically)
    // For local dev without ngrok, this callback won't work unless exposed.
    // Assuming backend is accessible via public URL or ngrok.
    const protocol = req.secure ? 'https' : 'http';
    const host = req.get('host');
    const callbackUrl = `${protocol}://${host}/demo/orb-voice`;

    const call = await client.calls.create({
      url: callbackUrl,
      to: phoneNumber,
      from: fromNumber
    });

    console.log(`📞 Call initiated: ${call.sid}`);

    res.json({
      status: 'success',
      message: 'Calling farmer...',
      callSid: call.sid,
      demoData
    });

  } catch (error) {
    console.error('🔥 Demo Trigger Error:', error);
    res.status(500).json({ error: error.message });
  }
});

// POST /demo/orb-voice
// Twilio Webhook: Returns TwiML (Voice XML) to speak to the user
app.post('/demo/orb-voice', (req, res) => {
  const VoiceResponse = twilio.twiml.VoiceResponse;
  const response = new VoiceResponse();

  // HARDCODED VOICE MESSAGE
  // "Namaskar. This is KrushiMitra Orb..."
  const message = `
    Namaskar. 
    This is Krushi Mitra Orb. 
    Today’s mandi price update near Tathawade. 
    Tomato is selling at 18 rupees per kilo. 
    Onion is 22 rupees per kilo. 
    Potato is 16 rupees per kilo. 
    You may plan selling accordingly. 
    Thank you.
  `;

  // <Say> verb with male voice for friendly, natural tone
  response.say({
    voice: 'man', // Male voice - sounds more like a friend
    language: 'hi-IN' // Hindi for natural Indian accent
  }, message);

  // End the call
  response.hangup();

  // Return XML
  res.type('text/xml');
  res.send(response.toString());
});

// 3. Activity Tracking

// POST /activities - Log activity for a farmer
app.post('/activities', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { farmerId, description, type, details } = req.body;

    // Validation
    if (!farmerId || !description) {
      const duration = Date.now() - startTime;
      logger.warn('Activity logging failed - missing required fields', {
        farmerId,
        missingFields: [!farmerId ? 'farmerId' : null, !description ? 'description' : null].filter(Boolean),
        durationMs: duration
      });

      return res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Farmer ID and description are required'
        }
      });
    }

    const activity = {
      farmerId,
      description,
      type: type || 'general',
      details: details || {},
      date: new Date()
    };

    // Insert activity into MongoDB
    const result = await activitiesCollection.insertOne(activity);
    activity._id = result.insertedId;

    const duration = Date.now() - startTime;
    logDBOperation('insertActivity', {
      farmerId,
      activityType: type,
      activityId: result.insertedId.toString(),
      durationMs: duration,
      status: 'success'
    });

    logger.info('Activity logged successfully', {
      farmerId,
      activityId: result.insertedId.toString(),
      durationMs: duration
    });

    res.status(201).json({
      status: 'success',
      data: activity
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('insertActivity', error, {
      farmerId: req.body?.farmerId,
      activityType: req.body?.type,
      durationMs: duration
    });
    logger.error('Error logging activity', {
      error: error.message,
      farmerId: req.body?.farmerId,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error logging activity'
      }
    });
  }
});

// GET /activities/:farmerId - Fetch activity log
app.get('/activities/:farmerId', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { farmerId } = req.params;
    const { limit = 10, offset = 0, type } = req.query;

    // Build query
    const query = { farmerId: farmerId };
    if (type) {
      query.type = type;
    }

    // Get total count
    const total = await activitiesCollection.countDocuments(query);

    // Get paginated activities
    const activities = await activitiesCollection
      .find(query)
      .sort({ date: -1 })
      .skip(parseInt(offset))
      .limit(parseInt(limit))
      .toArray();

    const duration = Date.now() - startTime;
    logDBOperation('findActivities', {
      farmerId,
      limit,
      offset,
      type,
      total,
      returned: activities.length,
      durationMs: duration,
      status: 'success'
    });

    logger.info('Activities retrieved successfully', {
      farmerId,
      count: activities.length,
      durationMs: duration
    });

    res.status(200).json({
      status: 'success',
      data: activities,
      pagination: {
        limit: parseInt(limit),
        offset: parseInt(offset),
        total
      }
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('findActivities', error, {
      farmerId: req.params?.farmerId,
      durationMs: duration
    });
    logger.error('Error fetching activities', {
      error: error.message,
      farmerId: req.params?.farmerId,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error fetching activities'
      }
    });
  }
});

// 4. Mandi Prices

// POST /mandiprices/update - Ingest mandi price data (bulk)
app.post('/mandiprices/update', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { prices } = req.body;

    // Validation
    if (!Array.isArray(prices)) {
      const duration = Date.now() - startTime;
      logger.warn('Mandi price update failed - prices must be an array', {
        durationMs: duration
      });

      return res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Prices must be an array'
        }
      });
    }

    // Insert prices into MongoDB
    const result = await mandipricesCollection.insertMany(prices);

    const duration = Date.now() - startTime;
    logDBOperation('insertMandiPrices', {
      priceCount: prices.length,
      insertedCount: result.insertedCount,
      durationMs: duration,
      status: 'success'
    });

    logger.info('Mandi prices updated successfully', {
      insertedCount: result.insertedCount,
      durationMs: duration
    });

    res.status(200).json({
      status: 'success',
      data: {
        inserted: result.insertedCount,
        updated: 0
      }
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('insertMandiPrices', error, {
      durationMs: duration
    });
    logger.error('Error updating mandi prices', {
      error: error.message,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error updating mandi prices'
      }
    });
  }
});

// GET /mandiprices - Get latest prices
app.get('/mandiprices', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { crop, location } = req.query;

    // Build aggregation pipeline to get latest prices
    const pipeline = [];

    // Match stage
    const match = {};
    if (crop) match.crop = crop;
    if (location) match.location = location;
    if (Object.keys(match).length > 0) {
      pipeline.push({ $match: match });
    }

    // Sort by date descending
    pipeline.push({ $sort: { date: -1 } });

    // Group by crop and location to get latest for each
    pipeline.push({
      $group: {
        _id: { crop: "$crop", location: "$location" },
        latestPrice: { $first: "$$ROOT" }
      }
    });

    // Project to get the original document structure
    pipeline.push({
      $replaceRoot: { newRoot: "$latestPrice" }
    });

    // Execute aggregation
    const latestPrices = await mandipricesCollection.aggregate(pipeline).toArray();

    const duration = Date.now() - startTime;
    logDBOperation('findMandiPrices', {
      crop,
      location,
      returned: latestPrices.length,
      durationMs: duration,
      status: 'success'
    });

    logger.info('Mandi prices retrieved successfully', {
      count: latestPrices.length,
      crop,
      location,
      durationMs: duration
    });

    res.status(200).json({
      status: 'success',
      data: latestPrices
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('findMandiPrices', error, {
      crop,
      location,
      durationMs: duration
    });
    logger.error('Error fetching mandi prices', {
      error: error.message,
      crop,
      location,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error fetching mandi prices'
      }
    });
  }
});

// POST /ai/chat - Send user query to AI
app.post('/ai/chat', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { farmerId, userId, query, context = {}, language: requestedLanguage } = req.body;
    const userIdentifier = normalizeUserKey(userId || farmerId || req.userId, farmerId);

    // Validation
    if (!userIdentifier || !query) {
      const duration = Date.now() - startTime;
      logger.warn('AI chat request failed - missing required fields', {
        userIdentifier,
        missingFields: [!userIdentifier ? 'userId/farmerId' : null, !query ? 'query' : null].filter(Boolean),
        durationMs: duration
      });

      return res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'User identifier and query are required'
        }
      });
    }

    let userDoc = null;
    try {
      userDoc = await findUserDocument(userIdentifier, farmerId);
    } catch (lookupError) {
      logger.warn('Could not fetch user document for AI context', {
        userId: userIdentifier,
        error: lookupError.message
      });
    }

    const memoryKey = normalizeUserKey(userDoc?._id, userIdentifier);
    const contextUserId = userDoc?._id || toObjectId(userIdentifier);

    // STEP 1: Fetch full UserContext (userData + last 5 query items)
    let userContextPayload = null;
    if (contextUserId) {
      try {
        const contextDoc = await fetchUserContext(contextUserId);
        userContextPayload = formatUserContextResponse(contextDoc);

        if (userContextPayload) {
          logger.info('UserContext loaded for AI request', {
            userId: contextUserId?.toString(),
            hasUserData: !!userContextPayload.userData,
            queryCount: userContextPayload.query?.length || 0
          });
        }
      } catch (contextError) {
        logger.warn('Failed to fetch user context for AI chat', {
          error: contextError.message,
          userId: contextUserId?.toString() || userIdentifier
        });
      }
    }

    const resolvedLanguage = requestedLanguage && typeof requestedLanguage === 'string' && requestedLanguage.trim().length > 0
      ? requestedLanguage.trim()
      : (userDoc?.preferredLanguage || userDoc?.profile?.language || 'en');

    // STEP 2: Include UserContext in AI request payload
    const aiContextPayload = {
      ...(context || {}),
      userContext: userContextPayload  // Contains userData + last 5 query items
    };

    // Get farmer profile for context
    let farmerProfile = null;
    try {
      if (farmerId) {
        farmerProfile = await farmersCollection.findOne({ phone: farmerId });
      }
      if (!farmerProfile && (userDoc?.phone || userDoc?.profile?.phone)) {
        const fallbackPhone = userDoc?.phone || userDoc?.profile?.phone;
        if (fallbackPhone) {
          farmerProfile = await farmersCollection.findOne({ phone: fallbackPhone });
        }
      }
    } catch (error) {
      logger.warn('Could not fetch farmer profile for AI context', {
        farmerId: farmerId || userIdentifier,
        error: error.message
      });
    }

    const memoryEntries = await getUserMemoryEntries(memoryKey, DEFAULT_MEMORY_SLICE);

    // STEP 3: Generate AI response with full context
    // Build structured prompt with user data + last 5 conversations + current query
    const userData = userContextPayload?.userData || {};
    const lastConversations = userContextPayload?.query || [];

    // Format the LLM prompt with all user context
    const llmPrompt = {
      query: query,
      user_data: {
        user_name: userData.name || 'Unknown',
        user_email: userData.email || 'Not provided',
        user_phone: userData.phone || 'Not provided',
        user_language: userData.language || resolvedLanguage,
        user_location: userData.location ? {
          address: userData.location.address,
          latitude: userData.location.latitude,
          longitude: userData.location.longitude
        } : null,
        user_weather: userData.weather ? {
          temperature: userData.weather.temperature,
          humidity: userData.weather.humidity,
          condition: userData.weather.condition
        } : null
      },
      last_5_conversations: lastConversations.map(conv => ({
        role: conv.role,
        message: conv.message
      })),
      farmer_profile: farmerProfile ? {
        phone: farmerProfile.phone,
        crops: farmerProfile.crops,
        land_size: farmerProfile.landSize
      } : null
    };

    // Log the exact prompt that LLM will receive
    logger.info('LLM Prompt Generated', {
      userId: contextUserId?.toString(),
      prompt: JSON.stringify(llmPrompt, null, 2)
    });

    // Call Groq API for Llama 3.1 response
    let aiResponse;
    try {
      const chatCompletion = await groq.chat.completions.create({
        messages: [
          {
            role: "system",
            content: `You are KrushiMitra Orb, a helpful agricultural AI assistant for Indian farmers.
                              User Context: ${JSON.stringify(llmPrompt.user_data)}
                              User Profile: ${JSON.stringify(llmPrompt.farmer_profile)}
                              Provide helpful, practical agricultural advice. Keep responses concise and friendly.
                              If language is not English, respond in the requested language (${resolvedLanguage}).`
          },
          ...llmPrompt.last_5_conversations,
          {
            role: "user",
            content: llmPrompt.query
          }
        ],
        model: "llama-3.1-8b-instant",
        temperature: 0.7,
        max_tokens: 1024,
        top_p: 1,
        stream: false,
        stop: null
      });

      aiResponse = chatCompletion.choices[0]?.message?.content || "I'm sorry, I couldn't generate a response at this time.";
    } catch (groqError) {
      logger.error('Groq API Error', { error: groqError.message });
      // Fallback or error message
      aiResponse = "I am currently experiencing high traffic. Please try again later. (AI Error)";
    }

    // Mock automations
    const automations = [
      {
        type: 'mandi_alert',
        triggered: true,
        details: {
          message: 'Wheat prices in your area are currently favorable'
        }
      }
    ];

    // Mock related data
    const relatedData = {
      weatherForecast: {
        nextWeekRainfall: '50mm',
        temperatureRange: '25-32°C'
      }
    };

    const memoryToAppend = [
      { role: 'user', content: query },
      { role: 'assistant', content: aiResponse }
    ];

    // Save AI interaction to database
    try {
      const aiInteraction = {
        farmerId: farmerId || userDoc?.phone || memoryKey,
        userId: memoryKey,
        query,
        response: aiResponse,
        context: aiContextPayload,
        language: resolvedLanguage,
        timestamp: new Date()
      };

      await aiinteractionsCollection.insertOne(aiInteraction);
      logger.info('AI interaction saved to database', { userIdentifier });
    } catch (dbError) {
      logger.error('Failed to save AI interaction to database', {
        error: dbError.message,
        userIdentifier
      });
      // Don't fail the whole request if we can't save to DB, just log the error
    }

    // STEP 4: Save user question and AI response to UserContext.query
    if (contextUserId) {
      try {
        logger.info('Attempting to save conversation to UserContext', {
          userId: contextUserId?.toString(),
          queryLength: query?.length,
          responseLength: aiResponse?.length
        });

        await appendChatMessage(contextUserId, [
          { role: 'user', message: query },        // Save user question
          { role: 'assistant', message: aiResponse } // Save AI response
        ]);

        logger.info('Conversation saved to UserContext.query', {
          userId: contextUserId?.toString(),
          messagesSaved: 2
        });
      } catch (contextAppendError) {
        logger.error('Failed to update UserContext query array', {
          error: contextAppendError.message,
          stack: contextAppendError.stack,
          userId: contextUserId?.toString() || userIdentifier
        });
      }
    } else {
      logger.warn('No contextUserId available, skipping UserContext update', {
        userIdentifier,
        hasUserDoc: !!userDoc
      });
    }

    const duration = Date.now() - startTime;
    logDBOperation('aiChat', {
      userId: memoryKey,
      durationMs: duration,
      status: 'success'
    });

    logger.info('AI chat response generated', {
      userId: memoryKey,
      durationMs: duration
    });

    const latestMemory = [...memoryEntries, ...memoryToAppend].slice(-DEFAULT_MEMORY_SLICE);

    res.status(200).json({
      status: 'success',
      data: {
        response: aiResponse,
        automations,
        relatedData,
        memory: latestMemory,
        language: resolvedLanguage,
        userContext: userContextPayload
      }
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('aiChat', error, {
      userId: req.body?.userId || req.body?.farmerId,
      durationMs: duration
    });
    logger.error('Error processing AI chat request', {
      error: error.message,
      userId: req.body?.userId || req.body?.farmerId,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error processing AI chat request'
      }
    });
  }
});

// POST /ai/interactions - Save AI interaction (user query and AI response)
app.post('/ai/interactions', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { farmerId, userId, query, response, context, language } = req.body;

    // Accept either farmerId (legacy) or userId (new)
    const userIdentifier = normalizeUserKey(userId || farmerId || req.userId, farmerId);

    // Validation
    if (!userIdentifier || !query || !response) {
      const duration = Date.now() - startTime;
      logger.warn('AI interaction save failed - missing required fields', {
        userIdentifier,
        missingFields: [
          !userIdentifier ? 'userId/farmerId' : null,
          !query ? 'query' : null,
          !response ? 'response' : null
        ].filter(Boolean),
        durationMs: duration
      });

      return res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'User ID, query, and response are required'
        }
      });
    }

    // Save AI interaction to database
    const aiInteraction = {
      userId: userIdentifier,  // Store as userId for consistency
      farmerId: farmerId || userIdentifier,  // Keep farmerId for backward compatibility
      query,
      response,
      context: context || {},
      language: language || 'en',
      timestamp: new Date()
    };

    const result = await aiinteractionsCollection.insertOne(aiInteraction);

    try {
      await appendUserMemoryEntries(userIdentifier, [
        { role: 'user', content: query },
        { role: 'assistant', content: response }
      ]);
    } catch (memoryError) {
      logger.warn('Failed to persist manual AI interaction memory', {
        error: memoryError.message,
        userIdentifier
      });
    }

    const duration = Date.now() - startTime;
    logDBOperation('saveAIInteraction', {
      userId: userIdentifier,
      interactionId: result.insertedId.toString(),
      durationMs: duration,
      status: 'success'
    });

    logger.info('AI interaction saved successfully', {
      userId: userIdentifier,
      interactionId: result.insertedId.toString(),
      durationMs: duration
    });

    res.status(201).json({
      status: 'success',
      data: {
        interactionId: result.insertedId
      }
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logDBError('saveAIInteraction', error, {
      userId: req.body?.userId || req.body?.farmerId,
      durationMs: duration
    });
    logger.error('Error saving AI interaction', {
      error: error.message,
      userId: req.body?.userId || req.body?.farmerId,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error saving AI interaction'
      }
    });
  }
});

// Weather cache - 10 minutes TTL
const weatherCache = new Map();
const WEATHER_CACHE_TTL = 10 * 60 * 1000; // 10 minutes

// Weather endpoint - Tomorrow.io integration with caching & fallback
app.get('/weather', async (req, res) => {
  const startTime = Date.now();
  try {
    const { lat, lon } = req.query;

    // Validation
    if (!lat || !lon) {
      const duration = Date.now() - startTime;
      logger.warn('Weather request failed - missing coordinates', {
        durationMs: duration
      });

      return res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Latitude and longitude are required'
        }
      });
    }

    // Check cache first
    const cacheKey = `${parseFloat(lat).toFixed(2)},${parseFloat(lon).toFixed(2)}`;
    const cachedData = weatherCache.get(cacheKey);

    if (cachedData && Date.now() - cachedData.timestamp < WEATHER_CACHE_TTL) {
      const duration = Date.now() - startTime;
      logger.info('Weather data served from cache', {
        lat, lon, durationMs: duration
      });
      return res.status(200).json({
        status: 'success',
        data: { ...cachedData.data, cached: true }
      });
    }

    const apiKey = process.env.TOMORROW_API_KEY;
    if (!apiKey) {
      logger.error('Tomorrow.io API key not configured');
      // Return fallback data if available
      if (cachedData) {
        return res.status(200).json({
          status: 'success',
          data: { ...cachedData.data, cached: true, stale: true }
        });
      }
      return res.status(500).json({
        error: {
          code: 'CONFIG_ERROR',
          message: 'Weather service not configured'
        }
      });
    }

    // Call Tomorrow.io API
    const tomorrowUrl = `https://api.tomorrow.io/v4/weather/forecast?location=${lat},${lon}&apikey=${apiKey}`;
    const response = await fetch(tomorrowUrl);

    if (!response.ok) {
      // Handle rate limiting - use cached data if available
      if (response.status === 429) {
        logger.warn('Tomorrow.io rate limit exceeded, using cache or fallback', {
          status: response.status
        });

        // Return stale cache if available
        if (cachedData) {
          return res.status(200).json({
            status: 'success',
            data: { ...cachedData.data, cached: true, stale: true }
          });
        }

        // Return fallback data
        return res.status(200).json({
          status: 'success',
          data: {
            temperature: 28,
            humidity: 65,
            windSpeed: 10,
            precipitationProbability: 20,
            weatherCode: 1101,
            condition: 'Partly Cloudy',
            advisory: 'Weather data temporarily unavailable. Using estimated conditions.',
            fallback: true
          }
        });
      }
      throw new Error(`Tomorrow.io API error: ${response.status}`);
    }

    const data = await response.json();

    // Parse current weather data
    const current = data.timelines?.minutely?.[0]?.values || data.timelines?.hourly?.[0]?.values || {};

    // Extract weather values
    const temperature = current.temperature || current.temperatureApparent || 0;
    const humidity = current.humidity || 0;
    const windSpeed = current.windSpeed || 0;
    const precipitationProbability = current.precipitationProbability || 0;
    const weatherCode = current.weatherCode || 0;

    // Extract 7-day forecast from daily timeline
    const dailyTimeline = data.timelines?.daily || [];
    const forecast = dailyTimeline.slice(0, 7).map(day => {
      const values = day.values || {};
      return {
        date: day.time,
        temperatureMax: Math.round(values.temperatureMax || values.temperature || 0),
        temperatureMin: Math.round(values.temperatureMin || values.temperature || 0),
        weatherCode: values.weatherCode || 0,
        precipitationProbability: Math.round(values.precipitationProbability || 0)
      };
    });

    // Map weather codes to descriptions
    const weatherDescriptions = {
      0: 'Unknown',
      1000: 'Clear',
      1001: 'Cloudy',
      1100: 'Mostly Clear',
      1101: 'Partly Cloudy',
      1102: 'Mostly Cloudy',
      2000: 'Fog',
      2100: 'Light Fog',
      3000: 'Light Wind',
      3001: 'Wind',
      3002: 'Strong Wind',
      4000: 'Drizzle',
      4001: 'Rain',
      4200: 'Light Rain',
      4201: 'Heavy Rain',
      5000: 'Snow',
      5001: 'Flurries',
      5100: 'Light Snow',
      5101: 'Heavy Snow',
      6000: 'Freezing Drizzle',
      6001: 'Freezing Rain',
      6200: 'Light Freezing Rain',
      6201: 'Heavy Freezing Rain',
      7000: 'Ice Pellets',
      7101: 'Heavy Ice Pellets',
      7102: 'Light Ice Pellets',
      8000: 'Thunderstorm'
    };

    const condition = weatherDescriptions[weatherCode] || 'Unknown';

    // Generate advisory based on weather conditions
    let advisory = '';
    if (precipitationProbability > 70) {
      advisory = 'High chance of rain. Postpone spraying activities. Good time for indoor planning.';
    } else if (precipitationProbability > 40) {
      advisory = 'Moderate rain expected. Good time for irrigation planning and soil preparation.';
    } else if (temperature > 35) {
      advisory = 'High temperature. Ensure adequate irrigation. Avoid midday fieldwork.';
    } else if (temperature < 15) {
      advisory = 'Cool weather. Monitor frost-sensitive crops. Good for harvesting.';
    } else if (windSpeed > 20) {
      advisory = 'Windy conditions. Avoid pesticide application. Secure farm equipment.';
    } else {
      advisory = 'Favorable conditions for farming activities. Plan your fieldwork accordingly.';
    }

    const duration = Date.now() - startTime;
    logger.info('Weather data retrieved successfully', {
      lat,
      lon,
      temperature,
      condition,
      durationMs: duration
    });

    // Prepare weather data for database
    const weatherDataDoc = {
      location: {
        type: 'Point',
        coordinates: [parseFloat(lon), parseFloat(lat)]
      },
      latitude: parseFloat(lat),
      longitude: parseFloat(lon),
      temperature: Math.round(temperature),
      humidity: Math.round(humidity),
      windSpeed: Math.round(windSpeed),
      precipitationProbability: Math.round(precipitationProbability),
      weatherCode,
      condition,
      advisory,
      forecast,
      timestamp: new Date(),
      source: 'tomorrow.io'
    };

    // Save to MongoDB
    try {
      await weatherDataCollection.insertOne(weatherDataDoc);
      logger.info('Weather data saved to database', { lat, lon });
    } catch (dbError) {
      logger.error('Failed to save weather data to database', {
        error: dbError.message,
        lat,
        lon
      });
      // Don't fail the request if DB save fails
    }

    // Cache the result
    weatherCache.set(cacheKey, {
      timestamp: Date.now(),
      data: {
        temperature: Math.round(temperature),
        humidity: Math.round(humidity),
        windSpeed: Math.round(windSpeed),
        precipitationProbability: Math.round(precipitationProbability),
        weatherCode,
        condition,
        advisory,
        forecast
      }
    });

    res.status(200).json({
      status: 'success',
      data: {
        temperature: Math.round(temperature),
        humidity: Math.round(humidity),
        windSpeed: Math.round(windSpeed),
        precipitationProbability: Math.round(precipitationProbability),
        weatherCode,
        condition,
        advisory,
        forecast,
        cached: false
      }
    });
  } catch (error) {
    const duration = Date.now() - startTime;
    logger.error('Error fetching weather data', {
      error: error.message,
      durationMs: duration
    });

    // Try to return cached data even if stale
    const cacheKey = `${parseFloat(req.query.lat).toFixed(2)},${parseFloat(req.query.lon).toFixed(2)}`;
    const cachedData = weatherCache.get(cacheKey);

    if (cachedData) {
      logger.info('Returning stale cached data due to error');
      return res.status(200).json({
        status: 'success',
        data: { ...cachedData.data, cached: true, stale: true }
      });
    }

    // Last resort fallback
    res.status(200).json({
      status: 'success',
      data: {
        temperature: 28,
        humidity: 65,
        windSpeed: 10,
        precipitationProbability: 20,
        weatherCode: 1101,
        condition: 'Partly Cloudy',
        advisory: 'Weather data temporarily unavailable. Using estimated conditions.',
        fallback: true
      }
    });
  }
});

// POST /weather/location - Save user location and address
app.post('/weather/location', authenticate, async (req, res) => {
  const startTime = Date.now();
  try {
    const { lat, lon, address, userId } = req.body;

    // Validation
    if (!lat || !lon) {
      const duration = Date.now() - startTime;
      logger.warn('Location save request failed - missing coordinates', {
        durationMs: duration
      });

      return res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Latitude and longitude are required'
        }
      });
    }

    const parsedLat = parseFloat(lat);
    const parsedLon = parseFloat(lon);
    const resolvedAddress = address?.trim() || 'Address not provided';

    // Prepare location document
    const locationDoc = {
      userId: userId || 'anonymous',
      location: {
        type: 'Point',
        coordinates: [parsedLon, parsedLat]
      },
      latitude: parsedLat,
      longitude: parsedLon,
      address: resolvedAddress,
      timestamp: new Date(),
      lastAccessed: new Date()
    };

    // Update or insert location (upsert based on userId)
    try {
      await weatherDataCollection.updateOne(
        { userId: locationDoc.userId },
        {
          $set: locationDoc,
          $setOnInsert: { createdAt: new Date() }
        },
        { upsert: true }
      );

      let userProfileUpdated = false;
      if (userId && userId !== 'anonymous' && usersCollection) {
        try {
          const resolvedUser = await findUserDocument(userId, null);
          if (resolvedUser) {
            await usersCollection.updateOne(
              { _id: resolvedUser._id },
              {
                $set: {
                  lastKnownLocation: {
                    latitude: parsedLat,
                    longitude: parsedLon,
                    address: resolvedAddress,
                    updatedAt: new Date()
                  },
                  'profile.address': resolvedAddress,
                  'profile.location': {
                    latitude: parsedLat,
                    longitude: parsedLon
                  }
                }
              }
            );
            userProfileUpdated = true;
          }
        } catch (userError) {
          logger.warn('Failed to update user profile with location', {
            userId,
            error: userError.message
          });
        }
      }

      const duration = Date.now() - startTime;
      logger.info('User location and address saved to database', {
        userId: locationDoc.userId,
        lat,
        lon,
        address: resolvedAddress,
        userProfileUpdated,
        durationMs: duration
      });

      res.status(200).json({
        status: 'success',
        message: 'Location and address saved successfully',
        userProfileUpdated
      });
    } catch (dbError) {
      const duration = Date.now() - startTime;
      logger.error('Failed to save location to database', {
        error: dbError.message,
        userId: locationDoc.userId,
        durationMs: duration
      });

      res.status(500).json({
        error: {
          code: 'DATABASE_ERROR',
          message: 'Failed to save location data'
        }
      });
    }
  } catch (error) {
    const duration = Date.now() - startTime;
    logger.error('Error processing location save request', {
      error: error.message,
      durationMs: duration
    });

    res.status(500).json({
      error: {
        code: 'SERVER_ERROR',
        message: 'Error saving location data'
      }
    });
  }
});

// Root endpoint - API info
app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    message: 'KrushiMitra Backend API',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      tts: '/tts?lang=hi&text=नमस्ते',
      weather: '/weather?lat=18.5204&lon=73.8567',
      auth: '/auth/verify',
      farmers: '/farmers (POST, GET)',
      activities: '/activities (POST, GET)',
      mandiprices: '/mandiprices (GET, POST)',
      ai: '/ai/chat (POST), /ai/interactions (POST)'
    }
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// GET /training/metrics - Reads training-metrics.json if present
app.get('/training/metrics', async (req, res) => {
  try {
    const metricsPath = path.join(__dirname, 'training-metrics.json');
    let metrics;
    if (fs.existsSync(metricsPath)) {
      const raw = await fs.promises.readFile(metricsPath, 'utf-8');
      metrics = JSON.parse(raw);
    } else {
      metrics = {
        model: 'yolov8n',
        dataset: process.env.TRAIN_DATASET || 'unknown',
        epoch: Number(process.env.TRAIN_EPOCH || 0),
        accuracyProxy: Number(process.env.VAL_F1 || 0.0),
        map50: Number(process.env.VAL_MAP50 || 0.0),
        timestamp: new Date().toISOString(),
        note: 'Metrics file not found; showing placeholder values.'
      };
    }
    res.json({ status: 'success', data: metrics });
  } catch (e) {
    res.status(500).json({ status: 'error', error: e.message });
  }
});

// 404 handler
app.use('*', (req, res) => {
  logger.warn('Endpoint not found', {
    url: req.originalUrl,
    method: req.method,
    ip: req.ip
  });

  res.status(404).json({
    error: {
      code: 'NOT_FOUND',
      message: 'Endpoint not found'
    }
  });
});

// Global error handler
app.use((error, req, res, next) => {
  logger.error('Unhandled error', {
    error: error.message,
    stack: error.stack,
    url: req.originalUrl,
    method: req.method,
    ip: req.ip
  });

  res.status(500).json({
    error: {
      code: 'SERVER_ERROR',
      message: 'Internal server error'
    }
  });
});

// Start server only if executed directly
if (require.main === module) {
  app.listen(PORT, '0.0.0.0', async () => {
    logger.info(`KrushiMitra API server running on port ${PORT} (bound to all interfaces)`);
    try {
      await initializeCollections();
      logger.info('Database collections initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize database collections', { error: error.message });
    }
  });
}

module.exports = { app };

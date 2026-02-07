const { ObjectId } = require('mongodb');

const COLLECTION_NAME = 'user_context';
let userContextCollection;

async function safelyCreateIndex(collection, keys, options) {
  try {
    await collection.createIndex(keys, options);
  } catch (error) {
    // Log but do not block server boot if indexes already exist or data conflicts
    const label = JSON.stringify({ keys, options });
    console.warn(`UserContext index creation skipped: ${label}`, error.message);
  }
}

function normalizeObjectId(value) {
  if (!value) {
    return null;
  }
  if (value instanceof ObjectId) {
    return value;
  }
  if (typeof value === 'string' && ObjectId.isValid(value)) {
    return new ObjectId(value);
  }
  if (value?._id && ObjectId.isValid(value._id)) {
    return new ObjectId(value._id);
  }
  return null;
}

function sanitizeProfile(profile = {}, fallback = {}) {
  const base = {
    name: fallback.name || null,
    email: fallback.email || null,
    phone: fallback.phone || null,
    language: fallback.language || null
  };
  return {
    ...base,
    ...profile
  };
}

function sanitizeUserData(userData = {}) {
  if (!userData || typeof userData !== 'object') {
    return {};
  }
  return {
    ...(userData.name ? { name: userData.name } : {}),
    ...(userData.email ? { email: userData.email } : {}),
    ...(userData.phone ? { phone: userData.phone } : {}),
    ...(userData.language ? { language: userData.language } : {}),
    ...(userData.landSize ? { landSize: userData.landSize } : {}),
    ...(userData.soilType ? { soilType: userData.soilType } : {}),
    ...(userData.crops ? { crops: userData.crops } : {}),
  };
}

function sanitizeLocation(location = {}) {
  if (!location || typeof location !== 'object') {
    return null;
  }
  const sanitized = {
    address: location.address || null,
    latitude: typeof location.lat === 'number' ? location.lat : location.latitude ?? null,
    longitude: typeof location.lon === 'number' ? location.lon : location.longitude ?? null,
    precision: location.precision || null,
    raw: location.raw || null,
    updatedAt: new Date()
  };
  if (
    sanitized.address === null &&
    sanitized.latitude === null &&
    sanitized.longitude === null &&
    sanitized.raw === null
  ) {
    return null;
  }
  return sanitized;
}

function sanitizeWeather(weather = {}) {
  if (!weather || typeof weather !== 'object') {
    return null;
  }
  const sanitized = {
    temperature: weather.temperature ?? weather.temp ?? null,
    humidity: weather.humidity ?? null,
    condition: weather.condition || weather.summary || null,
    windSpeed: weather.windSpeed ?? weather.wind ?? null,
    precipitationProbability: weather.precipitationProbability ?? weather.precipChance ?? null,
    source: weather.source || 'app',
    updatedAt: new Date()
  };
  const meaningful = Object.values({ ...sanitized, updatedAt: undefined }).some(
    (value) => value !== null && value !== undefined
  );
  return meaningful ? sanitized : null;
}

function getCollection() {
  if (!userContextCollection) {
    throw new Error('UserContext collection not initialized');
  }
  return userContextCollection;
}

async function initUserContextCollection(db) {
  const collection = db.collection(COLLECTION_NAME);
  userContextCollection = collection;

  await safelyCreateIndex(collection, { userId: 1 }, { unique: true, background: true });
  await safelyCreateIndex(collection, { 'location.updatedAt': -1 }, { background: true });
  await safelyCreateIndex(collection, { 'weather.updatedAt': -1 }, { background: true });

  return collection;
}

async function ensureUserContext(userId, profile = {}) {
  const collection = getCollection();
  const normalizedId = normalizeObjectId(userId);
  if (!normalizedId) {
    return null;
  }
  const now = new Date();
  
  // Build userData object from profile
  const userData = {
    ...sanitizeUserData(profile),
    location: null,
    weather: null
  };
  
  // Check if document exists
  const existing = await collection.findOne({ userId: normalizedId });
  
  if (existing) {
    // Update existing user's userData (preserving location/weather if not provided)
    const updateData = { ...userData };
    if (existing.userData?.location) {
      updateData.location = existing.userData.location;
    }
    if (existing.userData?.weather) {
      updateData.weather = existing.userData.weather;
    }
    
    await collection.updateOne(
      { userId: normalizedId },
      {
        $set: {
          userData: updateData,
          updatedAt: now
        }
      }
    );
  } else {
    // Create new user context with userData and query sections
    await collection.updateOne(
      { userId: normalizedId },
      {
        $setOnInsert: {
          userId: normalizedId,
          userData: userData,
          query: [],  // Last 5 AI conversations
          createdAt: now,
          updatedAt: now
        }
      },
      { upsert: true }
    );
  }
  
  return normalizedId;
}

async function updateLocationAndWeather(userId, { profile = {}, location, weather } = {}) {
  const collection = getCollection();
  const normalizedId = normalizeObjectId(userId);
  if (!normalizedId) {
    return null;
  }
  
  // Ensure document exists
  await ensureUserContext(userId, profile);
  
  const update = {
    $set: {
      updatedAt: new Date()
    }
  };
  
  const sanitizedLocation = sanitizeLocation(location);
  const sanitizedWeather = sanitizeWeather(weather);
  
  if (sanitizedLocation) {
    update.$set['userData.location'] = sanitizedLocation;
  }
  if (sanitizedWeather) {
    update.$set['userData.weather'] = sanitizedWeather;
  }
  
  await collection.updateOne({ userId: normalizedId }, update);
  return collection.findOne({ userId: normalizedId });
}

async function appendChatMessage(userId, chatEntry = {}) {
  const collection = getCollection();
  const normalizedId = normalizeObjectId(userId);
  if (!normalizedId) {
    return null;
  }
  
  // Ensure document exists
  const existing = await collection.findOne({ userId: normalizedId });
  if (!existing) {
    const sampleEntry = Array.isArray(chatEntry) ? chatEntry[0] : chatEntry;
    const profile = sampleEntry?.profile || {};
    await ensureUserContext(normalizedId, profile);
  }

  const entries = (Array.isArray(chatEntry) ? chatEntry : [chatEntry])
    .map((entry) => {
      const messageText = entry?.message ? entry.message.toString() : '';
      return {
        role: entry?.role === 'assistant' ? 'assistant' : 'user',
        message: messageText
      };
    })
    .filter((entry) => entry.message && entry.message.trim().length > 0);

  if (entries.length === 0) {
    return collection.findOne({ userId: normalizedId }, { projection: { query: 1, userId: 1 } });
  }

  // Append to query array with $slice: -5 to keep only last 5
  await collection.updateOne(
    { userId: normalizedId },
    {
      $push: {
        query: {
          $each: entries,
          $slice: -5
        }
      },
      $set: { updatedAt: new Date() }
    }
  );
  return collection.findOne({ userId: normalizedId }, { projection: { query: 1, userId: 1 } });
}

async function fetchUserContext(userId) {
  const collection = getCollection();
  const normalizedId = normalizeObjectId(userId);
  if (!normalizedId) {
    return null;
  }
  return collection.findOne({ userId: normalizedId });
}

module.exports = {
  initUserContextCollection,
  ensureUserContext,
  updateLocationAndWeather,
  appendChatMessage,
  fetchUserContext,
  sanitizeProfile,
  sanitizeUserData
};

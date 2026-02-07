/**
 * KrushiAI Prompt Template
 *
 * Builds a structured system prompt that reflects how the KrushiMitra app
 * gathers context (profile, weather, mandi data, memories) before querying
 * an LLM. The strings stay ASCII to keep builds portable while still
 * instructing the model to answer in regional languages.
 */

const LANGUAGE_LABELS = {
  en: 'English',
  hi: 'Hindi',
  mr: 'Marathi',
  ml: 'Malayalam'
};

const farmerPersonas = {
  en: `SYSTEM ROLE:
You are KrushiMitra Orb — a calm, trusted farming companion, not a chatbot.
You speak like a knowledgeable human friend who lives with the farmer's reality.

You REMEMBER:
- the farmer’s crops, land, past problems, past advice, and outcomes
- previous calls, alerts, mandi discussions, and weather concerns

You BEHAVE LIKE A HUMAN:
- You acknowledge time ("this morning", "yesterday", "last week")
- You connect past events to today’s advice
- You reassure before instructing
- You never overwhelm

Your goal is not to answer questions,
but to reduce farmer stress and guide the next right decision.

You combine:
- farmer profile
- memories
- weather
- mandi prices
- crop stage
- local context

Always prioritize:
trust, clarity, safety, affordability, and confidence.`,

  mr: `SYSTEM ROLE:
तू KrushiMitra Orb आहेस — एक शांत, विश्वासू शेतीसाथी.
तू चॅटबॉट नाहीस, तू शेतकऱ्याचा ओळखीचा माणूस आहेस.

तुला आठवतं:
- शेतकऱ्याची शेती, पिकं, आधीच्या अडचणी
- आधी दिलेला सल्ला, कॉल्स, हवामान आणि बाजार चर्चा

तू माणसासारखं बोलतोस:
- "आज सकाळी", "काल", "मागच्या वेळी" असं वेळेशी जोडतोस
- आधी धीर देतोस, मग मार्गदर्शन करतोस
- गरजेपेक्षा जास्त माहिती देत नाहीस

तुझं काम उत्तर देणं नाही —
शेतकऱ्याचं ओझं कमी करणं आहे.

नेहमी विश्वास, स्पष्टता, कमी खर्च आणि सुरक्षित सल्ला दे.`
};


const responseGuidelines = {
  en: `RESPONSE BEHAVIOR:
1. Always start by acknowledging the farmer’s situation or memory.
   Example: "You mentioned last week that…", "Since it rained yesterday…"

2. Speak in a calm, reassuring tone before giving advice.
   Never start with commands.

3. Use short human sentences. Avoid textbook explanations.

4. If giving advice:
   - explain WHY in one line
   - then give WHAT TO DO in steps

5. When data is missing, say it honestly:
   "I don’t have today’s soil moisture yet."

6. Gently remind the farmer:
   - to upload a photo
   - to log activity
   - to check mandi prices
   only if it truly helps now.

7. Never sound urgent unless there is real risk.
8. End by telling the farmer you are watching this with them.`,

  mr: `RESPONSE BEHAVIOR:
1. सुरुवातीला शेतकऱ्याची परिस्थिती ओळखा:
   "काल पाऊस झाला होता म्हणून…", "मागच्या वेळेस तुम्ही सांगितलं होतं…"

2. आधी धीर द्या, मग सल्ला द्या.

3. वाक्ये लहान आणि माणसासारखी ठेवा.

4. सल्ला देताना:
   - एक ओळीत कारण
   - मग पायऱ्यांमध्ये काय करायचं

5. माहिती नसेल तर प्रामाणिकपणे सांगा.

6. गरज असेल तेव्हाच KrushiMitra फीचर्स आठवण करून द्या.

7. शेवटी सांगाच:
   "मी याकडे लक्ष ठेवतोय."`
};


const MAX_MEMORY_ROWS = 5;

function getFarmerPersona(language) {
  return farmerPersonas[language] || farmerPersonas.en;
}

function getResponseGuidelines(language) {
  return responseGuidelines[language] || responseGuidelines.en;
}

function summarizeList(value) {
  if (!value) {
    return null;
  }
  if (Array.isArray(value)) {
    return value.filter(Boolean).join(', ');
  }
  return String(value);
}

function buildProfileSection(farmerProfile, userProfile) {
  const profile = farmerProfile || userProfile || {};
  const details = [];

  if (profile.name) details.push(`Name: ${profile.name}`);
  if (profile.location) details.push(`Location: ${profile.location}`);
  if (profile.village) details.push(`Village: ${profile.village}`);
  if (profile.taluka) details.push(`Taluka: ${profile.taluka}`);
  if (profile.district) details.push(`District: ${profile.district}`);
  if (profile.crops) details.push(`Crops: ${summarizeList(profile.crops)}`);
  if (profile.landSize) details.push(`Land Size: ${profile.landSize}`);
  if (profile.soilType) details.push(`Soil: ${profile.soilType}`);
  if (profile.irrigation) details.push(`Irrigation: ${profile.irrigation}`);

  if (details.length === 0) {
    return '';
  }

  return `Farmer Profile:\n${details.join('\n')}`;
}

function buildWeatherSection(weather) {
  if (!weather) {
    return '';
  }

  const parts = [];
  if (weather.condition) parts.push(`Condition: ${weather.condition}`);
  if (weather.temperature) parts.push(`Temperature: ${weather.temperature}`);
  if (weather.humidity) parts.push(`Humidity: ${weather.humidity}`);
  if (weather.rainfall) parts.push(`Rainfall: ${weather.rainfall}`);
  if (weather.wind) parts.push(`Wind: ${weather.wind}`);
  if (weather.advisory) parts.push(`Advisory: ${weather.advisory}`);

  if (parts.length === 0) {
    return '';
  }

  return `Weather Snapshot:\n${parts.join('\n')}`;
}

function buildMandiSection(mandi) {
  if (!mandi) {
    return '';
  }

  const parts = [];
  if (mandi.crop) parts.push(`Crop: ${mandi.crop}`);
  if (mandi.price) parts.push(`Price: ${mandi.price}`);
  if (mandi.trend) parts.push(`Trend: ${mandi.trend}`);
  if (mandi.market) parts.push(`Market: ${mandi.market}`);

  if (parts.length === 0) {
    return '';
  }

  return `Mandi Insight:\n${parts.join('\n')}`;
}

function buildAlertsSection(alerts) {
  if (!alerts || alerts.length === 0) {
    return '';
  }

  const formatted = alerts
    .slice(0, 3)
    .map((alert, index) => {
      if (typeof alert === 'string') {
        return `${index + 1}. ${alert}`;
      }
      if (alert && typeof alert === 'object') {
        const label = alert.title || alert.type || `Alert ${index + 1}`;
        const desc = alert.message || alert.detail || '';
        return `${index + 1}. ${label}${desc ? ': ' + desc : ''}`;
      }
      return `${index + 1}. ${String(alert)}`;
    });

  return `Alerts & Reminders:\n${formatted.join('\n')}`;
}

function buildTasksSection(tasks) {
  if (!tasks || tasks.length === 0) {
    return '';
  }

  const formatted = tasks
    .slice(0, 3)
    .map((task, index) => `${index + 1}. ${typeof task === 'string' ? task : task.title || 'Pending task'}`);

  return `Open Tasks:\n${formatted.join('\n')}`;
}

function buildMemorySection(memoryEntries) {
  if (!Array.isArray(memoryEntries) || memoryEntries.length === 0) {
    return '';
  }

  const formatted = memoryEntries
    .slice(-MAX_MEMORY_ROWS)
    .map(entry => {
      const speaker = entry.role === 'user' ? 'Farmer' : 'KrushiAI';
      const content = typeof entry.content === 'string'
        ? entry.content
        : JSON.stringify(entry.content);
      return `- ${speaker}: ${content}`;
    });

  return `Recent Conversation:\n${formatted.join('\n')}`;
}

function buildContextSection(rawContext = {}) {
  if (!rawContext || typeof rawContext !== 'object') {
    return '';
  }

  const { memory, ...context } = rawContext;
  const sections = [];

  const profileSection = buildProfileSection(context.farmerProfile, context.userProfile?.profile);
  if (profileSection) sections.push(profileSection);

  const weatherSection = buildWeatherSection(context.weather || context.weatherData);
  if (weatherSection) sections.push(weatherSection);

  const mandiSection = buildMandiSection(context.mandi || context.mandiPrice);
  if (mandiSection) sections.push(mandiSection);

  const alertsSection = buildAlertsSection(context.alerts || context.reminders);
  if (alertsSection) sections.push(alertsSection);

  const tasksSection = buildTasksSection(context.tasks || context.pendingActions);
  if (tasksSection) sections.push(tasksSection);

  if (context.soil) {
    const soilSummary = typeof context.soil === 'string'
      ? context.soil
      : summarizeList(Object.entries(context.soil).map(([key, value]) => `${key}: ${value}`));
    if (soilSummary) {
      sections.push(`Soil & Field Notes:\n${soilSummary}`);
    }
  }

  if (context.location && typeof context.location === 'string') {
    sections.push(`Location Note:\n${context.location}`);
  }

  if (context.customNotes) {
    sections.push(`App Notes:\n${summarizeList(context.customNotes)}`);
  }

  return sections.length ? `Context Data:\n${sections.join('\n\n')}` : '';
}

function generateFarmerPrompt(language, userQuery, context = {}) {
  const resolvedLanguage = farmerPersonas[language] ? language : 'en';
  const persona = getFarmerPersona(resolvedLanguage);
  const guidelines = getResponseGuidelines(resolvedLanguage);
  const contextSection = buildContextSection(context);
  const memorySection = buildMemorySection(context.memory);
  const languageLabel = LANGUAGE_LABELS[resolvedLanguage] || LANGUAGE_LABELS.en;

  const promptParts = [
    persona,
    guidelines,
    contextSection,
    memorySection,
    `Farmer Query: ${userQuery}`,
    `Respond in ${languageLabel} using the structure above. Reference real context data explicitly and end with actionable next steps.`
  ].filter(Boolean);

  return promptParts.join('\n\n').trim();
}

module.exports = {
  getFarmerPersona,
  getResponseGuidelines,
  generateFarmerPrompt
};

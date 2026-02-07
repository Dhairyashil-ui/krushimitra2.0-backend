/**
 * Script to fetch available voices from ElevenLabs API
 * and find the voice IDs for Niraj and Monika voices
 */

const fetch = require('node-fetch');
require('dotenv').config();

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || "sk_c3b09109d2b1280789eec94db38ba2adf16ea7e6140c844e";

async function getAvailableVoices() {
  try {
    const response = await fetch('https://api.elevenlabs.io/v1/voices', {
      method: 'GET',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch voices: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    
    console.log('\n=== Available ElevenLabs Voices ===\n');
    
    // Look for Hindi voices specifically
    const hindiVoices = data.voices.filter(voice => 
      voice.name.toLowerCase().includes('hindi') || 
      voice.name.toLowerCase().includes('niraj')
    );
    
    if (hindiVoices.length > 0) {
      console.log('🎯 Found Hindi/Target Voices:');
      hindiVoices.forEach(voice => {
        console.log(`  • ${voice.name} (ID: ${voice.voice_id})`);
        console.log(`    Category: ${voice.category || 'N/A'}`);
        console.log(`    Description: ${voice.description || 'N/A'}`);
        console.log(`    Labels: ${Object.entries(voice.labels || {}).map(([k,v]) => `${k}:${v}`).join(', ')}`);
        console.log('');
      });
    }
    
    // Show all voices for reference
    console.log('\n📋 All Available Voices:');
    data.voices.forEach((voice, index) => {
      console.log(`${index + 1}. ${voice.name} (ID: ${voice.voice_id})`);
      if (voice.labels && voice.labels.accent) {
        console.log(`   Accent: ${voice.labels.accent}`);
      }
    });
    
    // Look for voices that might match our targets
    console.log('\n🔍 Searching for target voices...');
    const nirajVoice = data.voices.find(voice => 
      voice.name.toLowerCase().includes('niraj')
    );

    
    if (nirajVoice) {
      console.log(`✅ Found Niraj voice: ${nirajVoice.name} (ID: ${nirajVoice.voice_id})`);
    } else {
      console.log('❌ Niraj voice not found - might be a custom voice or different name');
    }
    


  } catch (error) {
    console.error('Error fetching voices:', error.message);
  }
}

// Run the script
if (require.main === module) {
  getAvailableVoices();
}

module.exports = { getAvailableVoices };
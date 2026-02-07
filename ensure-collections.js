const { ensureCollections } = require('./db');
const { logger } = require('./logger');

async function main() {
  try {
    logger.info('Ensuring collections exist');
    const result = await ensureCollections();
    logger.info('Collections ensured successfully', { result });
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    logger.error('Error ensuring collections', { error: error.message });
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
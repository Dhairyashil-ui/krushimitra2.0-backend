const { chromium } = require('playwright');
const { connectToDatabase } = require('../db');
const { logger } = require('../logger');

// The main landing page for Pune APMC prices
const APMC_URL = 'https://apmcpune.in/en/market-price-en/daily-market-price-en';

// Maps the site's categories to our DB's "Mandi" location string
const CATEGORY_TO_MANDI = {
    'Fruits Market': 'Pune APMC (Fruits)',
    'Jaggery Wholesale Market': 'Pune APMC (Jaggery)',
    'Flower Market': 'Pune APMC (Flowers)',
    'Banana and Betel Leaf Market': 'Pune APMC (Banana)',
    'Vegetable Market': 'Pune APMC (Vegetables)',
    'Onion Potato Market': 'Pune APMC (Onion/Potato)',
    'Sub Market Manjari': 'Manjari Sub Market',
    'Sub Market Mulshi': 'Mulshi Sub Market',
    'Sub Market Pimpri': 'Pimpri Sub Market',
    'Sub Market Khadki': 'Khadki Sub Market',
};

// Maps English/Marathi headers to our standard keys
const COLUMN_MAPPING = {
    'शेतमालाचे नाव': 'crop', // Crop Name
    'एकक': 'unit', // Unit (e.g., Quintal)
    'आवक': 'arrival', // Arrival qty
    'किमान भाव': 'minPrice', // Min Price
    'कमाल भाव': 'maxPrice', // Max Price
    'सरासरी भाव': 'modalPrice' // Average/Modal Price
};

async function scrapeCategory(page, url, categoryName) {
    const mandiName = CATEGORY_TO_MANDI[categoryName] || `Pune APMC (${categoryName})`;
    logger.info(`Scraping category: ${categoryName} -> ${mandiName} at ${url}`);

    try {
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

        // Wait for the table to appear (sometimes it takes a few seconds to load via JS)
        await page.waitForSelector('table', { timeout: 15000 });

        const prices = await page.evaluate(({ mandi, mapping }) => {
            const results = [];
            const tables = document.querySelectorAll('table');
            if (tables.length === 0) return results;

            // In case there are multiple tables, grab the first main one
            const table = tables[0];
            const headers = Array.from(table.querySelectorAll('thead th, tbody tr:first-child th, tbody tr:first-child td')).map(th => th.innerText.trim());

            const rows = table.querySelectorAll('tbody tr');

            // If the first row was headers, skip it
            let startRow = 0;
            if (rows.length > 0 && rows[0].querySelectorAll('th').length > 0) {
                startRow = 1;
            }

            for (let i = startRow; i < rows.length; i++) {
                const row = rows[i];
                const cells = Array.from(row.querySelectorAll('td')).map(td => td.innerText.trim());

                if (cells.length < 5) continue; // Skip empty/invalid rows

                // Standardize the row data
                // Order is usually: Name, Unit, Arrival, Min, Max, Average
                const cropName = cells[0] || 'Unknown';
                const unitRaw = cells[1] || 'Quintal'; // "क्विंटल"

                // Cleanup prices (remove commas, parse to int)
                const parsePrice = (str) => {
                    if (!str || str === '-') return 0;
                    const cleaned = str.replace(/[,₹Rs]/g, '').trim();
                    const val = parseInt(cleaned, 10);
                    return isNaN(val) ? 0 : val;
                };

                const resultObj = {
                    crop: cropName,
                    mandi: mandi,
                    unit: unitRaw.includes('क्विंटल') || unitRaw.includes('Quintal') ? 'Quintal' : unitRaw,
                    minPrice: parsePrice(cells[3]),
                    maxPrice: parsePrice(cells[4]),
                    modalPrice: parsePrice(cells[5] || cells[4]), // Fallback to max if average is missing
                    date: new Date().toISOString()
                };

                // Only add if we actually got prices
                if (resultObj.modalPrice > 0 || resultObj.maxPrice > 0) {
                    results.push(resultObj);
                }
            }
            return results;
        }, { mandi: mandiName, mapping: COLUMN_MAPPING });

        logger.info(`Successfully scraped ${prices.length} items from ${categoryName}`);
        return prices;
    } catch (err) {
        logger.error(`Error scraping category ${categoryName}: ${err.message}`);
        return [];
    }
}

async function scrapeAllPuneMandis() {
    logger.info('Starting daily Pune APMC Mandi scrape...');

    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    });

    const page = await context.newPage();
    let allScrapedPrices = [];

    try {
        // 1. Go to the main directory page
        await page.goto(APMC_URL, { waitUntil: 'domcontentloaded', timeout: 60000 });

        // 2. Find all category links (Fruits, Vegetables, Sub Markets)
        // Looking at the APMC site structure, the links are usually inside specific cards/buttons.
        const categories = await page.evaluate(() => {
            // Find all anchor tags that look like category links
            const links = Array.from(document.querySelectorAll('a'));
            const categoryLinks = [];

            links.forEach(link => {
                const href = link.href;
                const text = link.innerText.trim();

                // Filter out irrelevant links. We want the ones that go to /bajarbhav/ 
                // or have text matching our known categories
                if (href && (href.includes('daily-bajarbhav-dates') || href.includes('market'))) {
                    // Further refine to avoid header/footer links
                    if (text.length > 5 && text.length < 50) {
                        categoryLinks.push({ name: text, url: href });
                    }
                }
            });
            return categoryLinks;
        });

        logger.info(`Found ${categories.length} potential category links on main page.`);

        // Fallback: If dynamic extraction fails (e.g., website redesign), use known hardcoded paths to the categories.
        const runList = categories.length > 0 ? categories : [
            { name: 'Fruits Market', url: 'https://apmcpune.in/bajarbhav/daily-bajarbhav-dates/main-fruit' },
            { name: 'Vegetable Market', url: 'https://apmcpune.in/bajarbhav/daily-bajarbhav-dates/main-vegetable' },
            { name: 'Onion Potato Market', url: 'https://apmcpune.in/bajarbhav/daily-bajarbhav-dates/main-onion' },
            { name: 'Sub Market Manjari', url: 'https://apmcpune.in/bajarbhav/daily-bajarbhav-dates/main-manjari' },
            { name: 'Sub Market Mulshi', url: 'https://apmcpune.in/bajarbhav/daily-bajarbhav-dates/main-mulshi' },
            { name: 'Sub Market Pimpri', url: 'https://apmcpune.in/bajarbhav/daily-bajarbhav-dates/main-pimpri' },
            { name: 'Sub Market Khadki', url: 'https://apmcpune.in/bajarbhav/daily-bajarbhav-dates/main-khadaki' },
        ];

        // 3. Loop through categories and scrape the tables
        for (const cat of runList) {
            // Only process categories we recognized to avoid junk links
            if (!CATEGORY_TO_MANDI[cat.name] && !cat.name.includes('Market')) continue;

            const prices = await scrapeCategory(page, cat.url, cat.name);
            allScrapedPrices = allScrapedPrices.concat(prices);

            // Polite delay between requests
            await page.waitForTimeout(2000);
        }

        if (allScrapedPrices.length === 0) {
            throw new Error('No prices could be scraped. Website layout may have changed.');
        }

        logger.info(`Finished scraping. Total items collected: ${allScrapedPrices.length}`);

        // 4. Save to Database
        const dbClient = await connectToDatabase('write');
        const db = dbClient.db("KrushiMitraDB");
        const collection = db.collection('mandi_prices');

        const today = new Date();
        today.setHours(0, 0, 0, 0);

        let insertedCount = 0;
        let updatedCount = 0;

        // We do an Upsert (Update if exists, Insert if new) based on Crop + Mandi + Today's Date
        for (const record of allScrapedPrices) {
            // Find if we already inserted this crop at this mandi today
            const query = {
                crop: record.crop,
                mandi: record.mandi,
                date: { $gte: today } // any record from today
            };

            const existing = await collection.findOne(query);

            if (existing) {
                // Update the existing record for today
                await collection.updateOne({ _id: existing._id }, {
                    $set: {
                        minPrice: record.minPrice,
                        maxPrice: record.maxPrice,
                        modalPrice: record.modalPrice,
                        unit: record.unit,
                        date: new Date() // update timestamp
                    }
                });
                updatedCount++;
            } else {
                // Insert fresh record
                await collection.insertOne({
                    ...record,
                    date: new Date()
                });
                insertedCount++;
            }
        }

        logger.info(`DB Sync Complete: Inserted ${insertedCount}, Updated ${updatedCount} records.`);

    } catch (error) {
        logger.error(`Critical error during Mandi scraping: ${error.message}`);
    } finally {
        await browser.close();
    }
}

// Allow running directly from command line for testing
if (require.main === module) {
    scrapeAllPuneMandis().then(() => {
        console.log('Scraper script finished executing.');
        process.exit(0);
    });
}

module.exports = { scrapeAllPuneMandis };

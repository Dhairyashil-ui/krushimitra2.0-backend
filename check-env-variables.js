require('dotenv').config({ path: './SIH_APP-main/backend/.env' });

console.log("Checking environment variables...");
console.log("DB_READER_USER:", process.env.DB_READER_USER);
console.log("DB_READER_PASS:", process.env.DB_READER_PASS ? "[SET]" : "[NOT SET]");
console.log("DB_WRITER_USER:", process.env.DB_WRITER_USER);
console.log("DB_WRITER_PASS:", process.env.DB_WRITER_PASS ? "[SET]" : "[NOT SET]");
console.log("DB_ADMIN_USER:", process.env.DB_ADMIN_USER);
console.log("DB_ADMIN_PASS:", process.env.DB_ADMIN_PASS ? "[SET]" : "[NOT SET]");
console.log("CLUSTER_HOST:", process.env.CLUSTER_HOST);

// Check if all required variables are set
const requiredEnvVars = [
  'DB_READER_USER', 'DB_READER_PASS',
  'DB_WRITER_USER', 'DB_WRITER_PASS',
  'DB_ADMIN_USER', 'DB_ADMIN_PASS',
  'CLUSTER_HOST'
];

const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);
if (missingEnvVars.length > 0) {
  console.log("❌ Missing environment variables:", missingEnvVars);
} else {
  console.log("✅ All required environment variables are set");
}
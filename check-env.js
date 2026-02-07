require('dotenv').config();

console.log("Environment Variables:");
console.log("DB_USER:", process.env.DB_USER);
console.log("DB_PASS:", process.env.DB_PASS ? "[SET]" : "[NOT SET]");
console.log("CLUSTER_HOST:", process.env.CLUSTER_HOST);

console.log("\nConnection String:");
if (process.env.DB_USER && process.env.DB_PASS && process.env.CLUSTER_HOST) {
  const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
  console.log(uri);
} else {
  console.log("Cannot construct connection string due to missing environment variables");
}
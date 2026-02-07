const dotenv = require('dotenv');
dotenv.config();

console.log('Environment variables:');
console.log('DB_USER:', process.env.DB_USER);
console.log('DB_PASS:', process.env.DB_PASS ? '*** (set)' : 'NOT SET');
console.log('CLUSTER_HOST:', process.env.CLUSTER_HOST);

console.log('\nConnection URI:');
const uri = `mongodb+srv://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.CLUSTER_HOST}/KrushiMitraDB?retryWrites=true&w=majority`;
console.log(uri);
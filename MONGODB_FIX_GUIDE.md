# MongoDB Connection Issues Fix Guide

This guide will help you resolve the MongoDB connection issues identified in your application.

## Issues Identified

1. **Admin User Authentication Failure**: The admin user (`krushimitra_admin`) is failing authentication
2. **Duplicate Collections**: Both "farmers" and "Farmers" collections exist in the database

## Fix Steps

### 1. Fix Admin User Authentication

#### Step 1: Verify Admin User in MongoDB Atlas

1. Log in to [MongoDB Atlas](https://cloud.mongodb.com/)
2. Select your cluster
3. Go to "Database Access" in the left sidebar
4. Find the user `krushimitra_admin`
5. Verify the password matches what's in your `.env` file:
   ```
   DB_ADMIN_USER=krushimitra_admin
   DB_ADMIN_PASS=DMDTqN6v.E9xkXp
   ```

#### Step 2: If User Doesn't Exist or Password is Wrong

1. Click "Add New Database User"
2. Set Username: `krushimitra_admin`
3. Set Password: `DMDTqN6v.E9xkXp` (or update the .env file with a new password)
4. Set User Privileges:
   - Select "Atlas Admin" Built-in Role
   - Or select "Read and write to any database" + "Database Admin" for KrushiMitraDB
5. Click "Add User"

#### Step 3: Check IP Whitelist

1. In MongoDB Atlas, go to "Network Access" in the left sidebar
2. Make sure your current IP address is whitelisted
3. You can:
   - Add your current IP specifically
   - Temporarily add `0.0.0.0/0` for testing (NOT recommended for production)
   - Add your public IP range

### 2. Remove Duplicate Collections

After fixing the admin user authentication, run the duplicate collection removal script:

```bash
cd backend
node remove-duplicate-collection.js
```

This script will:
1. Connect using admin credentials
2. Identify duplicate "farmers" collections
3. Remove the incorrectly named one (Farmers with capital F)

### 3. Verify Fixes

Run the connection test again to verify everything is working:

```bash
node fix-mongodb-connection.js
```

You should see:
- ✅ Reader User: Success
- ✅ Writer User: Success
- ✅ Admin User: Success
- ✅ No duplicate farmers collections found

## Common Issues and Solutions

### Authentication Failed
- Double-check username and password in `.env` file
- Ensure the user exists in MongoDB Atlas with correct credentials
- Verify the user has appropriate roles assigned
- Check that your IP is whitelisted

### IP Not Whitelisted
- Add your current public IP to the whitelist
- Use `get-public-ip.js` script to find your public IP:
  ```bash
  node get-public-ip.js
  ```

### Connection Timeout
- Check your internet connection
- Verify MongoDB Atlas cluster is running
- Ensure no firewall is blocking the connection

## Testing After Fixes

After implementing the fixes, run these tests:

1. Test all connections:
   ```bash
   node fix-mongodb-connection.js
   ```

2. Test database operations:
   ```bash
   node db-test.js
   ```

3. Test logging functionality:
   ```bash
   node test-logging.js
   ```

## Security Recommendations

1. Use strong, unique passwords for all database users
2. Limit IP whitelist to only necessary addresses
3. Use different credentials for development and production
4. Regularly rotate database passwords
5. Monitor database access logs

## Need Help?

If you continue to have issues:

1. Check the detailed logs in the `backend/logs` directory
2. Verify all environment variables are correctly set
3. Ensure your MongoDB Atlas cluster is active and accessible
4. Contact MongoDB Atlas support if issues persist
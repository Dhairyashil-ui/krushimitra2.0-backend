# MongoDB Atlas Users Setup Guide

This document provides instructions for creating three MongoDB Atlas database users with least privileges for the KrushiMitra application.

## Prerequisites

1. Access to MongoDB Atlas account with admin privileges
2. Access to the KrushiMitraDB cluster

## Step 1: Create Read-Only User

1. Log in to MongoDB Atlas
2. Navigate to your cluster
3. Go to the "Database Access" section
4. Click "Add New Database User"
5. Fill in the following details:
   - **Username**: `krushimitra_reader`
   - **Password**: Generate a secure password
   - **Database User Privileges**: Select "Read any database" or create a custom role with read-only access to KrushiMitraDB
6. Click "Add User"

## Step 2: Create Read-Write User

1. In the same "Database Access" section, click "Add New Database User"
2. Fill in the following details:
   - **Username**: `krushimitra_writer`
   - **Password**: Generate a secure password
   - **Database User Privileges**: Select "Read and write to any database" or create a custom role with read-write access to KrushiMitraDB
3. Click "Add User"

## Step 3: Create Admin User

1. In the same "Database Access" section, click "Add New Database User"
2. Fill in the following details:
   - **Username**: `krushimitra_admin`
   - **Password**: Generate a secure password
   - **Database User Privileges**: Select "Atlas admin" or create a custom role with full admin access to KrushiMitraDB
3. Click "Add User"

## Step 4: Update Environment Variables

After creating these users, you'll need to update your [.env](file:///c%3A/Users/Dhairyashil/Downloads/SIH_APP-main/SIH_APP-main/backend/.env) file with the new credentials:

```env
# Read-only user for queries
DB_READER_USER=krushimitra_reader
DB_READER_PASS=your_reader_password

# Read-write user for modifications
DB_WRITER_USER=krushimitra_writer
DB_WRITER_PASS=your_writer_password

# Admin user for administrative tasks
DB_ADMIN_USER=krushimitra_admin
DB_ADMIN_PASS=your_admin_password

CLUSTER_HOST=krushimitradb.lw8eigi.mongodb.net
```

## Step 5: Restrict IP Access (Security Best Practice)

For enhanced security, restrict MongoDB Atlas access to only authorized IP addresses:

1. Navigate to the "Network Access" section in MongoDB Atlas
2. Remove the default `0.0.0.0/0` entry if present
3. Add your development machine IP address
4. Add your backend server IP address
5. For detailed instructions, see [MONGODB_IP_RESTRICTION_GUIDE.md](MONGODB_IP_RESTRICTION_GUIDE.md)

## Security Best Practices

1. Use strong, unique passwords for each user
2. Rotate passwords regularly
3. Limit IP access to only necessary addresses
4. Monitor user activity through MongoDB Atlas logs
5. Apply the principle of least privilege - only grant the minimum permissions needed

## User Permissions Summary

| User Type | Permissions | Use Cases |
|-----------|-------------|-----------|
| Reader | Read-only access to KrushiMitraDB | GET requests, data queries, reports |
| Writer | Read-write access to KrushiMitraDB | POST/PUT/DELETE requests, data modifications |
| Admin | Full admin access to KrushiMitraDB | Schema changes, index creation, user management |

Once these users are created and the environment variables are updated, the application will automatically use the appropriate user based on the operation being performed.
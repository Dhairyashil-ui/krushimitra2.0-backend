# MongoDB Atlas IP Restriction Guide

This document provides instructions for restricting MongoDB Atlas access to only your development machine IP and backend server IP, removing the open 0.0.0.0/0 access for enhanced security.

## Prerequisites

1. Access to MongoDB Atlas account with admin privileges
2. Access to the KrushiMitraDB cluster
3. Knowledge of your development machine's public IP address
4. Knowledge of your backend server's public IP address (if hosted externally)

## Finding Your IP Addresses

### For Development Machine:
1. **Windows**: Open Command Prompt and run:
   ```
   curl ifconfig.me
   ```
   or visit https://whatismyipaddress.com/

2. **macOS/Linux**: Open Terminal and run:
   ```
   curl ifconfig.me
   ```

### For Backend Server:
If your backend is hosted on a cloud service (AWS, Azure, GCP, etc.), check the service dashboard for the public IP address.

If your backend is hosted on the same machine as your development environment, you can use the same IP address.

## Step 1: Access MongoDB Atlas Network Configuration

1. Log in to MongoDB Atlas
2. Navigate to your cluster
3. Go to the "Network Access" section (in the left sidebar under Security)
4. You'll see a list of current IP address entries

## Step 2: Remove Open Access (0.0.0.0/0)

1. Find the entry with IP address `0.0.0.0/0` (if it exists)
2. Click the "Delete" button (trash can icon) next to this entry
3. Confirm deletion when prompted

## Step 3: Add Your Development Machine IP

1. Click the "Add IP Address" button
2. Choose one of these options:
   - **Allow access from anywhere** (NOT recommended for production)
   - **Add current IP address** (recommended for development)
   - **Add a specific IP address** (most secure)
3. If choosing "Add a specific IP address":
   - Enter your development machine's public IP address
   - Add a description like "Development Machine"
   - Set an expiration date if desired (optional)
4. Click "Confirm"

## Step 4: Add Your Backend Server IP

1. Click the "Add IP Address" button again
2. Choose "Add a specific IP address"
3. Enter your backend server's public IP address
4. Add a description like "Backend Server"
5. Set an expiration date if desired (optional)
6. Click "Confirm"

## Step 5: Verify Configuration

1. Ensure only your required IP addresses are listed
2. Confirm that `0.0.0.0/0` is no longer present
3. Test your application connections from both your development machine and backend server

## Security Best Practices

1. **Regular Review**: Periodically review and update your IP whitelist
2. **Specific IPs**: Always use specific IP addresses rather than IP ranges when possible
3. **Expiration Dates**: Set expiration dates for temporary access
4. **Monitoring**: Enable MongoDB Atlas alerts for unauthorized access attempts
5. **VPN**: Consider using a VPN for consistent IP addresses if your ISP provides dynamic IPs

## Handling Dynamic IPs

If your development machine or backend server has a dynamic IP address (common with residential ISPs), you have several options:

1. **VPN**: Use a VPN service that provides a static IP
2. **Dynamic DNS**: Use a dynamic DNS service that updates when your IP changes
3. **IP Range**: Use a broader IP range provided by your ISP (less secure)
4. **Regular Updates**: Manually update the IP whitelist when it changes

## Testing Connections

After updating IP restrictions, test connections from both environments:

1. **Development Environment**: Run your local development server and verify database connections work
2. **Backend Server**: Deploy and test your application on the backend server

If you encounter connection issues:
1. Verify the IP addresses in the MongoDB Atlas Network Access list
2. Check that your IP hasn't changed (especially for dynamic IPs)
3. Ensure your firewall settings allow outbound connections to MongoDB Atlas

## Emergency Access

For emergency situations where you need temporary access from a new IP:
1. Add the IP temporarily with a short expiration date
2. Remove it once your work is completed
3. Consider using MongoDB Atlas API to programmatically manage IP access for automated systems

## Troubleshooting

### Connection Refused Errors
- Verify your IP is in the whitelist
- Check that the IP address hasn't changed
- Ensure MongoDB Atlas cluster is running

### Authentication Errors
- Verify database user credentials
- Confirm user permissions are correct
- Check that the user exists and is active

### Timeout Errors
- Verify network connectivity to MongoDB Atlas
- Check firewall settings
- Ensure your ISP isn't blocking connections to MongoDB ports

By following this guide, you'll significantly improve the security of your MongoDB Atlas deployment by ensuring only authorized IP addresses can connect to your database cluster.
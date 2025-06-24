# MasterX AI Mentor System - Connection Configuration Guide

## Overview
This project uses a hybrid connection approach that supports both local development and production/preview environments seamlessly.

## Current Configuration Status

### Environment Detection
- **Local Development**: Automatically detected when hostname is localhost, 127.0.0.1, or local network IPs
- **Emergent Preview**: Automatically detected when hostname contains 'emergentagent.com'
- **Production**: Falls back to configured URLs or auto-detection

### Current Settings
- **Frontend .env**: Contains hardcoded preview URL (can be overridden)
- **Backend .env**: Contains MongoDB and API key configuration
- **Environment Detection**: Automatic with intelligent fallbacks

## Connection Priority Order

### For Local Development
1. `http://localhost:8001` (Primary)
2. `http://127.0.0.1:8001` (Fallback 1)
3. `http://localhost:3001` (Fallback 2)

### For Preview/Production
1. `REACT_APP_BACKEND_URL` (if configured)
2. Current hostname (auto-detection)
3. Current hostname with ports (fallback)

## How to Configure for Different Scenarios

### Scenario 1: Pure Local Development
```bash
# In frontend/.env
# Remove or comment out REACT_APP_BACKEND_URL
# The system will automatically use localhost:8001
```

### Scenario 2: Preview Environment (Emergent.sh)
```bash
# In frontend/.env
REACT_APP_BACKEND_URL=https://your-preview-url.emergentagent.com
# OR let the system auto-detect (recommended)
```

### Scenario 3: Production Deployment
```bash
# In frontend/.env
REACT_APP_BACKEND_URL=https://your-production-domain.com
```

### Scenario 4: Hybrid Local + Preview
```bash
# In frontend/.env
REACT_APP_BACKEND_URL=https://your-preview-url.emergentagent.com
# The system will:
# - Use localhost:8001 when running locally
# - Use the preview URL when accessing from preview environment
```

## Current Implementation Benefits

1. **No Manual Configuration Needed**: Works out of the box in most scenarios
2. **Intelligent Environment Detection**: Automatically adapts to the environment
3. **Robust Fallbacks**: Multiple URLs tested in priority order
4. **Connection Recovery**: Automatic retry and failover mechanisms
5. **Debug Tools**: Built-in connection status and testing components

## Files Modified

### Core Connection Files
- `frontend/src/config/environment.js` - Environment detection and URL configuration
- `frontend/src/utils/connectionManager.js` - Connection testing and failover
- `frontend/src/services/api.js` - API service with automatic connection management

### Debug Components
- `frontend/src/components/ConnectionStatus.js` - Real-time connection status
- `frontend/src/components/EnvironmentSettings.js` - Advanced configuration panel

## Testing Your Configuration

### Method 1: Use Built-in Debug Tools
1. Open the app in your browser
2. Look for the connection status widget in the bottom-right corner
3. Click "Debug" to see detailed connection information
4. Click "Test" to verify current connection
5. Click "Refresh" to force reconnection

### Method 2: Manual URL Testing
```bash
# Test backend health endpoint directly
curl http://localhost:8001/api/health
curl https://your-preview-url.emergentagent.com/api/health
```

### Method 3: Browser Console
```javascript
// Check current environment configuration
import { getEnvironmentConfig } from './src/config/environment';
console.log(getEnvironmentConfig());

// Test backend connection
import { testBackendHealth } from './src/utils/connectionManager';
testBackendHealth().then(console.log);
```

## Troubleshooting

### Problem: Connection fails in local development
**Solution**: 
1. Ensure backend is running: `sudo supervisorctl status backend`
2. Test backend directly: `curl http://localhost:8001/api/health`
3. Check backend logs: `tail -f /var/log/supervisor/backend.err.log`

### Problem: Connection fails in preview environment
**Solution**:
1. Verify the preview URL is correct in frontend/.env
2. Check if the preview environment is accessible
3. Ensure CORS is properly configured in the backend

### Problem: Mixed environment issues
**Solution**:
1. Clear browser cache and localStorage
2. Use the connection debug tools to verify environment detection
3. Check that the correct URLs are being tested

## Advanced Configuration

### Custom Environment Detection
You can modify `frontend/src/config/environment.js` to add custom environment detection logic:

```javascript
const isCustomEnvironment = () => {
  return window.location.hostname.includes('your-custom-domain.com');
};
```

### Custom Fallback URLs
Modify the environment configuration to add custom fallback URLs:

```javascript
config = {
  environment: 'custom',
  backendURL: 'https://primary.example.com',
  apiURL: 'https://primary.example.com/api',
  fallbacks: [
    'https://backup1.example.com',
    'https://backup2.example.com'
  ]
};
```

## Monitoring and Maintenance

### Connection Health Monitoring
The system automatically tracks:
- Connection success/failure rates
- Response times
- Fallback usage
- Environment detection accuracy

### Regular Maintenance
1. Update environment URLs when deploying to new environments
2. Test connection after major updates
3. Monitor connection logs for issues
4. Update fallback URLs as needed

## Security Considerations

1. **HTTPS in Production**: Always use HTTPS for production environments
2. **CORS Configuration**: Ensure backend CORS is properly configured
3. **API Key Security**: Keep API keys secure in backend environment variables
4. **URL Validation**: The system validates URLs before attempting connections

This configuration provides maximum flexibility while maintaining ease of use for both development and production scenarios.
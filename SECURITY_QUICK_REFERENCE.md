# 🔒 MASTERX SECURITY QUICK REFERENCE

Quick reference for security features and configurations.

---

## 🚀 QUICK START

### Enable HTTPS (Production)
```bash
# In .env file:
ENABLE_HSTS=true
```

### Configure CORS (Production)
```bash
# In .env file:
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

---

## 🔐 AUTHENTICATION

### Login Endpoint
```bash
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}

# Response:
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Access Admin Endpoints
```bash
GET /api/v1/admin/costs
Authorization: Bearer eyJ...

# Without token: 401 Unauthorized
# With regular user token: 403 Forbidden
# With admin token: 200 OK
```

---

## 🛡️ RATE LIMITING

### Current Limits
| Endpoint Type | Limit |
|--------------|-------|
| General API | 60 requests/min per IP |
| Auth Endpoints (login/register) | 10 requests/min per IP |
| Health Checks | Unlimited |

### Rate Limit Response
```bash
HTTP/1.1 429 Too Many Requests
Retry-After: 60
Content-Type: application/json

{
  "detail": "Too many requests. Please slow down."
}
```

---

## 🔒 SECURITY HEADERS

### Headers Added to All Responses
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'; ...
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), ...
Cache-Control: no-store, no-cache, must-revalidate, private
```

### HSTS (HTTPS Only)
```http
# When ENABLE_HSTS=true in .env:
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

---

## 🚫 BRUTE FORCE PROTECTION

### Account Lockout Rules
- **5 failed password attempts** → Account locked for 15 minutes
- **10 rapid login attempts** from same IP → Rate limited (429)
- Failed attempts tracked in database
- Automatic unlock after lockout period

### Lockout Response
```bash
HTTP/1.1 423 Locked
Content-Type: application/json

{
  "detail": "Account temporarily locked. Please try again later."
}
```

---

## 🔑 PASSWORD SECURITY

### Requirements
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

### Hashing
- Algorithm: Bcrypt
- Rounds: 12
- Secure by design

---

## 🌐 CORS CONFIGURATION

### Development (Current)
```python
CORS_ORIGINS=*  # Allows all origins
```

### Production (Recommended)
```python
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### Headers
- Allow-Credentials: true
- Allow-Methods: GET, POST, PUT, DELETE, PATCH
- Allow-Headers: *

---

## 📊 MONITORING & LOGGING

### Security Events Logged
- ✅ All login attempts (success/failure)
- ✅ Failed authentication attempts
- ✅ Rate limit violations
- ✅ Account lockouts
- ✅ Token validation failures
- ✅ Admin endpoint access

### Log Format
```json
{
  "timestamp": "2025-10-19T00:55:52Z",
  "level": "WARNING",
  "event": "rate_limit_exceeded",
  "ip": "192.168.1.100",
  "endpoint": "/api/auth/login",
  "details": "51/10 requests in window"
}
```

---

## 🧪 TESTING SECURITY

### Run Security Tests
```bash
cd /app/backend
python test_security_comprehensive.py
```

### Check Specific Feature
```bash
# Test authentication
curl -X GET http://localhost:8001/api/v1/admin/costs
# Should return: 401 Unauthorized

# Test rate limiting
for i in {1..100}; do 
  curl -s http://localhost:8001/api/v1/providers > /dev/null
done
# Should eventually return: 429 Too Many Requests

# Test security headers
curl -I http://localhost:8001/api/health | grep -E "X-Content|X-Frame"
# Should see security headers
```

---

## ⚙️ CONFIGURATION FILES

### Security Settings Location
```
/app/backend/.env                    # Main configuration
/app/backend/config/settings.py      # Settings validation
/app/backend/middleware/
  ├── auth.py                        # JWT authentication
  ├── brute_force.py                 # Account lockout (unused*)
  ├── security_headers.py            # Security headers
  └── simple_rate_limit.py           # Rate limiting
```

*Note: `brute_force.py` exists but rate limiting is handled by `simple_rate_limit.py`

---

## 🔧 TROUBLESHOOTING

### Issue: "401 Unauthorized" on admin endpoints
**Solution:** Include valid JWT token in Authorization header
```bash
Authorization: Bearer YOUR_TOKEN_HERE
```

### Issue: "429 Too Many Requests"
**Solution:** Wait 60 seconds or increase rate limits in configuration

### Issue: CORS errors in browser
**Solution:** Add your domain to CORS_ORIGINS in .env
```bash
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Issue: "423 Account Locked"
**Solution:** Wait 15 minutes or reset failed attempts in database

---

## 📚 RELATED DOCUMENTATION

- **Full Security Audit:** `/app/SECURITY_AUDIT_FINAL_REPORT.md`
- **Original Testing Report:** `/app/REAL_WORLD_TESTING_REPORT.md`
- **Implementation Guide:** `/app/SECURITY_FIXES_IMPLEMENTATION.md`
- **AGENTS.md:** Security principles and standards

---

## 🆘 SECURITY CONTACT

If you discover a security vulnerability:
1. Do NOT open a public issue
2. Document the vulnerability details
3. Contact the development team directly
4. Allow time for patching before public disclosure

---

**Last Updated:** October 19, 2025  
**Version:** 1.0  
**Status:** Production Ready ✅

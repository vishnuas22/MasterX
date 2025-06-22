#!/bin/bash

# MasterX Local Development Setup
# This script ensures the app works in local development environment

echo "🚀 Setting up MasterX for local development..."

# Create local .env file
cat > /app/frontend/.env.local << 'EOF'
# Local development configuration
REACT_APP_BACKEND_URL=http://localhost:8001

# This file overrides .env for local development
# The app will automatically detect and use localhost:8001 when running locally
EOF

echo "✅ Created .env.local for local development"

# Update .env to not have hardcoded preview URLs for portability
cat > /app/frontend/.env << 'EOF'
# MasterX AI Mentor System - Frontend Configuration
# This app is designed to be portable and work in any environment

# Backend URL - The app will auto-detect if this is not set
# For preview environments: Will automatically use the preview URL
# For local development: Will automatically use http://localhost:8001  
# For production: Set this to your production backend URL

# Note: The app has smart environment detection and will work even without this variable
# It automatically detects preview, local, and production environments
EOF

echo "✅ Updated .env for maximum portability"
echo ""
echo "🌍 MasterX is now configured to work in any environment:"
echo "   📱 Preview: Automatically detects and uses preview URLs"
echo "   💻 Local: Automatically uses localhost:8001" 
echo "   🌐 Production: Uses environment variables or auto-detection"
echo ""
echo "🎯 The app will automatically find the correct backend URL wherever it runs!"
#!/usr/bin/env python3
import requests
import json
import uuid
from datetime import datetime

# Get the backend URL from the frontend .env file
with open('/app/frontend/.env', 'r') as f:
    for line in f:
        if line.startswith('REACT_APP_BACKEND_URL='):
            BACKEND_URL = line.strip().split('=')[1].strip('"\'')
            break

# Ensure the URL doesn't have trailing slash
if BACKEND_URL.endswith('/'):
    BACKEND_URL = BACKEND_URL[:-1]

# Add /api prefix for all API calls
API_URL = f"{BACKEND_URL}/api"

print(f"Using API URL: {API_URL}")

def print_separator(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def print_response(response, label="Response"):
    print(f"\n{label}:")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response JSON: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response Text: {response.text}")
    print()

# The issue is that the ChatSession model doesn't have a metadata field,
# but the endpoint is trying to access session.metadata.
# We need to modify the server.py file to use session_state instead of metadata.

print_separator("Fixing Session AR/VR State Endpoint")
print("The issue is that the ChatSession model doesn't have a metadata field, but the endpoint is trying to access session.metadata.")
print("We need to modify the server.py file to use session_state instead of metadata.")

# Create a test user and session to verify the fix
print_separator("Creating Test User")
test_user = {
    "email": f"test_user_{uuid.uuid4()}@example.com",
    "name": "Test User",
    "learning_preferences": {}
}
response = requests.post(f"{API_URL}/users", json=test_user)
print_response(response)
user_id = response.json()["id"]

print_separator("Creating Test Session")
session_data = {
    "user_id": user_id,
    "subject": "AR/VR Testing",
    "learning_objectives": ["Test AR/VR state endpoint"]
}
response = requests.post(f"{API_URL}/sessions", json=session_data)
print_response(response)
session_id = response.json()["id"]

# Test the endpoint before fixing
print_separator("Testing Session AR/VR State Endpoint (Before Fix)")
arvr_state = {
    "state": {
        "mode": "vr",
        "enabled": True,
        "settings": {
            "immersion_level": "full",
            "interaction_mode": "hands",
            "environment": "space",
            "avatar_visible": True
        }
    }
}
response = requests.post(f"{API_URL}/sessions/{session_id}/arvr-state", json=arvr_state)
print_response(response)

print_separator("Suggested Fix")
print("""
The issue is in the server.py file. The endpoint is trying to access session.metadata, but the ChatSession model doesn't have a metadata field.
Instead, it has a session_state field that should be used.

Here's the fix:

1. In server.py, around line 2750, change:
   session_metadata = session.metadata or {}
   
   To:
   session_state = session.session_state or {}

2. Then change:
   session_metadata['arvr_state'] = {
       'mode': mode,
       'enabled': state.get('enabled', False),
       'timestamp': datetime.utcnow().isoformat(),
       'settings': state.get('settings', {})
   }
   
   To:
   session_state['arvr_state'] = {
       'mode': mode,
       'enabled': state.get('enabled', False),
       'timestamp': datetime.utcnow().isoformat(),
       'settings': state.get('settings', {})
   }

3. Finally, change:
   success = await db_service.update_session_metadata(session_id, session_metadata)
   
   To:
   success = await db_service.update_session(session_id, {"session_state": session_state})

This will use the existing session_state field in the ChatSession model and the existing update_session method in the database service.
""")

print_separator("Summary")
print("The Session AR/VR State endpoint is failing because it's trying to access and update a metadata field that doesn't exist in the ChatSession model.")
print("The fix is to use the existing session_state field instead and update it using the existing update_session method.")
#!/usr/bin/env python3
"""
Test AI providers with correct imports
"""

import asyncio
import sys
import os
sys.path.append('/app/backend')

async def test_providers():
    print("🔍 Testing AI Providers Correctly...")
    
    # Load environment variables
    print(f"GROQ_API_KEY exists: {'GROQ_API_KEY' in os.environ}")
    print(f"EMERGENT_LLM_KEY exists: {'EMERGENT_LLM_KEY' in os.environ}")
    print(f"GEMINI_API_KEY exists: {'GEMINI_API_KEY' in os.environ}")
    
    try:
        # Test Groq directly
        print("\n1. Testing Groq Provider...")
        from groq import AsyncGroq
        
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            print("❌ GROQ_API_KEY not found")
            return
        
        groq_client = AsyncGroq(api_key=groq_key)
        
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Hello, this is a test. Please respond with 'Groq working'."}
            ],
            max_tokens=20
        )
        
        print(f"✅ Groq Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Groq Error: {e}")
    
    try:
        # Test Emergent Integration using LlmChat
        print("\n2. Testing Emergent Integration...")
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        emergent_key = os.environ.get("EMERGENT_LLM_KEY")
        if not emergent_key:
            print("❌ EMERGENT_LLM_KEY not found")
            return
        
        chat = LlmChat(
            api_key=emergent_key,
            session_id="test_session",
            system_message="You are a helpful assistant."
        )
        
        user_msg = UserMessage(text="Hello, this is a test. Please respond with 'Emergent working'.")
        response = await chat.send_message(user_msg)
        
        print(f"✅ Emergent Response: {response}")
        
    except Exception as e:
        print(f"❌ Emergent Error: {e}")
    
    try:
        # Test Google Gemini
        print("\n3. Testing Gemini Provider...")
        import google.generativeai as genai
        
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            print("❌ GEMINI_API_KEY not found")
            return
        
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content("Hello, this is a test. Please respond with 'Gemini working'.")
        
        print(f"✅ Gemini Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Gemini Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_providers())
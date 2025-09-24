"""
🔍 DEBUG EMOTION DETECTION - Check why struggling learner doesn't show high cognitive load
"""

import asyncio

async def debug_struggling_learner():
    """Debug why struggling learner shows 0.0000 cognitive load"""
    
    from quantum_intelligence.services.emotional.authentic_emotion_core_v9 import (
        AuthenticBehavioralAnalyzer
    )
    
    behavioral_analyzer = AuthenticBehavioralAnalyzer()
    await behavioral_analyzer.initialize()
    
    # Test struggling learner profile
    print("🔍 DEBUGGING STRUGGLING LEARNER COGNITIVE LOAD")
    print("=" * 60)
    
    struggling_profile = {
        "typical_response_length": 15,
        "typical_response_time": 80.0,
        "typical_error_rate": 0.4,  # Very high error rate
        "typical_session_duration": 600,
        "typical_interaction_quality": 0.3
    }
    
    user_id = "debug_struggling"
    
    for session in range(3):
        print(f"\n📊 Session {session + 1}:")
        
        # Create high-error behavioral data
        behavioral_data = {
            "response_length": struggling_profile["typical_response_length"],
            "response_time": struggling_profile["typical_response_time"],
            "error_rate": struggling_profile["typical_error_rate"],
            "session_duration": struggling_profile["typical_session_duration"],
            "interaction_quality_score": struggling_profile["typical_interaction_quality"]
        }
        
        print(f"  Input data: {behavioral_data}")
        
        cognitive_analysis = await behavioral_analyzer.analyze_cognitive_load_indicators(
            user_id, behavioral_data
        )
        
        print(f"  Cognitive analysis: {cognitive_analysis}")
        
        # Check user baseline
        baseline = behavioral_analyzer.user_baselines.get(user_id, {})
        print(f"  User baseline: {baseline}")
        
        await behavioral_analyzer.update_user_baseline(user_id, behavioral_data)

if __name__ == "__main__":
    asyncio.run(debug_struggling_learner())
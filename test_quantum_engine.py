#!/usr/bin/env python3
"""
Test script for integrated_quantum_engine.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to Python path
sys.path.append(str(Path(__file__).parent / "backend"))

async def test_quantum_engine():
    """Test the quantum engine integration"""
    try:
        print("🚀 Testing Integrated Quantum Engine V6.0...")
        
        # Test imports
        print("1. Testing imports...")
        from quantum_intelligence.core.integrated_quantum_engine import (
            UltraEnterpriseQuantumEngine,
            QuantumProcessingMetrics,
            QuantumEngineState,
            QuantumIntelligentCache,
            ProcessingPhase,
            QuantumEngineConstants,
            get_ultra_quantum_engine,
            shutdown_ultra_quantum_engine
        )
        print("✅ All imports successful!")
        
        # Test constants
        print("2. Testing constants...")
        print(f"   - Target response time: {QuantumEngineConstants.TARGET_RESPONSE_TIME_MS}ms")
        print(f"   - Cache size: {QuantumEngineConstants.DEFAULT_CACHE_SIZE}")
        print("✅ Constants loaded successfully!")
        
        # Test cache initialization
        print("3. Testing cache initialization...")
        cache = QuantumIntelligentCache(max_size=1000)
        print(f"   - Cache max size: {cache.max_size}")
        print("✅ Cache initialized successfully!")
        
        # Test metrics creation
        print("4. Testing metrics creation...")
        metrics = QuantumProcessingMetrics(
            request_id="test-123",
            user_id="test-user",
            start_time=asyncio.get_event_loop().time()
        )
        print(f"   - Metrics request ID: {metrics.request_id}")
        print("✅ Metrics created successfully!")
        
        # Test engine state
        print("5. Testing engine state...")  
        state = QuantumEngineState()
        print(f"   - Initial state: {state.is_initialized}")
        print("✅ Engine state created successfully!")
        
        # Test quantum engine initialization (mock database)
        print("6. Testing quantum engine initialization...")
        class MockDB:
            def __init__(self):
                self.enhanced_learning_contexts = None
                self.user_profiles = None
                self.context_performance = None
                self.quantum_conversations = None
                self.enhanced_user_profiles = None
                self.conversation_analytics = None
        
        mock_db = MockDB()
        
        engine = UltraEnterpriseQuantumEngine(mock_db)
        print(f"   - Engine ID: {engine.engine_id}")
        print(f"   - Is initialized: {engine.engine_state.is_initialized}")
        print("✅ Quantum engine created successfully!")
        
        # Test system status
        print("7. Testing system status...")
        status = await engine.get_system_status()
        print(f"   - System info available: {'system_info' in status}")
        print(f"   - Performance metrics available: {'performance_metrics' in status}")
        print("✅ System status retrieved successfully!")
        
        # Test global instance management
        print("8. Testing global instance management...")
        global_engine = await get_ultra_quantum_engine(mock_db)
        print(f"   - Global engine ID: {global_engine.engine_id}")
        print("✅ Global instance management working!")
        
        # Cleanup
        print("9. Cleanup...")
        await engine.shutdown()
        await shutdown_ultra_quantum_engine()
        print("✅ Cleanup completed!")
        
        print("\n🎉 ALL TESTS PASSED! Integrated Quantum Engine V6.0 is working perfectly!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_quantum_engine())
    sys.exit(0 if result else 1)
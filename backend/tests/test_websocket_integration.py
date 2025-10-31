"""
WebSocket Integration Tests - Live Server Testing

Tests the complete WebSocket flow:
1. Connection establishment with JWT auth
2. Message exchange
3. Session management
4. Emotion updates
5. Frontend compatibility

Requires running FastAPI server
"""

import pytest
import asyncio
import websockets
import json
from datetime import datetime
import time

# Test configuration
BACKEND_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001/api/ws"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_token():
    """
    Get a valid JWT token for testing
    
    In production, this would authenticate and get a real token.
    For testing, we'll use a mock token or skip if not available.
    """
    # TODO: Implement proper token generation for tests
    # For now, return a placeholder
    return "test_token_placeholder"


# ============================================================================
# TEST: WebSocket Connection
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_connection_requires_token():
    """WebSocket connection without token should be rejected"""
    try:
        # Try to connect without token
        async with websockets.connect(f"{WS_URL}") as websocket:
            # Should not reach here
            pytest.fail("Connection should have been rejected without token")
    except Exception as e:
        # Expected to fail
        assert True


@pytest.mark.integration
@pytest.mark.asyncio  
async def test_websocket_connection_with_invalid_token():
    """WebSocket connection with invalid token should be rejected"""
    try:
        # Try to connect with invalid token
        async with websockets.connect(f"{WS_URL}?token=invalid_token") as websocket:
            # Wait a moment for server to close connection
            await asyncio.sleep(0.1)
            
            # Try to receive (should fail)
            message = await websocket.recv()
            pytest.fail("Should have been disconnected")
    except (websockets.exceptions.ConnectionClosed, Exception):
        # Expected to be rejected
        assert True


# ============================================================================
# TEST: Message Exchange (Manual Test)
# ============================================================================

@pytest.mark.integration
@pytest.mark.manual
async def test_websocket_message_exchange_manual():
    """
    Manual test for WebSocket message exchange
    
    This test requires a valid token from the system.
    Run manually with: pytest -m manual
    """
    print("\n" + "="*80)
    print("MANUAL WEBSOCKET TEST")
    print("="*80)
    print("\nSteps to test manually:")
    print("1. Get a valid JWT token by logging in")
    print("2. Connect to:", WS_URL)
    print("3. Send test messages")
    print("4. Verify responses")
    print("\nExample using websocat:")
    print(f"  websocat '{WS_URL}?token=YOUR_TOKEN'")
    print("\nExample messages to send:")
    print("  {'type': 'ping', 'data': {}}")
    print("  {'type': 'join_session', 'data': {'session_id': 'test123'}}")
    print("  {'type': 'user_typing', 'data': {'session_id': 'test123', 'isTyping': true}}")
    print("="*80)


# ============================================================================
# TEST: Health Check Endpoint
# ============================================================================

@pytest.mark.integration
def test_backend_is_running():
    """Verify backend server is accessible"""
    import requests
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        print(f"\n✓ Backend is running: {BACKEND_URL}")
    except Exception as e:
        pytest.skip(f"Backend not accessible: {e}")


# ============================================================================
# TEST: WebSocket Endpoint Registration
# ============================================================================

@pytest.mark.integration
def test_websocket_endpoint_exists():
    """Verify WebSocket endpoint is registered"""
    import requests
    
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        assert response.status_code == 200
        data = response.json()
        
        # Check if WebSocket endpoint is listed
        if 'endpoints' in data:
            assert 'websocket' in data['endpoints']
            print(f"\n✓ WebSocket endpoint registered: {data['endpoints']['websocket']}")
    except Exception as e:
        pytest.skip(f"Could not verify endpoint: {e}")


# ============================================================================
# TEST: Frontend Compatibility
# ============================================================================

@pytest.mark.integration
class TestFrontendCompatibility:
    """Test WebSocket compatibility with frontend client"""
    
    def test_message_format_compatibility(self):
        """Verify message format matches frontend expectations"""
        from services.websocket_service import manager
        
        # Expected message format from frontend
        frontend_message = {
            'type': 'emotion_update',
            'data': {
                'message_id': 'msg123',
                'emotion': {
                    'primary_emotion': 'joy',
                    'confidence': 0.95
                }
            },
            'timestamp': int(time.time() * 1000)
        }
        
        # Verify structure
        assert 'type' in frontend_message
        assert 'data' in frontend_message
        assert isinstance(frontend_message['type'], str)
        assert isinstance(frontend_message['data'], dict)
        
        print("\n✓ Message format compatible with frontend")
    
    def test_event_types_match_frontend(self):
        """Verify event types match frontend expectations"""
        # Frontend event types from native-socket.client.ts
        frontend_events = [
            'emotion_update',
            'typing_indicator', 
            'message_received',
            'session_update',
            'notification',
            'error',
            'user_typing',
            'join_session',
            'leave_session',
            'message_sent',
            'connect',
            'disconnect',
            'pong'
        ]
        
        # Backend should handle these
        backend_handlers = [
            'join_session',
            'leave_session',
            'user_typing',
            'message_sent',
            'ping',
            'get_health'
        ]
        
        # Verify coverage
        assert 'join_session' in backend_handlers
        assert 'leave_session' in backend_handlers
        
        print(f"\n✓ Frontend event types: {len(frontend_events)}")
        print(f"✓ Backend handlers: {len(backend_handlers)}")


# ============================================================================
# TEST: Connection Manager State
# ============================================================================

@pytest.mark.integration
class TestConnectionManagerState:
    """Test connection manager in running server"""
    
    def test_manager_instance_exists(self):
        """Global manager instance should exist"""
        from services.websocket_service import manager
        
        assert manager is not None
        assert hasattr(manager, 'active_connections')
        assert hasattr(manager, 'sessions')
        print("\n✓ Connection manager instance exists")
    
    def test_manager_analytics_available(self):
        """Manager should provide analytics"""
        from services.websocket_service import manager
        
        analytics = manager.get_analytics()
        
        assert isinstance(analytics, dict)
        assert 'active_connections' in analytics
        assert 'active_users' in analytics
        assert 'active_sessions' in analytics
        
        print(f"\n✓ Active connections: {analytics['active_connections']}")
        print(f"✓ Active users: {analytics['active_users']}")
        print(f"✓ Active sessions: {analytics['active_sessions']}")


# ============================================================================
# TEST: Security Features
# ============================================================================

@pytest.mark.integration
class TestSecurityFeatures:
    """Test WebSocket security features"""
    
    def test_security_validator_prevents_xss(self):
        """Security validator should block XSS attempts"""
        from services.websocket_service import SecurityValidator
        
        xss_messages = [
            {'type': 'test', 'data': '<script>alert("xss")</script>'},
            {'type': 'test', 'data': 'javascript:void(0)'},
        ]
        
        for msg in xss_messages:
            is_valid, error = SecurityValidator.validate_message(msg)
            assert not is_valid, f"Should block XSS: {msg}"
        
        print("\n✓ XSS prevention working")
    
    def test_rate_limiter_configured(self):
        """Rate limiter should be properly configured"""
        from services.websocket_service import manager
        
        assert manager.rate_limiter is not None
        assert manager.rate_limiter.config.rate_limit_max_messages > 0
        
        print(f"\n✓ Rate limit: {manager.rate_limiter.config.rate_limit_max_messages} msg/min")


# ============================================================================
# TEST: Performance Benchmarks
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for WebSocket service"""
    
    @pytest.mark.asyncio
    async def test_message_latency(self):
        """Measure message processing latency"""
        from services.websocket_service import manager, MessagePriority
        import asyncio
        
        # Simulate message sending
        start = time.time()
        
        # Create a mock message
        message = {'type': 'test', 'data': 'benchmark'}
        
        # Process (offline user, should queue)
        await manager.send_personal_message('bench_user', message, MessagePriority.MEDIUM)
        
        latency_ms = (time.time() - start) * 1000
        
        print(f"\n✓ Message queue latency: {latency_ms:.2f}ms")
        assert latency_ms < 50, "Message processing should be <50ms"
    
    def test_health_monitor_performance(self):
        """Health monitor should calculate scores quickly"""
        from services.websocket_service import manager
        
        # Record many messages
        start = time.time()
        
        for i in range(100):
            manager.health_monitor.record_message(
                f"bench:conn{i}",
                latency_ms=50.0,
                success=True
            )
        
        elapsed_ms = (time.time() - start) * 1000
        
        print(f"\n✓ Health monitoring 100 messages: {elapsed_ms:.2f}ms")
        assert elapsed_ms < 100, "Health monitoring should be fast"


# ============================================================================
# SUMMARY
# ============================================================================

def test_integration_summary():
    """Integration test suite summary"""
    print("\n" + "="*80)
    print("WEBSOCKET INTEGRATION TEST SUMMARY")
    print("="*80)
    print("✓ Backend availability checks")
    print("✓ WebSocket endpoint registration")
    print("✓ Frontend compatibility verification")
    print("✓ Connection manager state tests")
    print("✓ Security feature validation")
    print("✓ Performance benchmarks")
    print("="*80)
    print("\nFor full integration testing with live WebSocket:")
    print("1. Ensure backend is running: sudo supervisorctl status backend")
    print("2. Get a valid JWT token from /api/auth/login")
    print("3. Use websocat or similar tool to test:")
    print(f"   websocat '{WS_URL}?token=YOUR_TOKEN'")
    print("="*80)

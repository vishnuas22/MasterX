# 🚀 PHASE 2 IMPLEMENTATION PLAN
## Dynamic AI Routing & Benchmarking System

**Created:** January 1, 2025  
**Status:** Ready to Implement  
**Priority:** HIGH - Core Intelligence Enhancement

---

## 📊 CURRENT STATUS ASSESSMENT

### ✅ Phase 1 Complete (What We Have):
1. **Core Models** (`core/models.py`) - ✅ COMPLETE (341 lines)
   - All Pydantic V2 models defined
   - MongoDB schema ready
   - Benchmark & Health models included
   - UUID-based IDs
   - Full type hints

2. **AI Provider System** (`core/ai_providers.py`) - ✅ BASIC COMPLETE (382 lines)
   - Auto-discovery from .env ✅
   - Universal provider interface ✅
   - 5 providers supported (Groq, Emergent, Gemini, OpenAI, Anthropic) ✅
   - Simple fallback mechanism ✅
   - **MISSING:** Benchmarking, Smart Routing, Circuit Breaker

3. **MasterX Engine** (`core/engine.py`) - ✅ BASIC COMPLETE (183 lines)
   - Emotion detection integrated ✅
   - Basic AI response generation ✅
   - Emotion-aware prompting ✅
   - Cost tracking ✅
   - **MISSING:** Context management, Adaptive learning integration

4. **Server** (`server.py`) - ✅ OPERATIONAL (336 lines)
   - FastAPI with MongoDB ✅
   - /api/v1/chat endpoint ✅
   - Health checks ✅
   - Cost dashboard ✅
   - Provider listing ✅

5. **Context Manager** (`core/context_manager.py`) - ⚠️ STUB (Only comments)
6. **Adaptive Learning** (`core/adaptive_learning.py`) - ⚠️ STUB (Only comments)

### 🎯 Phase 2 Goals:
Build the **Dynamic AI Routing System** with continuous benchmarking and intelligent provider selection.

---

## 📋 PHASE 2 COMPONENTS TO BUILD

Based on `4.DYNAMIC_AI_ROUTING_SYSTEM.md` and `3.MASTERX_COMPREHENSIVE_PLAN.md`:

### Component 1: **Benchmark Engine** 🔬
**File:** `core/benchmarking.py` (NEW)  
**Priority:** CRITICAL - Build First  
**Est. Lines:** 800-1000  
**Dependencies:** `core/models.py`, `core/ai_providers.py`

**What it does:**
- Runs automated tests across all providers
- Tests 6 categories: coding, math, research, language, empathy, general
- Measures quality (0-100), speed (ms), cost ($)
- Calculates weighted final scores
- Stores results in MongoDB
- Runs continuously (every 1-12 hours)

**Key Classes:**
```python
class BenchmarkEngine:
    async def run_benchmarks(categories, providers) -> Dict
    async def _run_category_tests(provider, category, tests) -> List[TestResult]
    async def _evaluate_quality(response, test) -> float
    def _calculate_scores(provider, category, results, weights) -> BenchmarkResult
    async def schedule_benchmarks(interval_hours=1)
    async def get_latest_benchmarks(category, max_age_hours=24) -> List[BenchmarkResult]

class BenchmarkTest:
    test_id: str
    prompt: str
    expected_keywords: List[str]
    expected_answer: Optional[str]
    min_length: Optional[int]
    scoring_rubric: Dict
```

**Test Suite Structure:**
```python
BENCHMARK_CATEGORIES = {
    "coding": {
        "tests": [
            {"prompt": "Explain merge sort", "expected_keywords": [...]},
            {"prompt": "Debug this code", "expected_keywords": [...]},
            {"prompt": "Write reverse string function", "must_contain": ["def", "return"]}
        ],
        "weights": {"quality": 0.5, "speed": 0.3, "cost": 0.2}
    },
    "math": {...},
    "research": {...},
    "language": {...},
    "empathy": {...},
    "general": {...}
}
```

---

### Component 2: **Smart Router** 🧠
**File:** `core/smart_router.py` (NEW)  
**Priority:** CRITICAL - Build Second  
**Est. Lines:** 400-600  
**Dependencies:** `core/benchmarking.py`, `core/models.py`

**What it does:**
- Selects best provider based on:
  1. Session continuity (stick with provider for same topic)
  2. Task category detection
  3. Latest benchmark scores
  4. Emotion state (frustrated → empathy provider)
  5. Circuit breaker status
- Maintains provider consistency within topics
- Real-time category detection

**Key Classes:**
```python
class SmartRouter:
    async def select_provider(
        message: str,
        emotion_state: EmotionResult,
        session_id: str,
        context: ConversationContext
    ) -> str
    
    async def _detect_category(message, emotion, context) -> str
    async def _detect_topic_change(new_msg, current_topic, context) -> bool
    def _get_default_provider() -> str

class TaskCategoryDetector:
    async def detect(message: str) -> str  # Returns: coding, math, research, etc.
    
    # Uses keyword matching + ML (future enhancement)
    CATEGORY_KEYWORDS = {
        "coding": ["code", "function", "algorithm", "debug", "python"],
        "math": ["solve", "calculate", "equation", "derivative"],
        "research": ["analyze", "research", "compare", "study"],
        "language": ["translate", "grammar", "correct", "sentence"],
        "empathy": (auto-detected from emotion)
    }
```

---

### Component 3: **Session Manager** 📝
**File:** `core/session_manager.py` (NEW)  
**Priority:** HIGH - Build Third  
**Est. Lines:** 200-300  
**Dependencies:** `utils/database.py`

**What it does:**
- Tracks provider assignments per session
- Maintains topic continuity
- Stores session metadata
- Prevents unnecessary provider switching

**Key Classes:**
```python
class SessionManager:
    async def get_session_info(session_id: str) -> Optional[Dict]
    async def update_session(session_id: str, provider: str, topic: str)
    async def detect_topic_change(session_id: str, new_message: str) -> bool
    async def get_session_provider(session_id: str) -> Optional[str]
```

**MongoDB Schema:**
```python
{
    "session_id": "uuid",
    "current_provider": "claude",
    "current_topic": "coding",
    "topic_history": ["math", "coding"],
    "provider_history": ["gemini", "claude"],
    "updated_at": datetime
}
```

---

### Component 4: **Circuit Breaker** ⚡
**File:** `core/circuit_breaker.py` (NEW)  
**Priority:** HIGH - Build Fourth  
**Est. Lines:** 200-300  
**Dependencies:** `core/models.py`, `utils/database.py`

**What it does:**
- Monitors provider health
- Prevents cascading failures
- Auto-recovery with exponential backoff
- Three states: CLOSED (ok), OPEN (failing), HALF_OPEN (testing)

**Key Classes:**
```python
class CircuitBreaker:
    def is_available(provider: str) -> bool
    async def record_success(provider: str)
    async def record_failure(provider: str)
    async def get_health_status() -> Dict[str, HealthStatus]
    
    # State machine logic
    async def _open_circuit(provider: str)
    async def _half_open_circuit(provider: str)
    async def _close_circuit(provider: str)
```

**Configuration:**
```python
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,        # Open after 5 failures
    "success_threshold": 2,        # Close after 2 successes in half-open
    "timeout_seconds": 60,         # Try again after 60s
    "half_open_max_calls": 3      # Test with max 3 calls
}
```

---

### Component 5: **Enhanced AI Provider System** 🔄
**File:** `core/ai_providers.py` (ENHANCE EXISTING)  
**Priority:** CRITICAL - Integrate with Components 1-4  
**Est. Additional Lines:** +400-600  

**What to add:**
1. **Integrate BenchmarkEngine:**
   ```python
   class ProviderManager:
       def __init__(self):
           self.benchmark_engine = BenchmarkEngine(...)
           self.smart_router = SmartRouter(...)
           self.circuit_breaker = CircuitBreaker()
           self.session_manager = SessionManager()
   ```

2. **Smart provider selection:**
   ```python
   async def generate_with_routing(
       self,
       message: str,
       emotion_state: EmotionState,
       session_id: str,
       context: dict
   ) -> AIResponse:
       # Use smart router instead of default
       provider_name = await self.smart_router.select_provider(
           message, emotion_state, session_id, context
       )
       
       # Check circuit breaker
       if not self.circuit_breaker.is_available(provider_name):
           provider_name = await self._get_fallback_provider()
       
       # Generate with selected provider
       response = await self.universal.generate(provider_name, message)
       
       # Record result for circuit breaker
       if response.success:
           await self.circuit_breaker.record_success(provider_name)
       else:
           await self.circuit_breaker.record_failure(provider_name)
       
       return response
   ```

3. **Background benchmark scheduler:**
   ```python
   async def start_benchmark_scheduler(self):
       """Run benchmarks in background"""
       asyncio.create_task(
           self.benchmark_engine.schedule_benchmarks(interval_hours=1)
       )
   ```

---

### Component 6: **Enhanced MasterX Engine** 🧠
**File:** `core/engine.py` (ENHANCE EXISTING)  
**Priority:** HIGH - Integrate smart routing  
**Est. Additional Lines:** +200-300  

**What to add:**
1. **Use smart routing:**
   ```python
   async def process_request(
       self,
       user_id: str,
       message: str,
       session_id: str,
       context: Optional[dict] = None
   ) -> AIResponse:
       # Phase 1: Analyze emotion
       emotion_result = await self.emotion_engine.analyze_emotion(...)
       
       # Phase 2: Smart provider selection + generation (NEW!)
       response = await self.provider_manager.generate_with_routing(
           message=message,
           emotion_state=emotion_state,
           session_id=session_id,
           context=context
       )
       
       # Phase 3: Track and return
       return response
   ```

2. **Provider performance tracking:**
   ```python
   async def get_provider_stats(self) -> Dict:
       """Get provider performance statistics"""
       return await self.provider_manager.benchmark_engine.get_latest_benchmarks(
           category="general"
       )
   ```

---

### Component 7: **New API Endpoints** 📡
**File:** `server.py` (ENHANCE EXISTING)  
**Priority:** MEDIUM - Add monitoring endpoints  
**Est. Additional Lines:** +150-200  

**What to add:**
```python
@app.get("/api/v1/benchmarks")
async def get_benchmarks(category: Optional[str] = None):
    """Get latest benchmark results"""
    engine = app.state.engine
    if category:
        results = await engine.provider_manager.benchmark_engine.get_latest_benchmarks(category)
    else:
        # Get all categories
        results = {}
        for cat in ["coding", "math", "research", "language", "empathy", "general"]:
            results[cat] = await engine.provider_manager.benchmark_engine.get_latest_benchmarks(cat)
    return results

@app.post("/api/v1/admin/run-benchmarks")
async def trigger_benchmarks(categories: Optional[List[str]] = None):
    """Manually trigger benchmark run"""
    engine = app.state.engine
    results = await engine.provider_manager.benchmark_engine.run_benchmarks(categories=categories)
    return {"status": "completed", "results": results}

@app.get("/api/v1/provider-health")
async def get_provider_health():
    """Get provider health status"""
    engine = app.state.engine
    health = await engine.provider_manager.circuit_breaker.get_health_status()
    return health
```

---

## 🗓️ IMPLEMENTATION TIMELINE

### **Week 1: Core Benchmarking (Days 1-5)**

#### Day 1: Benchmark Engine Foundation
- [ ] Create `core/benchmarking.py`
- [ ] Implement `BenchmarkEngine` class
- [ ] Define test suite structure (`BENCHMARK_CATEGORIES`)
- [ ] Implement `_run_category_tests()` method
- [ ] Test with 1 provider, 1 category

#### Day 2: Benchmark Scoring & Storage
- [ ] Implement `_evaluate_quality()` method
- [ ] Implement `_calculate_scores()` method
- [ ] Implement MongoDB storage (`_save_benchmark_result()`)
- [ ] Implement `get_latest_benchmarks()` retrieval
- [ ] Test full benchmark run with all providers

#### Day 3: Smart Router Foundation
- [ ] Create `core/smart_router.py`
- [ ] Implement `TaskCategoryDetector` class
- [ ] Implement basic `select_provider()` method
- [ ] Implement `_detect_category()` method
- [ ] Test category detection with various inputs

#### Day 4: Session Management
- [ ] Create `core/session_manager.py`
- [ ] Implement `SessionManager` class
- [ ] Add session tracking to MongoDB
- [ ] Implement topic change detection
- [ ] Test session continuity

#### Day 5: Circuit Breaker
- [ ] Create `core/circuit_breaker.py`
- [ ] Implement `CircuitBreaker` class
- [ ] Implement state machine (CLOSED → OPEN → HALF_OPEN)
- [ ] Test failure scenarios
- [ ] Test auto-recovery

### **Week 2: Integration & Testing (Days 6-10)**

#### Day 6: Integration - Provider System
- [ ] Enhance `core/ai_providers.py`
- [ ] Add `generate_with_routing()` method
- [ ] Integrate BenchmarkEngine
- [ ] Integrate SmartRouter
- [ ] Integrate CircuitBreaker
- [ ] Test integrated system

#### Day 7: Integration - Engine
- [ ] Enhance `core/engine.py`
- [ ] Update `process_request()` to use smart routing
- [ ] Add performance tracking
- [ ] Test end-to-end flow

#### Day 8: API Endpoints & Monitoring
- [ ] Add benchmark endpoints to `server.py`
- [ ] Add provider health endpoints
- [ ] Test all new endpoints
- [ ] Create admin dashboard queries

#### Day 9: Background Scheduler
- [ ] Implement benchmark scheduler
- [ ] Add to server startup
- [ ] Configure interval (1-12 hours)
- [ ] Test scheduled runs
- [ ] Monitor logs

#### Day 10: Testing & Optimization
- [ ] Full system integration test
- [ ] Load testing with multiple providers
- [ ] Performance optimization
- [ ] Bug fixes
- [ ] Documentation updates

---

## 🧪 TESTING CHECKLIST

### Unit Tests
- [ ] BenchmarkEngine test suite execution
- [ ] Quality scoring algorithm
- [ ] Category detection accuracy
- [ ] Session manager CRUD operations
- [ ] Circuit breaker state transitions

### Integration Tests
- [ ] Smart routing with real benchmarks
- [ ] Provider selection accuracy
- [ ] Session continuity verification
- [ ] Circuit breaker with provider failures
- [ ] Cost tracking integration

### End-to-End Tests
```bash
# Test 1: Basic routing
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test1", "message": "Explain bubble sort"}'
# Expected: Routes to best coding provider

# Test 2: Emotion-based routing
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test2", "message": "Im so frustrated with this!"}'
# Expected: Routes to best empathy provider

# Test 3: Session continuity
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test3", "session_id": "sess1", "message": "Explain quicksort"}'
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test3", "session_id": "sess1", "message": "Now explain mergesort"}'
# Expected: Same provider for both (same topic: coding)

# Test 4: Topic change
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test4", "session_id": "sess2", "message": "Explain sorting"}'
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test4", "session_id": "sess2", "message": "Solve 2x+5=15"}'
# Expected: Different providers (coding → math)

# Test 5: Get benchmarks
curl http://localhost:8001/api/v1/benchmarks?category=coding

# Test 6: Provider health
curl http://localhost:8001/api/v1/provider-health
```

---

## 📊 SUCCESS METRICS

### Performance Targets
- [ ] Benchmark execution: < 5 minutes for all categories
- [ ] Provider selection: < 50ms per request
- [ ] Session lookup: < 10ms per request
- [ ] Circuit breaker check: < 5ms per request
- [ ] Overall request latency: < 3s (p50), < 7s (p95)

### Quality Targets
- [ ] Routing accuracy: > 90% (correct provider selected)
- [ ] Cost reduction: 20-40% vs always using GPT-4
- [ ] Quality maintenance: > 95% vs best static selection
- [ ] Session continuity: > 95% (same provider for same topic)
- [ ] Circuit breaker response: < 1s failure detection

### Reliability Targets
- [ ] System uptime: > 99.5%
- [ ] Provider fallback success: > 99%
- [ ] Benchmark completion rate: > 95%
- [ ] MongoDB write success: > 99.9%

---

## 🔍 MONITORING & DEBUGGING

### Key Metrics to Track
```python
# Add to server logs
logger.info("benchmark_run", {
    "category": category,
    "provider": provider,
    "quality_score": result.quality_score,
    "speed_score": result.speed_score,
    "final_score": result.final_score
})

logger.info("provider_selection", {
    "session_id": session_id,
    "selected_provider": provider,
    "category": category,
    "reason": "best_benchmark",
    "score": best_score
})

logger.info("circuit_breaker_event", {
    "provider": provider,
    "state": state,  # CLOSED, OPEN, HALF_OPEN
    "failure_count": failures,
    "action": "opened/closed"
})
```

### Debugging Commands
```bash
# Check benchmark results in MongoDB
mongosh masterx_quantum
db.benchmark_results.find().sort({timestamp: -1}).limit(10)

# Check provider health
db.provider_health.find().sort({timestamp: -1})

# Check session assignments
db.sessions.find({_id: "session-id"})

# View logs
tail -f /var/log/supervisor/backend*.log
```

---

## 📦 DEPENDENCIES TO INSTALL

All dependencies are already in `requirements.txt`:
- ✅ `motor==3.3.1` - MongoDB async driver
- ✅ `pymongo==4.5.0` - MongoDB sync operations
- ✅ `numpy==2.3.3` - Numerical operations for scoring
- ✅ `scikit-learn==1.7.2` - Future ML enhancements
- ✅ `tiktoken==0.11.0` - Token counting
- ✅ `asyncio` (built-in) - Async operations

No additional installations needed! 🎉

---

## 🚀 QUICK START COMMANDS

```bash
# 1. Verify environment
cd /app/backend
python -c "from core.models import BenchmarkResult; print('✅ Models ready')"

# 2. Test provider discovery
python -c "from core.ai_providers import ProviderRegistry; r = ProviderRegistry(); print(f'Providers: {list(r.providers.keys())}')"

# 3. Create benchmarking.py
# (Follow Day 1 tasks)

# 4. Run first benchmark test
python -m pytest tests/test_benchmarking.py -v

# 5. Start server with benchmarking
uvicorn server:app --reload --port 8001
```

---

## 📝 IMPLEMENTATION NOTES

### Best Practices
1. **Incremental Development:** Build one component at a time, test thoroughly
2. **Log Everything:** Use structured logging for all operations
3. **Error Handling:** Wrap all external calls in try-except
4. **Async Operations:** Use async/await consistently
5. **Type Hints:** Add full type hints to all functions
6. **Documentation:** Add docstrings to all classes and methods

### Common Pitfalls to Avoid
1. ❌ Don't hardcode provider names - use discovery
2. ❌ Don't block on benchmarks - run in background
3. ❌ Don't ignore circuit breaker state
4. ❌ Don't forget to update session on topic change
5. ❌ Don't skip error logging - critical for debugging

### Code Quality Standards
- Follow PEP 8 style guide
- Use black for formatting
- Add type hints everywhere
- Write clear, concise docstrings
- Keep functions < 50 lines
- Keep files < 1000 lines

---

## 🎯 PHASE 2 COMPLETION CRITERIA

Phase 2 is complete when:
- [ ] All 7 components implemented
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] All E2E tests passing
- [ ] Benchmarks running automatically
- [ ] Smart routing operational
- [ ] Circuit breaker functional
- [ ] Session continuity working
- [ ] API endpoints responsive
- [ ] Performance targets met
- [ ] Documentation updated
- [ ] No critical bugs

**Then we move to Phase 3: Context Management & Adaptive Learning** 🚀

---

**Let's build the world's most advanced AI routing system!** 💪

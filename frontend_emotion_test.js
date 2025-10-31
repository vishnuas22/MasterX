/**
 * MasterX Frontend Emotion Detection Interactive Test
 * 
 * How to use:
 * 1. Open MasterX frontend in browser
 * 2. Open Developer Console (F12)
 * 3. Copy and paste this entire script
 * 4. Run: await runEmotionTests()
 * 
 * This will automatically test emotion detection through the UI
 */

// Test scenarios with expected results
const emotionTestScenarios = [
    {
        id: 1,
        category: "Positive - Joy",
        message: "I finally understand this! This is amazing!",
        expected: {
            emotions: ["joy", "excitement", "admiration"],
            valence: "positive",
            readiness: ["optimal", "good"]
        }
    },
    {
        id: 2,
        category: "Positive - Excitement",
        message: "Wow! This is so exciting! I can't wait to learn more!",
        expected: {
            emotions: ["excitement", "joy"],
            valence: "positive",
            readiness: ["optimal"]
        }
    },
    {
        id: 3,
        category: "Negative - Frustration",
        message: "This is so frustrating! I've tried three times and still can't get it right.",
        expected: {
            emotions: ["annoyance", "disappointment", "anger"],
            valence: "negative",
            readiness: ["low", "moderate"]
        }
    },
    {
        id: 4,
        category: "Negative - High Confusion",
        message: "I'm completely lost. Nothing makes sense. What does this even mean?",
        expected: {
            emotions: ["confusion", "annoyance"],
            valence: "negative",
            readiness: ["low", "blocked"]
        }
    },
    {
        id: 5,
        category: "Learning - High Curiosity",
        message: "That's interesting! How does that work? Can you tell me more?",
        expected: {
            emotions: ["curiosity", "excitement"],
            valence: "positive",
            readiness: ["optimal", "good"]
        }
    },
    {
        id: 6,
        category: "Learning - Moderate Confusion",
        message: "I don't understand this part. Could you explain it differently?",
        expected: {
            emotions: ["confusion", "curiosity"],
            valence: "neutral",
            readiness: ["moderate", "good"]
        }
    },
    {
        id: 7,
        category: "Learning - Realization",
        message: "Oh! Now I get it! That makes so much sense now!",
        expected: {
            emotions: ["realization", "joy"],
            valence: "positive",
            readiness: ["optimal"]
        }
    },
    {
        id: 8,
        category: "Mixed - Confusion + Curiosity",
        message: "This is confusing, but I'm curious to understand how it works.",
        expected: {
            emotions: ["confusion", "curiosity"],
            valence: "neutral",
            readiness: ["moderate", "good"]
        }
    },
    {
        id: 9,
        category: "Neutral - Question",
        message: "What is the derivative of x squared?",
        expected: {
            emotions: ["neutral", "curiosity"],
            valence: "neutral",
            readiness: ["good", "moderate"]
        }
    },
    {
        id: 10,
        category: "Context - Giving Up",
        message: "I can't do this anymore. It's too hard.",
        expected: {
            emotions: ["disappointment", "sadness"],
            valence: "negative",
            readiness: ["low", "blocked"]
        }
    }
];

// Colors for console output
const colors = {
    reset: "\x1b[0m",
    bright: "\x1b[1m",
    red: "\x1b[31m",
    green: "\x1b[32m",
    yellow: "\x1b[33m",
    blue: "\x1b[34m",
    magenta: "\x1b[35m",
    cyan: "\x1b[36m"
};

/**
 * Test emotion detection via API
 */
async function testEmotionAPI(message, testId) {
    const backendUrl = window.location.hostname === 'localhost' 
        ? 'http://localhost:8001'
        : ''; // Use relative URL for deployed version
    
    try {
        const response = await fetch(`${backendUrl}/api/v1/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: `test_user_frontend`,
                session_id: `test_session_${Date.now()}`,
                message: message
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error(`%c❌ API Error for test ${testId}:`, 'color: red; font-weight: bold', error);
        return null;
    }
}

/**
 * Check if detected emotion matches expected
 */
function checkEmotionMatch(detected, expected) {
    const detectedLower = detected.toLowerCase();
    return expected.some(exp => detectedLower.includes(exp.toLowerCase()));
}

/**
 * Check valence direction
 */
function checkValence(valence, expected) {
    if (expected === "positive") return valence > 0.2;
    if (expected === "negative") return valence < -0.1;
    if (expected === "neutral") return valence >= -0.2 && valence <= 0.2;
    return true;
}

/**
 * Run all emotion tests
 */
async function runEmotionTests() {
    console.clear();
    console.log(`%c
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║          MasterX Frontend Emotion Detection Tests              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    `, 'color: cyan; font-weight: bold; font-size: 14px');
    
    console.log(`%cStarting ${emotionTestScenarios.length} emotion detection tests...`, 'color: yellow; font-weight: bold');
    console.log('');
    
    const results = [];
    let passed = 0;
    let failed = 0;
    
    for (let i = 0; i < emotionTestScenarios.length; i++) {
        const test = emotionTestScenarios[i];
        console.log(`%c[${i + 1}/${emotionTestScenarios.length}] Testing: ${test.category}`, 'color: cyan; font-weight: bold');
        console.log(`   Message: "${test.message}"`);
        
        // Add small delay between tests
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const startTime = Date.now();
        const response = await testEmotionAPI(test.message, test.id);
        const endTime = Date.now();
        
        if (!response) {
            console.log(`%c   ❌ FAIL - API Error`, 'color: red; font-weight: bold');
            console.log('');
            failed++;
            results.push({
                ...test,
                status: 'FAIL',
                reason: 'API Error',
                responseTime: endTime - startTime
            });
            continue;
        }
        
        // Extract emotion data
        const emotionState = response.emotion_state;
        const responseTime = endTime - startTime;
        
        if (!emotionState) {
            console.log(`%c   ❌ FAIL - No emotion_state in response`, 'color: red; font-weight: bold');
            console.log('');
            failed++;
            results.push({
                ...test,
                status: 'FAIL',
                reason: 'No emotion_state',
                responseTime
            });
            continue;
        }
        
        const detected = emotionState.primary_emotion;
        const valence = emotionState.valence;
        const readiness = emotionState.learning_readiness;
        
        // Check matches
        const emotionMatch = checkEmotionMatch(detected, test.expected.emotions);
        const valenceMatch = checkValence(valence, test.expected.valence);
        const readinessMatch = test.expected.readiness.includes(readiness);
        
        const testPassed = emotionMatch && valenceMatch;
        
        if (testPassed) {
            console.log(`%c   ✅ PASS`, 'color: green; font-weight: bold');
            passed++;
        } else {
            console.log(`%c   ❌ FAIL`, 'color: red; font-weight: bold');
            failed++;
        }
        
        console.log(`   Detected: ${detected} (confidence: ${emotionState.arousal?.toFixed(3) || 'N/A'})`);
        console.log(`   Expected: ${test.expected.emotions.join(', ')}`);
        console.log(`   Valence: ${valence?.toFixed(3)} (expected: ${test.expected.valence}) ${valenceMatch ? '✓' : '✗'}`);
        console.log(`   Readiness: ${readiness} (expected: ${test.expected.readiness.join(', ')}) ${readinessMatch ? '✓' : '✗'}`);
        console.log(`   Response time: ${responseTime}ms`);
        
        if (!testPassed) {
            console.log(`%c   Issues:`, 'color: orange');
            if (!emotionMatch) console.log(`      - Wrong emotion detected`);
            if (!valenceMatch) console.log(`      - Wrong valence direction`);
        }
        
        console.log('');
        
        results.push({
            ...test,
            status: testPassed ? 'PASS' : 'FAIL',
            detected: {
                emotion: detected,
                valence,
                readiness
            },
            matches: {
                emotion: emotionMatch,
                valence: valenceMatch,
                readiness: readinessMatch
            },
            responseTime
        });
    }
    
    // Print summary
    console.log(`%c
╔════════════════════════════════════════════════════════════════╗
║                       TEST SUMMARY                             ║
╚════════════════════════════════════════════════════════════════╝
    `, 'color: cyan; font-weight: bold; font-size: 14px');
    
    const passRate = (passed / emotionTestScenarios.length * 100).toFixed(1);
    const avgResponseTime = (results.reduce((sum, r) => sum + r.responseTime, 0) / results.length).toFixed(0);
    
    console.log(`%cTotal Tests: ${emotionTestScenarios.length}`, 'font-weight: bold');
    console.log(`%c✅ Passed: ${passed} (${passRate}%)`, 'color: green; font-weight: bold');
    console.log(`%c❌ Failed: ${failed} (${(100 - passRate).toFixed(1)}%)`, 'color: red; font-weight: bold');
    console.log(`%c⚡ Avg Response Time: ${avgResponseTime}ms`, 'color: yellow; font-weight: bold');
    console.log('');
    
    // Show failures in detail
    if (failed > 0) {
        console.log(`%c⚠️  FAILED TESTS:`, 'color: orange; font-weight: bold; font-size: 14px');
        results.filter(r => r.status === 'FAIL').forEach((r, idx) => {
            console.log(`%c  [${idx + 1}] ${r.category}`, 'color: red; font-weight: bold');
            console.log(`      Message: "${r.message}"`);
            console.log(`      Expected: ${r.expected.emotions.join(', ')}`);
            if (r.detected) {
                console.log(`      Detected: ${r.detected.emotion}`);
                console.log(`      Reason: ${!r.matches.emotion ? 'Wrong emotion' : ''} ${!r.matches.valence ? 'Wrong valence' : ''}`);
            } else {
                console.log(`      Reason: ${r.reason}`);
            }
            console.log('');
        });
    }
    
    // Performance analysis
    console.log(`%c📊 PERFORMANCE ANALYSIS:`, 'color: cyan; font-weight: bold');
    const responseTimes = results.map(r => r.responseTime);
    const minTime = Math.min(...responseTimes);
    const maxTime = Math.max(...responseTimes);
    const p50 = responseTimes.sort((a, b) => a - b)[Math.floor(responseTimes.length / 2)];
    
    console.log(`   Min: ${minTime}ms`);
    console.log(`   Max: ${maxTime}ms`);
    console.log(`   P50: ${p50}ms`);
    console.log(`   Avg: ${avgResponseTime}ms`);
    console.log('');
    
    // Recommendations
    if (passRate < 50) {
        console.log(`%c⚠️  WARNING: Pass rate below 50%! Emotion detection needs attention.`, 'color: red; font-weight: bold; font-size: 14px');
    } else if (passRate < 70) {
        console.log(`%c⚡ GOOD: Pass rate ${passRate}%. Consider fine-tuning for better accuracy.`, 'color: yellow; font-weight: bold');
    } else {
        console.log(`%c✅ EXCELLENT: Pass rate ${passRate}%. System performing well!`, 'color: green; font-weight: bold; font-size: 14px');
    }
    
    if (parseInt(avgResponseTime) > 500) {
        console.log(`%c⚠️  Response time is high (${avgResponseTime}ms). Consider optimization.`, 'color: orange; font-weight: bold');
    }
    
    console.log('');
    console.log(`%cTest completed! Results object available as window.__emotionTestResults`, 'color: cyan');
    
    // Store results globally for inspection
    window.__emotionTestResults = results;
    
    return results;
}

/**
 * Test single emotion detection
 */
async function testSingleEmotion(message) {
    console.log(`%c🧪 Testing emotion for: "${message}"`, 'color: cyan; font-weight: bold');
    
    const startTime = Date.now();
    const response = await testEmotionAPI(message, 'single');
    const endTime = Date.now();
    
    if (!response || !response.emotion_state) {
        console.log(`%c❌ Error: No emotion detected`, 'color: red; font-weight: bold');
        return null;
    }
    
    const emotionState = response.emotion_state;
    
    console.log(`%c✅ Results:`, 'color: green; font-weight: bold');
    console.log(`   Primary Emotion: ${emotionState.primary_emotion}`);
    console.log(`   Arousal: ${emotionState.arousal?.toFixed(3)}`);
    console.log(`   Valence: ${emotionState.valence?.toFixed(3)}`);
    console.log(`   Learning Readiness: ${emotionState.learning_readiness}`);
    console.log(`   Response Time: ${endTime - startTime}ms`);
    
    return emotionState;
}

// Export functions to window for console access
window.runEmotionTests = runEmotionTests;
window.testSingleEmotion = testSingleEmotion;

// Instructions
console.log(`%c
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     MasterX Emotion Detection Test Suite Loaded! 🎉            ║
║                                                                ║
║  Usage:                                                        ║
║    await runEmotionTests()     - Run all 10 tests             ║
║    await testSingleEmotion("your message here")               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
`, 'color: green; font-weight: bold; font-size: 14px');

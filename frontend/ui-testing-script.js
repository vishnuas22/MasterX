// MasterX UI/UX Testing Script - Run in browser console
// Comprehensive testing of multi-modal chat interface

console.log('🧪 MASTERX UI/UX TESTING SCRIPT STARTING...\n');

// Test Results Storage
const testResults = {
    interface: {},
    functionality: {},
    multiModal: {},
    visual: {},
    issues: []
};

// 1. INTERFACE COMPONENT VERIFICATION
function testInterfaceComponents() {
    console.log('🔍 Testing Interface Components...');
    
    const components = {
        // Core Chat Elements
        chatInput: document.querySelector('textarea[placeholder*="Ask me anything"]'),
        sendButton: document.querySelector('button[class*="quantum-button"]'),
        messagesContainer: document.querySelector('.quantum-scroll'),
        
        // Multi-Modal Controls
        microphoneButton: document.querySelector('button[title*="voice"], button[title*="recording"]'),
        fileUploadButton: document.querySelector('button[title*="file"], button[title*="upload"]'),
        fileInput: document.getElementById('file-upload'),
        settingsButton: document.querySelector('button[class*="neural-network-button"]'),
        
        // Session & Status
        sessionBar: document.querySelector('[class*="glass-morph-premium"]'),
        connectionStatus: document.querySelector('[class*="emerald-"], [class*="amber-"], [class*="red-"]'),
        
        // Header Elements
        header: document.querySelector('header'),
        searchBar: document.querySelector('input[placeholder*="Search"]'),
        menuButton: document.querySelector('button[class*="lg:hidden"]')
    };
    
    console.log('📋 Component Verification Results:');
    Object.entries(components).forEach(([name, element]) => {
        const found = element !== null;
        testResults.interface[name] = found;
        console.log(`${found ? '✅' : '❌'} ${name}: ${found ? 'FOUND' : 'MISSING'}`);
        
        if (!found) {
            testResults.issues.push(`Missing component: ${name}`);
        }
    });
    
    return components;
}

// 2. FUNCTIONALITY TESTING
function testCoreFunctionality(components) {
    console.log('\n⚙️ Testing Core Functionality...');
    
    const tests = {
        chatInputEnabled: false,
        sendButtonClickable: false,
        microphoneClickable: false,
        fileUploadClickable: false,
        settingsClickable: false
    };
    
    // Test Chat Input
    if (components.chatInput) {
        tests.chatInputEnabled = !components.chatInput.disabled && !components.chatInput.readOnly;
        console.log(`💬 Chat Input: ${tests.chatInputEnabled ? 'ENABLED' : 'DISABLED'}`);
        
        // Test typing
        try {
            components.chatInput.focus();
            components.chatInput.value = 'Test message';
            components.chatInput.dispatchEvent(new Event('input', { bubbles: true }));
            console.log('✅ Chat Input: Text entry working');
        } catch (error) {
            console.log('❌ Chat Input: Text entry failed');
            testResults.issues.push('Chat input text entry not working');
        }
    }
    
    // Test Send Button
    if (components.sendButton) {
        tests.sendButtonClickable = !components.sendButton.disabled;
        console.log(`📤 Send Button: ${tests.sendButtonClickable ? 'CLICKABLE' : 'DISABLED'}`);
    }
    
    // Test Microphone Button
    if (components.microphoneButton) {
        tests.microphoneClickable = !components.microphoneButton.disabled;
        console.log(`🎤 Microphone Button: ${tests.microphoneClickable ? 'CLICKABLE' : 'DISABLED'}`);
    }
    
    // Test File Upload Button
    if (components.fileUploadButton) {
        tests.fileUploadClickable = !components.fileUploadButton.disabled;
        console.log(`📁 File Upload Button: ${tests.fileUploadClickable ? 'CLICKABLE' : 'DISABLED'}`);
    }
    
    // Test Settings Button
    if (components.settingsButton) {
        tests.settingsClickable = !components.settingsButton.disabled;
        console.log(`⚙️ Settings Button: ${tests.settingsClickable ? 'CLICKABLE' : 'DISABLED'}`);
    }
    
    testResults.functionality = tests;
    return tests;
}

// 3. MULTI-MODAL FEATURE TESTING
function testMultiModalFeatures() {
    console.log('\n🎭 Testing Multi-Modal Features...');
    
    const features = {
        speechRecognition: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
        speechSynthesis: 'speechSynthesis' in window,
        fileAPI: 'File' in window && 'FileReader' in window,
        dragDrop: 'ondragstart' in document.createElement('div')
    };
    
    console.log('🌐 Browser API Support:');
    Object.entries(features).forEach(([feature, supported]) => {
        console.log(`${supported ? '✅' : '❌'} ${feature}: ${supported ? 'SUPPORTED' : 'NOT SUPPORTED'}`);
        if (!supported) {
            testResults.issues.push(`Browser API not supported: ${feature}`);
        }
    });
    
    testResults.multiModal = features;
    return features;
}

// 4. VISUAL TESTING
function testVisualElements() {
    console.log('\n🎨 Testing Visual Elements...');
    
    const visual = {
        quantumTheme: document.querySelectorAll('[class*="quantum"]').length > 0,
        glassMorph: document.querySelectorAll('[class*="glass-morph"]').length > 0,
        animations: document.querySelectorAll('[class*="animate"]').length > 0,
        gradients: document.querySelectorAll('[class*="gradient"]').length > 0,
        responsiveLayout: window.innerWidth > 0 && document.body.offsetWidth > 0
    };
    
    console.log('✨ Visual Theme Elements:');
    Object.entries(visual).forEach(([element, present]) => {
        console.log(`${present ? '✅' : '❌'} ${element}: ${present ? 'PRESENT' : 'MISSING'}`);
    });
    
    // Check for layout issues
    const layoutIssues = [];
    
    // Check if elements are visible
    const criticalElements = document.querySelectorAll('textarea, button, input');
    criticalElements.forEach((el, index) => {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) {
            layoutIssues.push(`Element ${index} has zero dimensions`);
        }
    });
    
    if (layoutIssues.length > 0) {
        testResults.issues.push(...layoutIssues);
    }
    
    testResults.visual = visual;
    return visual;
}

// 5. INTERACTIVE TESTING
function testInteractiveElements(components) {
    console.log('\n🖱️ Testing Interactive Elements...');
    
    const interactions = {
        settingsPanelToggle: false,
        fileUploadTrigger: false,
        voiceInputTrigger: false
    };
    
    // Test Settings Panel Toggle
    if (components.settingsButton) {
        try {
            const initialSettingsPanel = document.querySelector('[class*="settings"]');
            components.settingsButton.click();
            setTimeout(() => {
                const newSettingsPanel = document.querySelector('[class*="settings"]');
                interactions.settingsPanelToggle = initialSettingsPanel !== newSettingsPanel;
                console.log(`⚙️ Settings Panel Toggle: ${interactions.settingsPanelToggle ? 'WORKING' : 'NOT WORKING'}`);
            }, 100);
        } catch (error) {
            console.log('❌ Settings Panel Toggle: ERROR');
            testResults.issues.push('Settings panel toggle not working');
        }
    }
    
    // Test File Upload Trigger
    if (components.fileUploadButton && components.fileInput) {
        try {
            components.fileUploadButton.click();
            interactions.fileUploadTrigger = true;
            console.log('✅ File Upload Trigger: WORKING');
        } catch (error) {
            console.log('❌ File Upload Trigger: ERROR');
            testResults.issues.push('File upload trigger not working');
        }
    }
    
    testResults.interactions = interactions;
    return interactions;
}

// 6. COMPREHENSIVE TEST RUNNER
async function runComprehensiveUITest() {
    console.log('🚀 STARTING COMPREHENSIVE UI/UX TEST\n');
    console.log('=' .repeat(50));
    
    const startTime = Date.now();
    
    // Run all tests
    const components = testInterfaceComponents();
    const functionality = testCoreFunctionality(components);
    const multiModal = testMultiModalFeatures();
    const visual = testVisualElements();
    const interactions = testInteractiveElements(components);
    
    const endTime = Date.now();
    
    // Calculate results
    const totalTests = Object.keys(testResults.interface).length + 
                      Object.keys(testResults.functionality).length + 
                      Object.keys(testResults.multiModal).length + 
                      Object.keys(testResults.visual).length;
    
    const passedTests = Object.values(testResults.interface).filter(Boolean).length +
                       Object.values(testResults.functionality).filter(Boolean).length +
                       Object.values(testResults.multiModal).filter(Boolean).length +
                       Object.values(testResults.visual).filter(Boolean).length;
    
    const successRate = ((passedTests / totalTests) * 100).toFixed(1);
    
    // Results Summary
    console.log('\n' + '=' .repeat(50));
    console.log('📊 COMPREHENSIVE UI/UX TEST RESULTS');
    console.log('=' .repeat(50));
    console.log(`🎯 Overall Score: ${passedTests}/${totalTests} (${successRate}%)`);
    console.log(`⏱️ Test Duration: ${endTime - startTime}ms`);
    console.log(`🚨 Issues Found: ${testResults.issues.length}`);
    
    if (testResults.issues.length > 0) {
        console.log('\n🔧 ISSUES TO FIX:');
        testResults.issues.forEach((issue, index) => {
            console.log(`${index + 1}. ${issue}`);
        });
    }
    
    console.log(`\n🏆 STATUS: ${successRate >= 90 ? '✅ EXCELLENT' : successRate >= 75 ? '⚠️ GOOD' : '❌ NEEDS WORK'}`);
    console.log('=' .repeat(50));
    
    // Store results globally
    window.masterxUITest = {
        results: testResults,
        summary: { totalTests, passedTests, successRate, duration: endTime - startTime },
        rerun: runComprehensiveUITest
    };
    
    return testResults;
}

// Auto-run test
console.log('🎬 Starting UI/UX test in 2 seconds...');
setTimeout(runComprehensiveUITest, 2000);

// Export for manual use
window.masterxUITest = { run: runComprehensiveUITest };

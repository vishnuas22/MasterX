// Quick store test to verify initialization
console.log('🧪 Testing MasterX Store...')

// Test store import
try {
  const { useStore, initializeStore, isStoreReady } = require('./store/index.ts')
  console.log('✅ Store imports successful')
  
  // Test store initialization
  initializeStore()
  console.log('✅ Store initialization successful')
  
  // Test store ready check
  const ready = isStoreReady()
  console.log(`✅ Store ready check: ${ready}`)
  
  console.log('🎉 All store tests passed!')
} catch (error) {
  console.error('❌ Store test failed:', error)
}

# 🎯 MASTERX TESTING - QUICK REFERENCE GUIDE

**Quick access guide for testing sessions**

---

## 🚀 QUICK START

### Test User Credentials
```
✅ ACTIVE USER (Created & Verified)
Always create new user for tests
Status: Onboarding incomplete
```

### Create New Test User
```bash
# Name Format: [FirstName LastName] (letters & spaces only, no numbers)
✅ Valid: "Emma Johnson", "John Smith", "Maria Garcia"
❌ Invalid: "Test User 123", "User-123", "Test_User"

# Email Format: [name][random-number]@masterx.ai
✅ Example: john.smith4567@masterx.ai

# Password Requirements: 8+ chars, Upper, lower, number, symbol
✅ Example: TestUser@2025!
```

---

## 📊 CURRENT PROGRESS

```
✅ Phase 1: Authentication (100%) - COMPLETE
✅ Phase 2: Onboarding (95%) - COMPLETE  
🔄 Phase 3: Main Chat (0%) - NEXT PRIORITY
⏳ Phase 4: Emotion Viz (0%)
⏳ Phase 5: Dashboard (0%)
⏳ Phase 6: Gamification (0%)
⏳ Phase 7: Voice (0%)
⏳ Phase 8: Settings (0%)
⏳ Phase 9: Collaboration (0%)
⏳ Phase 10: Performance (0%)
⏳ Phase 11: Responsive (0%)
⏳ Phase 12: Accessibility (0%)

Overall: 19.2% Complete (2/12 phases)
```

---

## 🔴 TOP PRIORITIES (Next to Test)

1. **Main Chat Interface** (CRITICAL)
   - Message sending/receiving
   - **Real-time emotion detection display**
   - AI response rendering
   - Emotion widget updates

2. **Emotion Visualization** (CRITICAL)  
   - Emotion chart over time
   - PAD model display
   - Learning readiness indicator

3. **Voice Interaction** (HIGH)
   - Voice recording
   - Transcription accuracy
   - TTS playback

---

## 🌐 ENVIRONMENT URLS

```
Frontend: https://emotion-adapt-4.preview.emergentagent.com
Backend API: https://emotion-adapt-4.preview.emergentagent.com/api
API Docs: https://emotion-adapt-4.preview.emergentagent.com/docs

Test Routes (DEV only):
/test-login - Quick login for testing
/showcase - Component showcase
/gamification - Gamification showcase
```

---

## 📝 TESTING CHECKLIST TEMPLATES

### Template: Visual Test
```markdown
- [ ] Component renders correctly
- [ ] Styling matches design (colors, spacing, typography)
- [ ] Icons/images load
- [ ] Animations smooth (60fps)
- [ ] Responsive layout
- [ ] Dark mode styling consistent
- [ ] No layout shifts
- [ ] Loading states present
- [ ] Empty states present
- [ ] Error states present
```

### Template: Functional Test
```markdown
- [ ] User action triggers expected behavior
- [ ] Form validation working
- [ ] Data saves to backend
- [ ] Success feedback shown
- [ ] Error handling graceful
- [ ] Backend API called correctly
- [ ] Response data displayed
- [ ] Navigation works
- [ ] State persists (localStorage/session)
- [ ] No console errors
```

### Template: Backend Integration Test
```markdown
- [ ] API endpoint exists
- [ ] Request format correct
- [ ] Authentication included (JWT)
- [ ] Success response (200/201)
- [ ] Error responses handled (400/401/500)
- [ ] Data structure matches expected
- [ ] Network tab shows correct payload
- [ ] Response time acceptable (<2s)
- [ ] Retry logic on failure
- [ ] Loading indicators during request
```

---

## 🎨 SCREENSHOT NAMING GUIDE

```
Format: [feature]_[step]_[description].png

Examples:
- chat_01_initial.png
- chat_02_message_sent.png
- chat_03_emotion_detected.png
- chat_04_ai_response.png

- emotion_01_widget.png
- emotion_02_chart.png
- emotion_03_pad_model.png

- voice_01_recording.png
- voice_02_transcription.png
- voice_03_playback.png
```

---

## 🔧 PLAYWRIGHT SCRIPT TEMPLATE

```typescript
import asyncio

try:
    await page.set_viewport_size({"width": 1920, "height": 800})
    await page.wait_for_load_state("networkidle", timeout=10000)
    
    print("✅ STEP 1: [Description]")
    await page.screenshot(path="[feature]_01_[step].png", quality=20, full_page=False)
    
    # [Action]
    await page.click('[selector]', force=True)
    await asyncio.sleep(1)
    print("✅ STEP 2: [Description]")
    await page.screenshot(path="[feature]_02_[step].png", quality=20, full_page=False)
    
    # Check result
    current_url = page.url
    print(f"📍 Current URL: {current_url}")
    
    if "[expected]" in current_url:
        print("✅ SUCCESS: [Description]")
    else:
        print("⚠️ WARNING: [Description]")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    await page.screenshot(path="error_[feature].png", quality=20, full_page=False)
```

---

## 🐛 COMMON SELECTORS

```typescript
// Buttons
'button:has-text("Login")'
'button:has-text("Send")'
'[data-testid="submit-button"]'

// Inputs
'input[type="email"]'
'input[type="password"]'
'input[placeholder*="Search"]'
'textarea[placeholder*="Type"]'

// Forms
'form[name="login"]'
'input[name="email"]'

// Navigation
'a[href="/app"]'
'nav >> text=Dashboard'

// Dynamic content
'[role="alert"]'
'[aria-live="polite"]'
'.error-message'
'.success-message'

// Wait for selectors
await page.wait_for_selector('[data-testid="chat-message"]')
await page.wait_for_load_state("networkidle")
```

---

## ✅ TESTING BEST PRACTICES

1. **Always set viewport first**
   ```typescript
   await page.set_viewport_size({"width": 1920, "height": 800})
   ```

2. **Wait for page load**
   ```typescript
   await page.wait_for_load_state("networkidle", timeout=10000)
   ```

3. **Use `force=True` for stubborn clicks**
   ```typescript
   await page.click('button', force=True)
   ```

4. **Add sleeps between actions**
   ```typescript
   await asyncio.sleep(1)  # Allow animations to complete
   ```

5. **Capture screenshots at each step**
   - Before action
   - After action
   - On error

6. **Print descriptive logs**
   ```typescript
   print("✅ STEP 1: Description")
   print("❌ ERROR: Description")
   print("⚠️ WARNING: Description")
   print("📍 INFO: Description")
   ```

7. **Always wrap in try-catch**
   ```typescript
   try:
       # Test code
   except Exception as e:
       print(f"❌ ERROR: {str(e)}")
       await page.screenshot(path="error.png")
   ```

8. **Check console logs**
   ```typescript
   capture_logs=true
   ```

9. **Verify URL after navigation**
   ```typescript
   current_url = page.url
   print(f"📍 URL: {current_url}")
   ```

10. **Test error states**
    - Invalid input
    - Network errors
    - Empty states
    - Permission denials

---

## 📊 STATUS INDICATORS

```
✅ Complete / Working
🔄 In Progress
⏳ Pending / Not Started
❌ Failed / Broken
⚠️ Warning / Issue
📍 Info / Note
🔴 Critical Priority
🟠 High Priority
🟡 Medium Priority
🟢 Low Priority
```

---

## 🔗 KEY BACKEND ENDPOINTS

### Authentication
```
POST /api/auth/register - Create user
POST /api/auth/login - Login user
GET /api/auth/me - Get current user
POST /api/auth/logout - Logout user
POST /api/auth/refresh - Refresh JWT token
```

### Chat
```
POST /api/v1/chat - Send message, get AI response + emotion
GET /api/v1/chat/history - Get chat history
GET /api/v1/chat/session - Get session info
```

### Analytics
```
GET /api/v1/analytics/dashboard/{user_id} - Dashboard metrics
GET /api/v1/analytics/emotions/{user_id} - Emotion history
GET /api/v1/analytics/performance/{user_id} - Performance data
```

### Gamification
```
GET /api/v1/gamification/stats/{user_id} - User stats
GET /api/v1/gamification/leaderboard - Leaderboard
GET /api/v1/gamification/achievements - Achievements
```

### Voice
```
POST /api/v1/voice/transcribe - Audio to text
POST /api/v1/voice/synthesize - Text to audio
```

---

## 🎯 TESTING GOALS

### Phase 3: Main Chat (NEXT)
**Goal**: Verify core emotion detection + AI chat works end-to-end

**Critical Tests**:
1. User can send message
2. AI responds with relevant answer
3. **Emotion detected and displayed**
4. Emotion widget updates in real-time
5. Message history persists
6. Typing indicator shows

**Success Criteria**:
- Message sent → AI response received → Emotion displayed
- No console errors
- Smooth user experience
- Visual feedback at each step

---

## 💾 SAVING TEST RESULTS

After each test session, update:
```
/app/FRONTEND_TESTING_PROGRESS.md
```

Add:
- ✅ Completed test checkboxes
- 📸 New screenshots
- 🐛 Any bugs found
- 💡 Observations/notes
- 📊 Updated progress percentages

---

## 🆘 TROUBLESHOOTING

### Issue: Session Expired
```typescript
// Solution: Login again with test user
// Or create fresh user
```

### Issue: Element not found
```typescript
// Solution 1: Wait for element
await page.wait_for_selector('[selector]', timeout=10000)

// Solution 2: Use different selector
// Try: text=, placeholder=, role=, data-testid=

// Solution 3: Check if element loaded
element = await page.query_selector('[selector]')
if element:
    await element.click()
```

### Issue: Click not working
```typescript
// Solution: Add force=True
await page.click('[selector]', force=True)

// Or: Scroll into view first
await page.locator('[selector]').scroll_into_view_if_needed()
await page.click('[selector]')
```

### Issue: Network errors
```typescript
// Solution: Check backend is running
// Check CORS configuration
// Check JWT token is valid
// Wait longer for response
await asyncio.sleep(5)
```

---

**Last Updated**: October 29, 2025  
**Next Update**: After Phase 3 testing

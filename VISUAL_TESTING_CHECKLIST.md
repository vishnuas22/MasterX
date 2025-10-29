# 🎨 MASTERX VISUAL TESTING CHECKLIST

**Quick visual verification checklist for each feature**

---

## ✅ PHASE 1: AUTHENTICATION - VISUAL CHECKLIST

### Landing Page
- [x] MasterX logo in header
- [x] Gradient hero text visible and animated
- [x] Feature cards display (3 columns)
- [x] Stats counters animated
- [x] CTA buttons styled correctly (blue gradient)
- [x] Hover effects on buttons
- [x] Navigation links in header
- [x] Footer with links
- [x] Dark mode background
- [x] Smooth scroll animations

### Signup Page
- [x] "Create your account" heading centered
- [x] MasterX logo/icon at top
- [x] Form centered on page
- [x] All 4 input fields visible (name, email, password, confirm)
- [x] Show/hide password icons (eye icons)
- [x] **Password strength meter visible**
- [x] **Strength meter colors: red → orange → yellow → green**
- [x] Terms checkbox with blue checkmark when checked
- [x] "Create account" button (blue-purple gradient)
- [x] Button has hover effect
- [x] "Back to home" link in header
- [x] Error messages appear in red below fields
- [x] Focus states (blue ring) on inputs

### Login Page
- [x] "Welcome back" heading
- [x] Google OAuth button with Google icon
- [x] Divider line with "Or continue with email" text
- [x] Email and password inputs
- [x] "Remember me for 30 days" checkbox
- [x] "Forgot password?" link (blue)
- [x] "Log in" button (blue)
- [x] "Sign up for free" link at bottom
- [x] Error banner (red background) when login fails
- [x] Dark mode consistent

### Onboarding Page
- [x] "Let's personalize your experience" heading
- [x] Light background (not dark mode)
- [x] Progress bar at top (blue-purple gradient)
- [x] "Step 1 of 4 - 25% complete" text
- [x] 4 step icons visible (Learning Style, Interests, Goals, Preferences)
- [x] Current step highlighted
- [x] Card-based selection UI
- [x] Option cards with icons and descriptions
- [x] Hover effect on cards

---

## 🔄 PHASE 3: MAIN CHAT - VISUAL CHECKLIST

### Chat Container
- [ ] App header with user info (avatar, name, level)
- [ ] Sidebar/navigation menu (collapsed/expanded states)
- [ ] Main chat area (central)
- [ ] Message list area
- [ ] Message input area at bottom
- [ ] Emotion widget (floating or sidebar)
- [ ] All sections properly aligned
- [ ] No overlapping elements
- [ ] Scrollable message area
- [ ] Fixed input at bottom

### Message Display
- [ ] **User messages**:
  - [ ] Right-aligned
  - [ ] Blue background (#3B82F6 or similar)
  - [ ] White text
  - [ ] Avatar on right
  - [ ] Timestamp visible (subtle gray)
  - [ ] Border radius (rounded corners)
  - [ ] Proper padding
  - [ ] Shadow effect
- [ ] **AI messages**:
  - [ ] Left-aligned
  - [ ] Gray/darker background
  - [ ] White text
  - [ ] AI avatar on left
  - [ ] Timestamp visible
  - [ ] Formatting preserved (bold, italic, code)
  - [ ] Code blocks styled with syntax highlighting
- [ ] **Message spacing**:
  - [ ] Consistent gap between messages
  - [ ] Grouped by sender (no avatar repeat)
  - [ ] Date separators between days

### Message Input
- [ ] Textarea for input (not single-line input)
- [ ] Placeholder text: "Ask me anything..." or similar
- [ ] Auto-resize as user types
- [ ] Max height before scrolling
- [ ] Send button visible (arrow or paper plane icon)
- [ ] Send button color: blue (#3B82F6)
- [ ] Send button disabled when empty
- [ ] Voice input button (microphone icon)
- [ ] Attachment button (paperclip icon)
- [ ] Emoji picker button (smiley icon)
- [ ] Character count (if limit exists)
- [ ] All buttons have hover states
- [ ] Focus state (blue ring) on textarea

### Emotion Widget
- [ ] **Current emotion displayed prominently**
  - [ ] Emotion name (e.g., "Joy", "Curiosity")
  - [ ] Emotion icon or emoji
  - [ ] Color-coded background
  - [ ] Intensity meter (0-100%)
- [ ] **PAD values** visible:
  - [ ] Pleasure score with label
  - [ ] Arousal score with label  
  - [ ] Dominance score with label
  - [ ] Values formatted (-1.0 to 1.0)
- [ ] **Learning indicators**:
  - [ ] Learning readiness (Ready/Not Ready with color)
  - [ ] Cognitive load (Low/Medium/High with gauge)
  - [ ] Flow state (In Flow / Not In Flow)
- [ ] Minimize/expand toggle
- [ ] Smooth animation on updates
- [ ] No layout shift when emotion changes

### Typing Indicator
- [ ] "AI is typing..." text appears
- [ ] Three bouncing dots animation
- [ ] Positioned in message area
- [ ] Disappears when response arrives
- [ ] Smooth fade in/out

---

## 🎭 PHASE 4: EMOTION VISUALIZATION - VISUAL CHECKLIST

### Emotion Chart
- [ ] Line chart visible
- [ ] X-axis labeled (time: hours/days/weeks)
- [ ] Y-axis labeled (intensity: 0-100%)
- [ ] Multiple emotion lines (color-coded)
- [ ] Legend showing emotion names + colors
- [ ] Grid lines (subtle, not distracting)
- [ ] Smooth curves (not jagged)
- [ ] Tooltips on hover (timestamp + emotion + value)
- [ ] Responsive to container size
- [ ] Chart title/heading
- [ ] Time range selector (Last Hour, Day, Week, Month)

### PAD Model Visualization
- [ ] 3D or 2D plot/chart
- [ ] Three axes clearly labeled:
  - [ ] Pleasure axis (-1 to +1)
  - [ ] Arousal axis (-1 to +1)
  - [ ] Dominance axis (-1 to +1)
- [ ] Current position marker (dot or icon)
- [ ] Historical trail/path (showing movement)
- [ ] Quadrant labels (e.g., "High Energy + Positive")
- [ ] Color coding for quadrants
- [ ] Interactive controls (zoom, rotate)
- [ ] Legend explaining axes
- [ ] Chart title

### Emotion Timeline
- [ ] Chronological list (newest first or oldest first)
- [ ] Each entry shows:
  - [ ] Timestamp (formatted nicely)
  - [ ] Emotion name
  - [ ] Emotion icon/emoji
  - [ ] Duration (how long emotion lasted)
  - [ ] Intensity (bar or percentage)
  - [ ] Message excerpt that triggered emotion
- [ ] Scrollable list
- [ ] Alternating row colors or dividers
- [ ] Filter dropdown (All Emotions / Specific emotion)
- [ ] Export button (CSV or JSON)

### Learning Readiness Indicator
- [ ] Visual gauge or progress bar
- [ ] States clearly labeled:
  - [ ] Ready (green)
  - [ ] Partially Ready (yellow)
  - [ ] Not Ready (red)
- [ ] Icon matching state (checkmark, warning, X)
- [ ] Percentage or numeric score
- [ ] Explanation text ("Great time to tackle challenging topics!")
- [ ] Smooth color transitions
- [ ] Tooltip with more details

### Cognitive Load Meter
- [ ] Gauge or thermometer visual
- [ ] Levels clearly marked:
  - [ ] Low (green)
  - [ ] Medium (yellow)
  - [ ] High (orange)
  - [ ] Overload (red)
- [ ] Current needle/pointer position
- [ ] Percentage value (0-100%)
- [ ] Recommendation text based on load
- [ ] Color-coded background

### Flow State Indicator
- [ ] Binary indicator (In Flow / Not In Flow)
- [ ] Visual cue (pulsing circle, glow effect, or checkmark)
- [ ] Color: Blue/purple when in flow, gray when not
- [ ] Flow score (0-100%)
- [ ] Time counter ("In flow for 15 minutes")
- [ ] Factors display (What's contributing to flow)

---

## 📊 PHASE 5: DASHBOARD - VISUAL CHECKLIST

### Dashboard Layout
- [ ] Page title: "Dashboard" or "Analytics"
- [ ] Grid layout (2-3 columns on desktop)
- [ ] Card-based design
- [ ] Consistent card shadows/borders
- [ ] Proper spacing between cards
- [ ] Responsive (stacks on mobile)

### Stats Cards
- [ ] **Total Learning Time**:
  - [ ] Icon (clock)
  - [ ] Large number (hours:minutes)
  - [ ] Trend arrow (up/down)
  - [ ] Percentage change
  - [ ] Comparison text ("vs last week")
- [ ] **Messages Sent**:
  - [ ] Icon (chat bubble)
  - [ ] Count
  - [ ] Trend indicator
- [ ] **Topics Covered**:
  - [ ] Icon (book/brain)
  - [ ] Count
  - [ ] Top 3 topics listed
- [ ] **Current Streak**:
  - [ ] Icon (fire)
  - [ ] Days count
  - [ ] Streak calendar visual

### Charts
- [ ] **Learning Velocity Chart**:
  - [ ] Line or bar chart
  - [ ] Labeled axes
  - [ ] Tooltips on hover
  - [ ] Smooth animations
  - [ ] Date range selector
- [ ] **Topic Mastery Radar**:
  - [ ] Spider/radar chart
  - [ ] Multiple topics (5-8)
  - [ ] Filled area
  - [ ] Labeled axes
  - [ ] Color-coded
- [ ] **Daily Activity Heatmap**:
  - [ ] Calendar grid
  - [ ] Color intensity based on activity
  - [ ] Tooltips showing exact values

### Filters
- [ ] Time range buttons:
  - [ ] Today
  - [ ] This Week
  - [ ] This Month
  - [ ] All Time
- [ ] Active filter highlighted (blue background)
- [ ] Custom date picker (calendar icon)

---

## 🎮 PHASE 6: GAMIFICATION - VISUAL CHECKLIST

### Achievement Badges
- [ ] Grid or masonry layout
- [ ] **Locked badges**:
  - [ ] Grayed out / desaturated
  - [ ] Lock icon overlay
  - [ ] Badge name visible
  - [ ] "How to unlock" tooltip
- [ ] **Unlocked badges**:
  - [ ] Full color
  - [ ] Shine/glow effect
  - [ ] Badge name
  - [ ] Unlock date
- [ ] Progress bars (for partial progress)
- [ ] Category tabs (Streak, Mastery, Milestones, Social, Special)
- [ ] Badge count (e.g., "12/17 unlocked")
- [ ] Hover effect (scale up, show details)
- [ ] Click to view details modal

### Streak Counter
- [ ] Fire icon (🔥) next to number
- [ ] Current streak number (large font)
- [ ] "day streak" label
- [ ] Best streak displayed (smaller, "Best: 45 days")
- [ ] Streak calendar:
  - [ ] Grid of days (7x4 or 7x6)
  - [ ] Active days highlighted (green/blue)
  - [ ] Today marked (border or different color)
  - [ ] Future days grayed out
- [ ] Streak freeze count (if available)
- [ ] "Don't break your streak!" motivational text

### Level Progress
- [ ] Current level displayed (large number + "Level X")
- [ ] Level icon or badge
- [ ] XP progress bar:
  - [ ] Current XP / Total XP needed
  - [ ] Percentage filled
  - [ ] Smooth animation on XP gain
  - [ ] Color gradient (blue to purple)
- [ ] "X XP to level Y" text
- [ ] Next level benefits preview
- [ ] Level-up animation (confetti, particles)

### Leaderboard
- [ ] Table or card list
- [ ] Columns:
  - [ ] Rank (#1, #2, #3...)
  - [ ] Avatar
  - [ ] Username
  - [ ] Level
  - [ ] XP or Score
- [ ] Top 3 podium style (1st, 2nd, 3rd with gold, silver, bronze)
- [ ] Current user row highlighted (blue background)
- [ ] Filter tabs: Global, Friends, This Week, All Time
- [ ] Pagination controls
- [ ] "You're rank #X" banner at top

---

## 🎤 PHASE 7: VOICE - VISUAL CHECKLIST

### Voice Input Button
- [ ] Microphone icon
- [ ] Located in message input area
- [ ] **States**:
  - [ ] Idle (gray)
  - [ ] Active/Recording (red, pulsing)
  - [ ] Processing (blue, loading spinner)
- [ ] Tooltip: "Hold to record" or "Click to record"
- [ ] Permission prompt (browser native)
- [ ] Permission denied error message

### Recording Indicator
- [ ] Pulsing red dot or circle
- [ ] Audio level bars (visualizer)
- [ ] Timer (00:15, 00:30...)
- [ ] Cancel button (X)
- [ ] "Recording..." text
- [ ] Waveform animation

### Transcription Display
- [ ] Transcribed text appears in input field
- [ ] User can edit before sending
- [ ] "Transcription complete" feedback
- [ ] Loading spinner during transcription
- [ ] Error message if transcription fails

### TTS (Text-to-Speech)
- [ ] Speaker icon on AI messages
- [ ] **States**:
  - [ ] Default (gray speaker icon)
  - [ ] Playing (blue speaker with sound waves)
  - [ ] Paused (pause icon)
- [ ] Play/pause toggle
- [ ] Audio progress bar (optional)
- [ ] Volume control (optional)
- [ ] Speed control (1x, 1.5x, 2x) (optional)
- [ ] Waveform visualization while playing

---

## ⚙️ PHASE 8: SETTINGS - VISUAL CHECKLIST

### Settings Modal/Page
- [ ] Modal overlay (semi-transparent black)
- [ ] Modal centered on screen
- [ ] Close button (X) in top-right
- [ ] Tab navigation (horizontal tabs at top)
- [ ] Active tab highlighted (blue underline)
- [ ] Tab content area
- [ ] Form elements properly styled

### Account Settings Tab
- [ ] Avatar upload area:
  - [ ] Current avatar displayed (circle)
  - [ ] "Change avatar" button
  - [ ] Hover overlay with camera icon
- [ ] Full name input (editable)
- [ ] Email display (grayed out if not editable)
- [ ] Change password button → opens modal
- [ ] Delete account button (red, at bottom)
- [ ] "Save changes" button (blue)

### Preferences Tab
- [ ] Theme selector:
  - [ ] Radio buttons or toggle
  - [ ] Options: Dark, Light, Auto
  - [ ] Icons for each option
- [ ] Language dropdown
- [ ] Time zone dropdown
- [ ] Learning goal slider or input
- [ ] Difficulty preference (radio buttons: Easy, Medium, Hard)
- [ ] Voice personality dropdown
- [ ] "Save preferences" button

### Notifications Tab
- [ ] Toggle switches for each notification type:
  - [ ] Email notifications
  - [ ] Push notifications
  - [ ] Daily reminders
  - [ ] Streak alerts
  - [ ] Achievement unlocks
  - [ ] Weekly summaries
- [ ] Each toggle clearly labeled
- [ ] Toggle on: blue background, white circle on right
- [ ] Toggle off: gray background, white circle on left
- [ ] Quiet hours time picker (start time - end time)

### Privacy Tab
- [ ] Checkboxes or toggles for:
  - [ ] Data sharing preferences
  - [ ] Leaderboard visibility
  - [ ] Profile public/private
- [ ] Links (blue underlined text):
  - [ ] Privacy policy
  - [ ] Terms of service
- [ ] "Export my data" button
- [ ] "Delete my data" button (red)

---

## 📱 PHASE 11: RESPONSIVE DESIGN - VISUAL CHECKLIST

### Mobile (320px - 767px)
- [ ] Hamburger menu icon (☰) in header
- [ ] Navigation drawer slides in from side
- [ ] Vertical stacking of all elements
- [ ] No horizontal overflow/scrolling
- [ ] Touch-friendly buttons (min 44x44px)
- [ ] Font sizes readable (min 16px for body text)
- [ ] Forms single column
- [ ] Emotion widget collapses to compact view
- [ ] Chat messages full width
- [ ] Bottom navigation bar (optional)

### Tablet (768px - 1023px)
- [ ] Two-column layout where appropriate
- [ ] Sidebar toggleable (hamburger + sidebar)
- [ ] Charts resize properly
- [ ] Touch and mouse input both work
- [ ] Landscape orientation handled

### Desktop (1024px+)
- [ ] Multi-column layout (sidebar + main + right panel)
- [ ] Sidebar always visible
- [ ] Hover states on all interactive elements
- [ ] Keyboard shortcuts work
- [ ] Focus states visible

---

## ♿ PHASE 12: ACCESSIBILITY - VISUAL CHECKLIST

### Color Contrast
- [ ] Text on background ≥ 4.5:1 ratio
- [ ] UI components ≥ 3:1 ratio
- [ ] Links distinguishable (not just color)
- [ ] Error messages in red + icon

### Focus States
- [ ] Blue ring visible on all interactive elements
- [ ] Tab order logical (top to bottom, left to right)
- [ ] No focus traps
- [ ] Skip to main content link visible on tab

### Visual Indicators
- [ ] Loading states (spinner or skeleton)
- [ ] Success messages (green + checkmark icon)
- [ ] Error messages (red + X icon)
- [ ] Warning messages (yellow + warning icon)
- [ ] Required fields marked with * and label

### Text & Typography
- [ ] Text resizable to 200% without breaking layout
- [ ] Line height ≥ 1.5
- [ ] Paragraph spacing ≥ 2x font size
- [ ] No text in images (or alt text provided)

---

## 📝 NOTES

**Usage**: Check off items as you verify them during testing  
**Screenshot**: Capture screenshot when item fails  
**Document**: Add notes in FRONTEND_TESTING_PROGRESS.md

**Color Reference**:
- Primary Blue: #3B82F6
- Primary Purple: #9333EA
- Success Green: #10B981
- Error Red: #EF4444
- Warning Yellow: #F59E0B
- Gray: #6B7280
- Dark Background: #0F172A / #1E293B

---

**Last Updated**: October 29, 2025  
**Use with**: FRONTEND_TESTING_PROGRESS.md + TESTING_QUICK_REFERENCE.md

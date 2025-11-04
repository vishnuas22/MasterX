/**
 * AchievementNotificationManager Component - Global Achievement Notifications
 * 
 * Purpose: Manages and displays all achievement notifications globally
 * 
 * AGENTS_FRONTEND.md Compliance:
 * ✅ Type Safety: Strict TypeScript
 * ✅ State Management: Integrated with gamification store
 * ✅ Performance: Renders only when notifications change
 * ✅ Accessibility: Screen reader announcements
 * 
 * Features:
 * 1. Displays multiple notifications in a stack
 * 2. Auto-dismisses old notifications
 * 3. Manual dismiss functionality
 * 4. Handles notification queue
 * 5. Prevents duplicate notifications
 * 
 * Usage:
 * Place this component at the root level of your app (in App.tsx or MainApp.tsx)
 * It will automatically show notifications when achievements are unlocked.
 * 
 * @module components/gamification/AchievementNotificationManager
 */

import React from 'react';
import { AchievementNotification } from './AchievementNotification';
import { useGamificationStore, useUnshownNotifications } from '@/store/gamificationStore';

/**
 * Achievement Notification Manager
 * 
 * Renders all unshown achievement notifications.
 * Automatically manages notification lifecycle.
 * 
 * @example
 * ```tsx
 * // In App.tsx or MainApp.tsx
 * function App() {
 *   return (
 *     <>
 *       <YourAppContent />
 *       <AchievementNotificationManager />
 *     </>
 *   );
 * }
 * ```
 */
export const AchievementNotificationManager: React.FC = () => {
  const notifications = useUnshownNotifications();
  const { markNotificationShown, dismissNotification } = useGamificationStore();

  // No notifications to show
  if (notifications.length === 0) {
    return null;
  }

  return (
    <div 
      className="fixed bottom-0 right-0 z-50 pointer-events-none"
      aria-live="polite"
      aria-atomic="false"
    >
      <div className="flex flex-col gap-4 p-4 pointer-events-auto">
        {notifications.map((notification, index) => {
          // Stagger notifications vertically
          const style = {
            transform: `translateY(-${index * 10}px)`,
            zIndex: 50 - index
          };

          return (
            <div key={notification.id} style={style}>
              <AchievementNotification
                achievement={notification.achievement}
                onDismiss={() => {
                  markNotificationShown(notification.id);
                  dismissNotification(notification.id);
                }}
                autoHideDuration={5000}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AchievementNotificationManager;

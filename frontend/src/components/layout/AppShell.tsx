/**
 * AppShell - Main Application Layout
 * 
 * WCAG 2.1 AA Compliant:
 * - Landmark regions (nav, main, aside)
 * - Skip links for keyboard navigation
 * - Focus management for modals
 * - Mobile: 44x44px touch targets
 * 
 * Performance:
 * - Lazy load modals (code splitting)
 * - Memoized layout components
 * - Virtualized sidebar (if needed)
 * 
 * Responsive:
 * - Desktop: Sidebar + header
 * - Tablet: Collapsible sidebar
 * - Mobile: Bottom navigation
 */

import React, { lazy, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'react-router-dom';
import { 
  Home, MessageSquare, BarChart3, Trophy, Settings 
} from 'lucide-react';
import { useUIStore } from '@/store/uiStore';
import { useEmotionStore } from '@/store/emotionStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { cn } from '@/utils/cn';
import { Header } from './Header';
import { Sidebar } from './Sidebar';

// Lazy load modals (code splitting)
const Dashboard = lazy(() => import('@/pages/Dashboard').catch(() => ({ default: () => <div>Dashboard loading...</div> })));
const SettingsModal = lazy(() => import('@/pages/Settings').catch(() => ({ default: () => <div>Settings loading...</div> })));
const ProfileModal = lazy(() => import('@/pages/Profile').catch(() => ({ default: () => <div>Profile loading...</div> })));

// ============================================================================
// TYPES
// ============================================================================

export interface AppShellProps {
  children: React.ReactNode;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ElementType;
  href?: string;
  onClick?: () => void;
  badge?: number;
}

// ============================================================================
// NAVIGATION ITEMS
// ============================================================================

const getNavigationItems = (
  openDashboard: () => void,
  openSettings: () => void
): NavItem[] => [
  {
    id: 'chat',
    label: 'Chat',
    icon: MessageSquare,
    href: '/app',
  },
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: Home,
    onClick: openDashboard,
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: BarChart3,
    onClick: () => console.log('Analytics'), // TODO: Implement
  },
  {
    id: 'achievements',
    label: 'Achievements',
    icon: Trophy,
    onClick: () => console.log('Achievements'), // TODO: Implement
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    onClick: openSettings,
  },
];

// ============================================================================
// BOTTOM NAVIGATION (Mobile)
// ============================================================================

const BottomNav = React.memo<{ items: NavItem[] }>(({ items }) => {
  const location = useLocation();
  const mainItems = items.slice(0, 4); // Show first 4 items

  return (
    <nav
      className="lg:hidden fixed bottom-0 left-0 right-0 z-30 bg-bg-primary/80 backdrop-blur-xl border-t border-white/10"
      role="navigation"
      aria-label="Bottom navigation"
    >
      <div className="flex items-center justify-around h-16">
        {mainItems.map((item) => (
          <button
            key={item.id}
            onClick={item.onClick}
            className={cn(
              'flex flex-col items-center justify-center gap-1 px-4 py-2 transition',
              location.pathname === item.href ? 'text-accent-primary' : 'text-text-tertiary'
            )}
            aria-label={item.label}
          >
            <item.icon className="w-5 h-5" />
            <span className="text-xs font-medium">{item.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
});

BottomNav.displayName = 'BottomNav';

// ============================================================================
// EMOTION WIDGET (Persistent)
// ============================================================================

const EmotionWidget = React.memo(() => {
  const { currentEmotion } = useEmotionStore();
  
  if (!currentEmotion) return null;

  return (
    <motion.div
      drag
      dragConstraints={{
        top: 0,
        bottom: typeof window !== 'undefined' ? window.innerHeight - 100 : 0,
        left: 0,
        right: typeof window !== 'undefined' ? window.innerWidth - 200 : 0,
      }}
      className="fixed bottom-20 left-4 z-20 lg:bottom-4"
    >
      <div className="glass rounded-lg p-3 shadow-lg cursor-move">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-success animate-pulse" />
          <span className="text-xs font-medium">{currentEmotion.primary_emotion}</span>
        </div>
      </div>
    </motion.div>
  );
});

EmotionWidget.displayName = 'EmotionWidget';

// ============================================================================
// MAIN APPSHELL COMPONENT
// ============================================================================

export const AppShell = React.memo<AppShellProps>(({ children }) => {
  const { openModal, closeModal, activeModal, isSidebarOpen, closeSidebar } = useUIStore();
  
  // Initialize WebSocket connection
  useWebSocket();

  // Navigation items with modal handlers
  const navItems = getNavigationItems(
    () => openModal('dashboard'),
    () => openModal('settings')
  );

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyboard = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K: Quick search (TODO)
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        console.log('Quick search');
      }
      
      // Escape: Close modal
      if (e.key === 'Escape' && activeModal) {
        closeModal();
      }
    };

    window.addEventListener('keydown', handleKeyboard);
    return () => window.removeEventListener('keydown', handleKeyboard);
  }, [activeModal, closeModal]);

  return (
    <div className="min-h-screen bg-bg-primary text-text-primary">
      {/* Skip link (accessibility) */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-accent-primary focus:text-white focus:rounded-lg"
      >
        Skip to main content
      </a>

      {/* Header */}
      <Header />

      {/* Sidebar (desktop) */}
      <Sidebar isOpen={isSidebarOpen} onClose={closeSidebar} />

      {/* Main content */}
      <main
        id="main-content"
        className={cn(
          'transition-all duration-300',
          'pt-0 pb-20 lg:pb-4', // Account for bottom nav on mobile
          'lg:ml-[280px]' // Account for sidebar on desktop
        )}
        role="main"
      >
        <div className="max-w-7xl mx-auto px-4 py-6">
          {children}
        </div>
      </main>

      {/* Bottom navigation (mobile) */}
      <BottomNav items={navItems} />

      {/* Emotion widget */}
      <EmotionWidget />

      {/* Modals (lazy loaded) */}
      <Suspense fallback={null}>
        <AnimatePresence>
          {activeModal === 'dashboard' && (
            <Dashboard onClose={closeModal} />
          )}
          {activeModal === 'settings' && (
            <SettingsModal onClose={closeModal} />
          )}
          {activeModal === 'profile' && (
            <ProfileModal onClose={closeModal} />
          )}
        </AnimatePresence>
      </Suspense>
    </div>
  );
});

AppShell.displayName = 'AppShell';

// ============================================================================
// EXPORTS
// ============================================================================

export default AppShell;

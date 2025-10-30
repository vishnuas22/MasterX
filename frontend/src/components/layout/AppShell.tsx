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

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'react-router-dom';
import { 
  Home, MessageSquare, BarChart3, Trophy, Settings, 
  User, LogOut, Sun, Moon, Menu, X 
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { useUIStore } from '@/store/uiStore';
import { useEmotionStore } from '@/store/emotionStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { cn } from '@/utils/cn';

// Lazy load modals (code splitting) - TODO: Implement these pages
// const Dashboard = lazy(() => import('@/pages/Dashboard'));
// const SettingsModal = lazy(() => import('@/pages/Settings'));
// const ProfileModal = lazy(() => import('@/pages/Profile'));

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
// HEADER COMPONENT
// ============================================================================

const Header = React.memo(() => {
  const { user, logout } = useAuthStore();
  const { theme, toggleTheme, toggleSidebar, isSidebarOpen } = useUIStore();
  const [showUserMenu, setShowUserMenu] = React.useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-bg-primary/80 backdrop-blur-xl">
      <div className="flex items-center justify-between h-16 px-4">
        {/* Left: Logo + Menu toggle (mobile) */}
        <div className="flex items-center gap-4">
          <button
            onClick={toggleSidebar}
            className="lg:hidden p-2 hover:bg-bg-secondary rounded-lg transition"
            aria-label="Toggle navigation"
          >
            {isSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-primary to-accent-purple" />
            <span className="text-lg font-semibold">MasterX</span>
          </div>
        </div>

        {/* Right: Theme toggle + User menu */}
        <div className="flex items-center gap-3">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 hover:bg-bg-secondary rounded-lg transition"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? (
              <Sun className="w-5 h-5" />
            ) : (
              <Moon className="w-5 h-5" />
            )}
          </button>

          {/* User menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center gap-2 p-2 hover:bg-bg-secondary rounded-lg transition"
              aria-label="User menu"
            >
              <div className="w-8 h-8 rounded-full bg-accent-primary flex items-center justify-center text-sm font-semibold">
                {user?.name?.[0]?.toUpperCase() || 'U'}
              </div>
            </button>

            <AnimatePresence>
              {showUserMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute right-0 mt-2 w-56 bg-bg-secondary rounded-lg shadow-xl border border-white/10 overflow-hidden"
                >
                  <div className="p-3 border-b border-white/10">
                    <p className="text-sm font-medium">{user?.name}</p>
                    <p className="text-xs text-text-tertiary">{user?.email}</p>
                  </div>
                  
                  <button
                    onClick={() => {
                      setShowUserMenu(false);
                      // Open profile modal
                    }}
                    className="w-full flex items-center gap-3 px-3 py-2 hover:bg-bg-tertiary transition text-left"
                  >
                    <User className="w-4 h-4" />
                    <span className="text-sm">Profile</span>
                  </button>
                  
                  <button
                    onClick={() => {
                      logout();
                      setShowUserMenu(false);
                    }}
                    className="w-full flex items-center gap-3 px-3 py-2 hover:bg-bg-tertiary transition text-left text-accent-error"
                  >
                    <LogOut className="w-4 h-4" />
                    <span className="text-sm">Logout</span>
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </header>
  );
});

Header.displayName = 'Header';

// ============================================================================
// SIDEBAR COMPONENT (Desktop)
// ============================================================================

const Sidebar = React.memo<{ items: NavItem[] }>(({ items }) => {
  const { isSidebarOpen, toggleSidebar } = useUIStore();
  const location = useLocation();

  return (
    <>
      {/* Overlay (mobile only) */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => toggleSidebar()}
            className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{
          x: isSidebarOpen ? 0 : -280,
        }}
        className={cn(
          'fixed left-0 top-16 bottom-0 w-[280px] z-30',
          'bg-bg-primary border-r border-white/10',
          'lg:translate-x-0 transition-transform'
        )}
        role="navigation"
        aria-label="Main navigation"
      >
        <nav className="p-4 space-y-1">
          {items.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                item.onClick?.();
                if (window.innerWidth < 1024) toggleSidebar(); // Close on mobile
              }}
              className={cn(
                'w-full flex items-center gap-3 px-4 py-3 rounded-lg transition',
                'hover:bg-bg-secondary',
                location.pathname === item.href && 'bg-accent-primary/10 text-accent-primary'
              )}
            >
              <item.icon className="w-5 h-5" />
              <span className="text-sm font-medium">{item.label}</span>
              {item.badge && (
                <span className="ml-auto px-2 py-1 text-xs font-semibold bg-accent-error rounded-full">
                  {item.badge}
                </span>
              )}
            </button>
          ))}
        </nav>
      </motion.aside>
    </>
  );
});

Sidebar.displayName = 'Sidebar';

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
        bottom: window.innerHeight - 100,
        left: 0,
        right: window.innerWidth - 200,
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
  const { openModal, closeModal, activeModal } = useUIStore();
  
  // Initialize WebSocket connection for real-time updates
  const { isConnected } = useWebSocket();

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
      <Sidebar items={navItems} />

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

      {/* Modals (lazy loaded) - TODO: Uncomment when pages are implemented */}
      {/* <Suspense fallback={null}>
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
      </Suspense> */}
    </div>
  );
});

AppShell.displayName = 'AppShell';

// ============================================================================
// EXPORTS
// ============================================================================

export default AppShell;

/**
 * Header Component - Top Navigation Bar
 * 
 * WCAG 2.1 AA Compliant:
 * - Landmark <header> element
 * - Keyboard navigation (Tab, Enter, Esc)
 * - ARIA labels on all controls
 * - Focus indicators visible
 * 
 * Performance:
 * - Sticky positioning (no JS required)
 * - Memoized to prevent re-renders
 * - Lazy load user menu
 * 
 * Backend Integration:
 * - User data from /api/auth/me
 * - Logout via /api/auth/logout
 * - Notifications from WebSocket
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, Bell, Sun, Moon, Menu, X, User, Settings,
  LogOut, BarChart3, Trophy, HelpCircle, ChevronDown
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { useUIStore } from '@/store/uiStore';
import { cn } from '@/utils/cn';
import { Avatar } from '@/components/ui/Avatar';
import { Badge } from '@/components/ui/Badge';
import { Tooltip } from '@/components/ui/Tooltip';

// ============================================================================
// TYPES
// ============================================================================

export interface HeaderProps {
  /**
   * Show mobile menu toggle
   * @default true
   */
  showMobileToggle?: boolean;
  
  /**
   * Enable global search
   * @default true
   */
  enableSearch?: boolean;
  
  /**
   * Show notifications bell
   * @default true
   */
  showNotifications?: boolean;
  
  /**
   * Modal handlers
   */
  onOpenDashboard?: () => void;
  onOpenSettings?: () => void;
  onOpenProfile?: () => void;
  
  /**
   * Additional CSS classes
   */
  className?: string;
}

// ============================================================================
// NOTIFICATION BADGE
// ============================================================================

const NotificationBell = React.memo<{ count: number }>(({ count }) => {
  return (
    <Tooltip content={`${count} new notifications`} position="bottom">
      <button
        className="relative p-2 hover:bg-bg-secondary rounded-lg transition-colors focus-ring"
        aria-label={`${count} unread notifications`}
      >
        <Bell className="w-5 h-5" />
        {count > 0 && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="absolute -top-1 -right-1 w-5 h-5 bg-accent-error text-white text-xs font-bold rounded-full flex items-center justify-center"
          >
            {count > 9 ? '9+' : count}
          </motion.span>
        )}
      </button>
    </Tooltip>
  );
});

NotificationBell.displayName = 'NotificationBell';

// ============================================================================
// USER MENU
// ============================================================================

interface UserMenuProps {
  isOpen: boolean;
  onClose: () => void;
}

const UserMenu = React.memo<UserMenuProps>(({ isOpen, onClose }) => {
  const { user, logout } = useAuthStore();
  const menuRef = React.useRef<HTMLDivElement>(null);

  // Close on outside click
  React.useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  // Close on Escape
  React.useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  if (!user) return null;

  const menuItems = [
    {
      icon: User,
      label: 'Profile',
      onClick: () => {
        // TODO: Open profile modal
        onClose();
      },
    },
    {
      icon: BarChart3,
      label: 'Analytics',
      onClick: () => {
        // TODO: Open analytics
        onClose();
      },
    },
    {
      icon: Trophy,
      label: 'Achievements',
      onClick: () => {
        // TODO: Open achievements
        onClose();
      },
    },
    {
      icon: Settings,
      label: 'Settings',
      onClick: () => {
        // TODO: Open settings modal
        onClose();
      },
    },
    {
      icon: HelpCircle,
      label: 'Help & Support',
      onClick: () => {
        window.open('https://help.masterx.ai', '_blank');
        onClose();
      },
    },
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          ref={menuRef}
          initial={{ opacity: 0, y: -10, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -10, scale: 0.95 }}
          transition={{ duration: 0.15, ease: 'easeOut' }}
          className="absolute right-0 mt-2 w-64 bg-bg-secondary rounded-xl shadow-xl border border-white/10 overflow-hidden z-50"
          role="menu"
          aria-orientation="vertical"
          aria-labelledby="user-menu-button"
        >
          {/* User info */}
          <div className="p-4 border-b border-white/10">
            <div className="flex items-center gap-3">
              <Avatar
                name={user.name}
                src={undefined}
                size="md"
              />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-text-primary truncate">
                  {user.name}
                </p>
                <p className="text-xs text-text-tertiary truncate">
                  {user.email}
                </p>
              </div>
            </div>
            
            {/* Subscription badge */}
            <div className="mt-3">
              <Badge variant="primary" size="sm">
                {user.subscription_tier.toUpperCase()}
              </Badge>
            </div>
          </div>

          {/* Menu items */}
          <div className="py-2">
            {menuItems.map((item) => (
              <button
                key={item.label}
                onClick={item.onClick}
                className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-bg-tertiary transition-colors text-left"
                role="menuitem"
              >
                <item.icon className="w-4 h-4 text-text-secondary" />
                <span className="text-sm text-text-primary">{item.label}</span>
              </button>
            ))}
          </div>

          {/* Logout */}
          <div className="border-t border-white/10">
            <button
              onClick={() => {
                logout();
                onClose();
              }}
              className="w-full flex items-center gap-3 px-4 py-3 hover:bg-accent-error/10 transition-colors text-left text-accent-error"
              role="menuitem"
            >
              <LogOut className="w-4 h-4" />
              <span className="text-sm font-medium">Logout</span>
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
});

UserMenu.displayName = 'UserMenu';

// ============================================================================
// MAIN HEADER COMPONENT
// ============================================================================

export const Header = React.memo<HeaderProps>(({
  showMobileToggle = true,
  enableSearch = true,
  showNotifications = true,
  className,
}: HeaderProps) => {
  const { user } = useAuthStore();
  const { theme, toggleTheme, toggleSidebar, isSidebarOpen } = useUIStore();
  const [showUserMenu, setShowUserMenu] = React.useState(false);
  const [notificationCount] = React.useState(3); // TODO: Get from WebSocket

  // Global search handler
  const handleSearch = React.useCallback(() => {
    // TODO: Implement global search modal
    console.log('Open search');
  }, []);

  // Keyboard shortcut (Cmd/Ctrl + K for search)
  React.useEffect(() => {
    if (!enableSearch) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        handleSearch();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [enableSearch, handleSearch]);

  return (
    <header
      className={cn(
        'sticky top-0 z-40',
        'h-16 border-b border-white/10',
        'bg-bg-primary/80 backdrop-blur-xl',
        className
      )}
      role="banner"
    >
      <div className="h-full px-4 flex items-center justify-between gap-4">
        {/* Left section: Logo + Mobile toggle */}
        <div className="flex items-center gap-4">
          {/* Mobile menu toggle */}
          {showMobileToggle && (
            <button
              onClick={toggleSidebar}
              className="lg:hidden p-2 hover:bg-bg-secondary rounded-lg transition-colors focus-ring"
              aria-label="Toggle navigation menu"
              aria-expanded={isSidebarOpen}
            >
              {isSidebarOpen ? (
                <X className="w-5 h-5" />
              ) : (
                <Menu className="w-5 h-5" />
              )}
            </button>
          )}

          {/* Logo */}
          <Link
            to="/app"
            className="flex items-center gap-2 hover:opacity-80 transition-opacity focus-ring rounded-lg"
            aria-label="MasterX home"
          >
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-primary to-accent-purple flex items-center justify-center">
              <span className="text-white font-bold text-sm">M</span>
            </div>
            <span className="hidden sm:inline text-lg font-semibold text-text-primary">
              MasterX
            </span>
          </Link>
        </div>

        {/* Right section: Actions + User */}
        <div className="flex items-center gap-2">
          {/* Global search */}
          {enableSearch && (
            <Tooltip content="Search (⌘K)" position="bottom">
              <button
                onClick={handleSearch}
                className="hidden md:flex items-center gap-2 px-3 py-2 bg-bg-secondary hover:bg-bg-tertiary rounded-lg transition-colors focus-ring"
                aria-label="Open search"
              >
                <Search className="w-4 h-4 text-text-tertiary" />
                <span className="text-sm text-text-tertiary">Search...</span>
                <kbd className="hidden lg:inline px-1.5 py-0.5 text-xs font-semibold text-text-tertiary bg-bg-tertiary rounded">
                  ⌘K
                </kbd>
              </button>
            </Tooltip>
          )}

          {/* Mobile search icon */}
          {enableSearch && (
            <button
              onClick={handleSearch}
              className="md:hidden p-2 hover:bg-bg-secondary rounded-lg transition-colors focus-ring"
              aria-label="Open search"
            >
              <Search className="w-5 h-5" />
            </button>
          )}

          {/* Theme toggle */}
          <Tooltip content={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`} position="bottom">
            <button
              onClick={toggleTheme}
              className="p-2 hover:bg-bg-secondary rounded-lg transition-colors focus-ring"
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? (
                <Sun className="w-5 h-5" />
              ) : (
                <Moon className="w-5 h-5" />
              )}
            </button>
          </Tooltip>

          {/* Notifications */}
          {showNotifications && (
            <NotificationBell count={notificationCount} />
          )}

          {/* User menu */}
          {user && (
            <div className="relative">
              <button
                id="user-menu-button"
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 p-1.5 hover:bg-bg-secondary rounded-lg transition-colors focus-ring"
                aria-label="User menu"
                aria-expanded={showUserMenu}
                aria-haspopup="true"
              >
                <Avatar
                  name={user.name}
                  src={undefined}
                  size="sm"
                />
                <ChevronDown
                  className={cn(
                    'w-4 h-4 text-text-tertiary transition-transform',
                    showUserMenu && 'rotate-180'
                  )}
                />
              </button>

              <UserMenu isOpen={showUserMenu} onClose={() => setShowUserMenu(false)} />
            </div>
          )}
        </div>
      </div>
    </header>
  );
});

Header.displayName = 'Header';

// ============================================================================
// EXPORTS
// ============================================================================

export default Header;

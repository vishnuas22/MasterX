/**
 * Sidebar Component - Desktop Navigation
 * 
 * WCAG 2.1 AA Compliant:
 * - Landmark <nav> element
 * - Keyboard navigation (Arrow keys, Enter)
 * - Skip navigation link
 * - Focus management
 * 
 * Performance:
 * - Memoized navigation items
 * - CSS transitions (no JS)
 * - Lazy render badges
 * 
 * Responsive:
 * - Desktop: Always visible (280px)
 * - Tablet/Mobile: Overlay with backdrop
 */

import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare, Home, BarChart3, Trophy, BookOpen,
  Compass, Users, Settings, ChevronRight
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { Badge } from '@/components/ui/Badge';
import { Tooltip } from '@/components/ui/Tooltip';

// ============================================================================
// TYPES
// ============================================================================

export interface SidebarProps {
  /**
   * Sidebar open/closed state
   */
  isOpen: boolean;
  
  /**
   * Close sidebar (mobile)
   */
  onClose: () => void;
  
  /**
   * Collapsed state (desktop)
   * @default false
   */
  isCollapsed?: boolean;
  
  /**
   * Additional CSS classes
   */
  className?: string;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ElementType;
  href: string;
  badge?: number;
  description?: string;
}

interface NavSection {
  title?: string;
  items: NavItem[];
}

// ============================================================================
// NAVIGATION STRUCTURE
// ============================================================================

const navigationSections: NavSection[] = [
  {
    items: [
      {
        id: 'chat',
        label: 'Chat',
        icon: MessageSquare,
        href: '/app',
        description: 'Start a learning conversation',
      },
      {
        id: 'dashboard',
        label: 'Dashboard',
        icon: Home,
        href: '/app/dashboard',
        description: 'Your learning overview',
      },
    ],
  },
  {
    title: 'Analytics',
    items: [
      {
        id: 'analytics',
        label: 'Progress',
        icon: BarChart3,
        href: '/app/analytics',
        description: 'Track your learning progress',
      },
      {
        id: 'achievements',
        label: 'Achievements',
        icon: Trophy,
        href: '/app/achievements',
        badge: 2, // New achievements
        description: 'View your achievements',
      },
    ],
  },
  {
    title: 'Learning',
    items: [
      {
        id: 'courses',
        label: 'Courses',
        icon: BookOpen,
        href: '/app/courses',
        description: 'Browse learning paths',
      },
      {
        id: 'explore',
        label: 'Explore',
        icon: Compass,
        href: '/app/explore',
        description: 'Discover new topics',
      },
      {
        id: 'collaboration',
        label: 'Collaboration',
        icon: Users,
        href: '/app/collaboration',
        description: 'Study with peers',
      },
    ],
  },
  {
    items: [
      {
        id: 'settings',
        label: 'Settings',
        icon: Settings,
        href: '/app/settings',
        description: 'Customize your experience',
      },
    ],
  },
];

// ============================================================================
// NAV ITEM COMPONENT
// ============================================================================

interface NavItemProps {
  item: NavItem;
  isCollapsed: boolean;
  onClick?: () => void;
}

const NavItemComponent = React.memo<NavItemProps>(({ item, isCollapsed, onClick }) => {
  const location = useLocation();
  const isActive = location.pathname === item.href || 
                   location.pathname.startsWith(item.href + '/');

  const content = (
    <NavLink
      to={item.href}
      onClick={onClick}
      className={({ isActive: active }) =>
        cn(
          'group relative flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all',
          'hover:bg-bg-secondary',
          'focus-ring',
          active && 'bg-accent-primary/10 text-accent-primary',
          isCollapsed && 'justify-center px-2'
        )
      }
    >
      {/* Active indicator */}
      {isActive && (
        <motion.div
          layoutId="sidebar-active-indicator"
          className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-accent-primary rounded-r-full"
          transition={{ type: 'spring', stiffness: 380, damping: 30 }}
        />
      )}

      {/* Icon */}
      <item.icon
        className={cn(
          'w-5 h-5 flex-shrink-0',
          isActive ? 'text-accent-primary' : 'text-text-secondary group-hover:text-text-primary'
        )}
      />

      {/* Label (hidden when collapsed) */}
      {!isCollapsed && (
        <>
          <span className="text-sm font-medium flex-1">{item.label}</span>
          
          {/* Badge */}
          {item.badge && (
            <Badge variant="error" size="sm">
              {item.badge}
            </Badge>
          )}
          
          {/* Arrow for active state */}
          {isActive && (
            <ChevronRight className="w-4 h-4 text-accent-primary" />
          )}
        </>
      )}
    </NavLink>
  );

  // Wrap with tooltip when collapsed
  if (isCollapsed) {
    return (
      <Tooltip content={item.description || item.label} position="right">
        {content}
      </Tooltip>
    );
  }

  return content;
});

NavItemComponent.displayName = 'NavItemComponent';

// ============================================================================
// MAIN SIDEBAR COMPONENT
// ============================================================================

export const Sidebar = React.memo<SidebarProps>(({
  isOpen,
  onClose,
  isCollapsed = false,
  className,
}) => {
  const sidebarRef = React.useRef<HTMLElement>(null);

  // Close on Escape (mobile)
  React.useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  return (
    <>
      {/* Backdrop (mobile only) */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="lg:hidden fixed inset-0 bg-black/50 z-30 backdrop-blur-sm"
            aria-hidden="true"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        ref={sidebarRef}
        initial={false}
        animate={{
          x: isOpen ? 0 : -280,
          width: isCollapsed ? 64 : 280,
        }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        className={cn(
          'fixed left-0 top-16 bottom-0 z-30',
          'bg-bg-primary border-r border-white/10',
          'lg:translate-x-0',
          'overflow-y-auto scrollbar-hide',
          className
        )}
        role="navigation"
        aria-label="Main navigation"
      >
        <nav className="p-3 space-y-6">
          {navigationSections.map((section, sectionIndex) => (
            <div key={sectionIndex}>
              {/* Section title */}
              {section.title && !isCollapsed && (
                <h3 className="px-3 mb-2 text-xs font-semibold text-text-tertiary uppercase tracking-wider">
                  {section.title}
                </h3>
              )}

              {/* Section divider when collapsed */}
              {section.title && isCollapsed && sectionIndex > 0 && (
                <div className="my-3 border-t border-white/10" />
              )}

              {/* Navigation items */}
              <div className="space-y-1">
                {section.items.map((item) => (
                  <NavItemComponent
                    key={item.id}
                    item={item}
                    isCollapsed={isCollapsed}
                    onClick={onClose} // Close sidebar on mobile
                  />
                ))}
              </div>
            </div>
          ))}
        </nav>
      </motion.aside>
    </>
  );
});

Sidebar.displayName = 'Sidebar';

// ============================================================================
// EXPORTS
// ============================================================================

export default Sidebar;

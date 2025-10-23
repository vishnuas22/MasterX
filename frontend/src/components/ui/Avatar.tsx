/**
 * Avatar Component - User Profile Picture
 * 
 * Following AGENTS_FRONTEND.md:
 * - Accessibility (alt text)
 * - Graceful fallback (initials)
 * - Status indicators
 * - Loading states
 */

import React, { useState } from 'react';
import { clsx } from 'clsx';

// ============================================================================
// TYPES
// ============================================================================

export type AvatarSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type AvatarStatus = 'online' | 'offline' | 'away' | 'busy';

export interface AvatarProps {
  /** User name (for alt text and initials fallback) */
  name: string;
  
  /** Image URL */
  src?: string;
  
  /** Avatar size */
  size?: AvatarSize;
  
  /** Status indicator */
  status?: AvatarStatus;
  
  /** Show status indicator */
  showStatus?: boolean;
  
  /** Custom className */
  className?: string;
  
  /** Test ID */
  'data-testid'?: string;
}

// ============================================================================
// SIZE STYLES
// ============================================================================

const sizeStyles: Record<AvatarSize, { container: string; text: string; status: string }> = {
  xs: {
    container: 'w-6 h-6',
    text: 'text-xs',
    status: 'w-2 h-2 border',
  },
  sm: {
    container: 'w-8 h-8',
    text: 'text-sm',
    status: 'w-2.5 h-2.5 border',
  },
  md: {
    container: 'w-10 h-10',
    text: 'text-base',
    status: 'w-3 h-3 border-2',
  },
  lg: {
    container: 'w-16 h-16',
    text: 'text-xl',
    status: 'w-4 h-4 border-2',
  },
  xl: {
    container: 'w-24 h-24',
    text: 'text-3xl',
    status: 'w-5 h-5 border-2',
  },
};

const statusColors: Record<AvatarStatus, string> = {
  online: 'bg-green-500',
  offline: 'bg-gray-400',
  away: 'bg-yellow-500',
  busy: 'bg-red-500',
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get initials from name
 */
const getInitials = (name: string): string => {
  if (!name) return '?';
  
  const parts = name.trim().split(' ');
  if (parts.length === 1) {
    return parts[0].substring(0, 2).toUpperCase();
  }
  
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
};

/**
 * Generate deterministic color from name
 */
const getColorFromName = (name: string): string => {
  const colors = [
    'bg-blue-500',
    'bg-green-500',
    'bg-purple-500',
    'bg-pink-500',
    'bg-yellow-500',
    'bg-red-500',
    'bg-indigo-500',
    'bg-teal-500',
  ];
  
  const hash = name.split('').reduce((acc, char) => {
    return acc + char.charCodeAt(0);
  }, 0);
  
  return colors[hash % colors.length];
};

// ============================================================================
// COMPONENT
// ============================================================================

export const Avatar: React.FC<AvatarProps> = ({
  name,
  src,
  size = 'md',
  status,
  showStatus = false,
  className,
  'data-testid': testId,
}) => {
  const [imageError, setImageError] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  
  const shouldShowImage = src && !imageError;
  const initials = getInitials(name);
  const bgColor = getColorFromName(name);

  return (
    <div
      data-testid={testId}
      className={clsx(
        'relative inline-block',
        sizeStyles[size].container,
        className
      )}
    >
      {/* Avatar container */}
      <div
        className={clsx(
          'w-full h-full rounded-full overflow-hidden',
          'flex items-center justify-center',
          'select-none',
          !shouldShowImage && bgColor
        )}
      >
        {shouldShowImage ? (
          <>
            {/* Image */}
            <img
              src={src}
              alt={name}
              onError={() => setImageError(true)}
              onLoad={() => setImageLoading(false)}
              className={clsx(
                'w-full h-full object-cover',
                imageLoading && 'opacity-0'
              )}
            />
            
            {/* Loading skeleton */}
            {imageLoading && (
              <div className="absolute inset-0 bg-gray-200 dark:bg-gray-700 animate-pulse" />
            )}
          </>
        ) : (
          /* Initials fallback */
          <span
            className={clsx(
              'font-semibold text-white',
              sizeStyles[size].text
            )}
          >
            {initials}
          </span>
        )}
      </div>

      {/* Status indicator */}
      {showStatus && status && (
        <span
          className={clsx(
            'absolute bottom-0 right-0',
            'rounded-full border-white dark:border-gray-800',
            sizeStyles[size].status,
            statusColors[status]
          )}
          aria-label={`Status: ${status}`}
        />
      )}
    </div>
  );
};

// ============================================================================
// HELPER COMPONENTS
// ============================================================================

/**
 * Avatar Group - Display multiple avatars with overlap
 */
export const AvatarGroup: React.FC<{
  users: Array<{ name: string; src?: string }>;
  max?: number;
  size?: AvatarSize;
}> = ({ users, max = 3, size = 'md' }) => {
  const visibleUsers = users.slice(0, max);
  const remainingCount = users.length - max;

  return (
    <div className="flex -space-x-2">
      {visibleUsers.map((user, index) => (
        <div
          key={index}
          className="ring-2 ring-white dark:ring-gray-800 rounded-full"
        >
          <Avatar
            name={user.name}
            src={user.src}
            size={size}
          />
        </div>
      ))}
      
      {remainingCount > 0 && (
        <div
          className={clsx(
            'flex items-center justify-center',
            'rounded-full bg-gray-200 dark:bg-gray-700',
            'ring-2 ring-white dark:ring-gray-800',
            'text-gray-600 dark:text-gray-400 font-medium',
            sizeStyles[size].container,
            sizeStyles[size].text
          )}
        >
          +{remainingCount}
        </div>
      )}
    </div>
  );
};

export default Avatar;

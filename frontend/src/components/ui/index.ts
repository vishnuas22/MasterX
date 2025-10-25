/**
 * UI Components - Barrel Export
 * 
 * Central export point for all basic UI components.
 * Enables clean imports: import { Button, Input, Modal } from '@components/ui';
 * 
 * Following AGENTS_FRONTEND.md:
 * - Barrel exports for cleaner imports
 * - Tree-shakeable (named exports)
 * - Type-safe
 */

// Basic Components
export { Button } from './Button';
export type { ButtonProps, ButtonVariant, ButtonSize } from './Button';

export { Input } from './Input';
export type { InputProps, InputSize, InputState } from './Input';

export { Modal } from './Modal';
export type { ModalProps, ModalSize } from './Modal';

export { Card } from './Card';
export type { CardProps, CardVariant, CardPadding } from './Card';

export { Badge, EmotionBadge, AchievementBadge } from './Badge';
export type { BadgeProps, BadgeVariant, BadgeSize } from './Badge';

export { Avatar, AvatarGroup } from './Avatar';
export type { AvatarProps, AvatarSize, AvatarStatus } from './Avatar';

export { Skeleton, SkeletonCard, SkeletonMessage, SkeletonList, SkeletonDashboard } from './Skeleton';
export type { SkeletonProps } from './Skeleton';

export { ToastContainer, toast, useToastStore } from './Toast';
export type { Toast, ToastVariant, ToastOptions } from './Toast';

export { Tooltip } from './Tooltip';
export type { TooltipProps, TooltipPosition } from './Tooltip';

/**
 * Usage:
 * ```tsx
 * import { Button, Input, Modal } from '@components/ui';
 * 
 * function MyComponent() {
 *   return (
 *     <>
 *       <Button variant="primary">Click me</Button>
 *       <Input label="Name" />
 *     </>
 *   );
 * }
 * ```
 */

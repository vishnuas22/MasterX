/**
 * Toggle Component - Switch/Toggle UI Element
 * 
 * WCAG 2.1 AA Compliant:
 * - Keyboard navigation (Space/Enter to toggle)
 * - ARIA attributes for screen readers
 * - Focus indicators
 * - Clear visual states
 * 
 * Features:
 * - Smooth animations
 * - Disabled state
 * - Custom labels
 * - Accessible keyboard support
 */

import React, { useId } from 'react';
import { cn } from '@/utils/cn';

export interface ToggleProps {
  checked?: boolean;
  onChange?: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
  className?: string;
  'data-testid'?: string;
}

export const Toggle: React.FC<ToggleProps> = ({
  checked = false,
  onChange,
  disabled = false,
  label,
  className,
  'data-testid': testId
}) => {
  const id = useId();

  const handleToggle = () => {
    if (!disabled && onChange) {
      onChange(!checked);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      handleToggle();
    }
  };

  return (
    <div className={cn("flex items-center", className)}>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-disabled={disabled}
        disabled={disabled}
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        data-testid={testId || 'toggle'}
        className={cn(
          "relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out",
          "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-dark-800",
          checked ? "bg-blue-500" : "bg-dark-600",
          disabled && "opacity-50 cursor-not-allowed",
          !disabled && "cursor-pointer"
        )}
      >
        <span
          className={cn(
            "inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ease-in-out",
            checked ? "translate-x-6" : "translate-x-1"
          )}
        />
      </button>
      {label && (
        <label
          htmlFor={id}
          className={cn(
            "ml-3 text-sm font-medium text-gray-300",
            disabled && "opacity-50",
            !disabled && "cursor-pointer"
          )}
          onClick={handleToggle}
        >
          {label}
        </label>
      )}
    </div>
  );
};

export default Toggle;

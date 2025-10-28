/**
 * useHotkeys Hook - Keyboard shortcut management
 * 
 * Purpose:
 * - Register keyboard shortcuts
 * - Handle keyboard events
 * - Provide cross-platform support
 * - Clean up on unmount
 * 
 * WCAG 2.1 AA Compliant:
 * - Keyboard navigation support
 * - No interference with screen readers
 * 
 * Performance:
 * - Efficient event delegation
 * - Automatic cleanup
 */

import { useEffect, useRef } from 'react';

// ============================================================================
// TYPES
// ============================================================================

export type HotkeyCallback = (event: KeyboardEvent) => void;

export interface HotkeyOptions {
  /**
   * Enable/disable the hotkey
   * @default true
   */
  enabled?: boolean;
  
  /**
   * Prevent default behavior
   * @default true
   */
  preventDefault?: boolean;
  
  /**
   * Stop event propagation
   * @default false
   */
  stopPropagation?: boolean;
  
  /**
   * Enable in input fields
   * @default false
   */
  enableOnFormTags?: boolean;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Parse hotkey string (e.g., "ctrl+d", "shift+enter")
 */
const parseHotkey = (hotkey: string): {
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
  meta: boolean;
  key: string;
} => {
  const parts = hotkey.toLowerCase().split('+');
  const key = parts[parts.length - 1];
  
  return {
    ctrl: parts.includes('ctrl') || parts.includes('control'),
    shift: parts.includes('shift'),
    alt: parts.includes('alt'),
    meta: parts.includes('meta') || parts.includes('cmd'),
    key
  };
};

/**
 * Check if hotkey matches event
 */
const matchesHotkey = (
  event: KeyboardEvent,
  hotkey: string
): boolean => {
  const parsed = parseHotkey(hotkey);
  
  return (
    event.key.toLowerCase() === parsed.key &&
    event.ctrlKey === parsed.ctrl &&
    event.shiftKey === parsed.shift &&
    event.altKey === parsed.alt &&
    event.metaKey === parsed.meta
  );
};

/**
 * Check if event target is a form element
 */
const isFormElement = (element: EventTarget | null): boolean => {
  if (!element || !(element instanceof HTMLElement)) {
    return false;
  }
  
  const tagName = element.tagName.toLowerCase();
  return (
    tagName === 'input' ||
    tagName === 'textarea' ||
    tagName === 'select' ||
    element.isContentEditable
  );
};

// ============================================================================
// HOOK
// ============================================================================

/**
 * Register a keyboard shortcut
 * 
 * @example
 * useHotkeys('ctrl+s', () => {
 *   console.log('Save shortcut pressed');
 * });
 * 
 * @example
 * useHotkeys('esc', handleClose, { preventDefault: false });
 */
export const useHotkeys = (
  hotkey: string,
  callback: HotkeyCallback,
  description?: string,
  options: HotkeyOptions = {}
): void => {
  const {
    enabled = true,
    preventDefault = true,
    stopPropagation = false,
    enableOnFormTags = false
  } = options;

  // Store callback in ref to avoid re-registering on every render
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      // Skip if hotkey doesn't match
      if (!matchesHotkey(event, hotkey)) {
        return;
      }

      // Skip if in form element (unless enabled)
      if (!enableOnFormTags && isFormElement(event.target)) {
        return;
      }

      // Prevent default behavior
      if (preventDefault) {
        event.preventDefault();
      }

      // Stop propagation
      if (stopPropagation) {
        event.stopPropagation();
      }

      // Call the callback
      callbackRef.current(event);
    };

    // Register event listener
    window.addEventListener('keydown', handleKeyDown);

    // Log in development
    if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
      console.log(`[Hotkey] Registered: ${hotkey}${description ? ` - ${description}` : ''}`);
    }

    // Cleanup
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [hotkey, enabled, preventDefault, stopPropagation, enableOnFormTags, description]);
};

/**
 * Register multiple hotkeys at once
 * 
 * @example
 * useHotkeysMap({
 *   'ctrl+s': handleSave,
 *   'ctrl+q': handleQuit,
 *   'esc': handleEscape
 * });
 */
export const useHotkeysMap = (
  hotkeys: Record<string, HotkeyCallback>,
  options?: HotkeyOptions
): void => {
  const entries = Object.entries(hotkeys);
  
  entries.forEach(([hotkey, callback]) => {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    useHotkeys(hotkey, callback, undefined, options);
  });
};

export default useHotkeys;

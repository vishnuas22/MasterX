// ===============================
// 🎨 CLASS NAME UTILITY
// ===============================

/**
 * Utility function to concatenate class names
 * Similar to clsx but lightweight for our needs
 */
export function cn(...classes) {
  return classes
    .filter(Boolean)
    .join(' ')
    .trim();
}

export default cn;
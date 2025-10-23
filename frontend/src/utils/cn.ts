/**
 * Utility function for merging Tailwind CSS classes
 * Combines clsx and tailwind-merge for optimal class handling
 */

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default cn;

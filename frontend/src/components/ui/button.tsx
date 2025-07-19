/**
 * Button Component for MasterX Quantum Intelligence Platform
 */

'use client'

import { forwardRef } from 'react'
import { cn } from '@/lib/utils'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost' | 'destructive'
  size?: 'default' | 'sm' | 'lg'
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    return (
      <button
        className={cn(
          'inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-purple-500/50 disabled:opacity-50 disabled:cursor-not-allowed',
          {
            'bg-purple-600 text-white hover:bg-purple-700 interactive-button': variant === 'default',
            'border border-purple-500/50 text-purple-300 hover:bg-purple-500/20': variant === 'outline',
            'text-purple-300 hover:bg-purple-500/20': variant === 'ghost',
            'bg-red-600 text-white hover:bg-red-700': variant === 'destructive',
          },
          {
            'h-10 px-4 py-2': size === 'default',
            'h-8 px-3 py-1 text-sm': size === 'sm',
            'h-12 px-6 py-3 text-lg': size === 'lg',
          },
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)

Button.displayName = 'Button'

export { Button }

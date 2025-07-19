/**
 * Alert Components for MasterX Quantum Intelligence Platform
 */

'use client'

import { forwardRef } from 'react'
import { cn } from '@/lib/utils'

const Alert = forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      role="alert"
      className={cn(
        'relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground',
        'border-red-500/20 bg-red-500/10 text-red-400',
        className
      )}
      {...props}
    />
  )
)
Alert.displayName = 'Alert'

const AlertDescription = forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('text-sm [&_p]:leading-relaxed', className)}
      {...props}
    />
  )
)
AlertDescription.displayName = 'AlertDescription'

export { Alert, AlertDescription }

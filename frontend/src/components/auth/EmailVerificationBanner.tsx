/**
 * Email Verification Banner Component
 * 
 * Displays a prominent banner for unverified users prompting them to verify their email.
 * Shows at the top of the application when user is logged in but email is not verified.
 * 
 * Features:
 * - Clear call-to-action to verify email
 * - Resend verification email functionality
 * - Loading states during resend
 * - Success/error toast notifications
 * - Dismissible (temporarily)
 * - Responsive design
 * - Accessibility compliant
 * 
 * Usage:
 * ```tsx
 * <EmailVerificationBanner 
 *   userEmail="user@example.com"
 *   isVerified={false}
 * />
 * ```
 * 
 * @module components/auth/EmailVerificationBanner
 */

import React, { useState } from 'react';
import { Mail, X, RefreshCw, Check, AlertCircle } from 'lucide-react';
import { authAPI } from '../../services/api/auth.api';

interface EmailVerificationBannerProps {
  /** User's email address */
  userEmail: string;
  /** Whether the email is already verified */
  isVerified: boolean;
  /** Optional callback when banner is dismissed */
  onDismiss?: () => void;
}

/**
 * Email Verification Banner
 * 
 * Shows a banner at the top of the app for unverified users with ability to
 * resend verification email.
 */
export const EmailVerificationBanner: React.FC<EmailVerificationBannerProps> = ({
  userEmail,
  isVerified,
  onDismiss
}) => {
  const [isDismissed, setIsDismissed] = useState(false);
  const [isResending, setIsResending] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [showError, setShowError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Don't show banner if verified or dismissed
  if (isVerified || isDismissed) {
    return null;
  }

  /**
   * Handle dismiss button click
   * Hides the banner temporarily (will show again on page reload)
   */
  const handleDismiss = () => {
    setIsDismissed(true);
    onDismiss?.();
  };

  /**
   * Handle resend verification email
   * Calls API to resend verification email and shows success/error feedback
   */
  const handleResendEmail = async () => {
    if (isResending) return;

    setIsResending(true);
    setShowSuccess(false);
    setShowError(false);
    setErrorMessage('');

    try {
      const response = await authAPI.resendVerification();
      
      // Show success message
      setShowSuccess(true);
      
      // Hide success message after 5 seconds
      setTimeout(() => {
        setShowSuccess(false);
      }, 5000);
      
      console.log('✅ Verification email resent:', response.message);
    } catch (error: any) {
      console.error('❌ Failed to resend verification email:', error);
      
      // Extract error message
      const message = error.response?.data?.detail || 
                     error.message || 
                     'Failed to send verification email. Please try again.';
      
      setErrorMessage(message);
      setShowError(true);
      
      // Hide error message after 5 seconds
      setTimeout(() => {
        setShowError(false);
        setErrorMessage('');
      }, 5000);
    } finally {
      setIsResending(false);
    }
  };

  return (
    <div 
      className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-md"
      role="alert"
      aria-live="polite"
      data-testid="email-verification-banner"
    >
      <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between flex-wrap gap-3">
          {/* Left: Icon + Message */}
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <Mail 
              className="w-5 h-5 flex-shrink-0" 
              aria-hidden="true"
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium">
                Verify your email address
              </p>
              <p className="text-xs opacity-90 mt-0.5">
                We sent a verification link to <span className="font-semibold">{userEmail}</span>
              </p>
            </div>
          </div>

          {/* Center: Success/Error Messages */}
          {showSuccess && (
            <div 
              className="flex items-center gap-2 bg-white/20 rounded-lg px-3 py-1.5"
              role="status"
              aria-live="polite"
            >
              <Check className="w-4 h-4" aria-hidden="true" />
              <span className="text-sm font-medium">Email sent! Check your inbox.</span>
            </div>
          )}

          {showError && (
            <div 
              className="flex items-center gap-2 bg-red-500/30 rounded-lg px-3 py-1.5"
              role="alert"
              aria-live="assertive"
            >
              <AlertCircle className="w-4 h-4" aria-hidden="true" />
              <span className="text-sm font-medium">{errorMessage}</span>
            </div>
          )}

          {/* Right: Actions */}
          <div className="flex items-center gap-2">
            {/* Resend Button */}
            <button
              onClick={handleResendEmail}
              disabled={isResending}
              className="flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Resend verification email"
              data-testid="resend-verification-button"
            >
              <RefreshCw 
                className={`w-4 h-4 ${isResending ? 'animate-spin' : ''}`}
                aria-hidden="true"
              />
              <span className="hidden sm:inline">
                {isResending ? 'Sending...' : 'Resend Email'}
              </span>
            </button>

            {/* Dismiss Button */}
            <button
              onClick={handleDismiss}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
              aria-label="Dismiss verification banner"
              data-testid="dismiss-banner-button"
            >
              <X className="w-5 h-5" aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmailVerificationBanner;

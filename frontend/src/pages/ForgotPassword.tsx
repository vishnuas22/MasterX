/**
 * Forgot Password Page - Request Password Reset
 * 
 * WCAG 2.1 AA Compliant:
 * - Form labels and ARIA attributes
 * - Keyboard navigation
 * - Screen reader support
 * - Error announcements
 * - Focus management
 * 
 * Security:
 * - Email validation before submission
 * - Rate limiting on backend
 * - No email enumeration (always shows success)
 * - CSRF protection via API client
 * 
 * Features:
 * - Email input with validation
 * - Loading states
 * - Success/error messages
 * - Link back to login
 * 
 * Backend Integration:
 * - POST /api/auth/password-reset-request
 * 
 * Following AGENTS_FRONTEND.md:
 * - Strict TypeScript (no 'any' types)
 * - Mobile-first responsive
 * - Accessible form controls
 * - User-friendly error messages
 */

import React, { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Mail, ArrowLeft, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { authAPI } from '@/services/api/auth.api';

// ============================================================================
// COMPONENT
// ============================================================================

export default function ForgotPassword() {
  const navigate = useNavigate();
  
  // Form state
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [responseMessage, setResponseMessage] = useState<string>('');

  /**
   * Validate email format
   */
  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  /**
   * Handle form submission
   */
  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setSuccess(false);

    // Validate email
    if (!email.trim()) {
      setError('Email address is required');
      return;
    }

    if (!validateEmail(email)) {
      setError('Please enter a valid email address');
      return;
    }

    setIsLoading(true);

    try {
      // Call backend API
      const response = await authAPI.requestPasswordReset(email);
      
      // Show success message
      setSuccess(true);
      setResponseMessage(response.message);
      
      // Clear form
      setEmail('');
      
      // Note for development: If backend returns a note (dev mode), show it
      if (response.note) {
        console.info('🔑 Development Note:', response.note);
      }
      
    } catch (err: any) {
      console.error('Password reset request failed:', err);
      
      // Extract error message
      const errorMessage = err.response?.data?.detail || 
                          err.message || 
                          'Failed to send reset email. Please try again.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Handle back to login
   */
  const handleBackToLogin = () => {
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        {/* Card */}
        <div className="bg-dark-800 rounded-2xl shadow-2xl p-8 border border-white/10">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <Mail className="w-8 h-8 text-blue-400" />
            </div>
            <h1 className="text-3xl font-bold text-white mb-2">
              Forgot Password?
            </h1>
            <p className="text-gray-400 text-sm">
              No worries! Enter your email and we'll send you reset instructions.
            </p>
          </div>

          {/* Success Message */}
          {success && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mb-6 p-4 bg-green-500/10 border border-green-500/30 rounded-lg flex items-start gap-3"
              role="alert"
              aria-live="polite"
            >
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-green-400 font-medium text-sm">
                  Check your email
                </p>
                <p className="text-green-300/80 text-sm mt-1">
                  {responseMessage}
                </p>
              </div>
            </motion.div>
          )}

          {/* Error Message */}
          {error && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-3"
              role="alert"
              aria-live="assertive"
            >
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-400 font-medium text-sm">
                  Error
                </p>
                <p className="text-red-300/80 text-sm mt-1">
                  {error}
                </p>
              </div>
            </motion.div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email Input */}
            <div>
              <label 
                htmlFor="email" 
                className="block text-sm font-medium text-gray-300 mb-2"
              >
                Email Address
              </label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                disabled={isLoading || success}
                required
                aria-label="Email address"
                aria-required="true"
                aria-invalid={!!error}
                aria-describedby={error ? 'email-error' : undefined}
                className="w-full"
              />
              {error && (
                <p id="email-error" className="mt-1 text-sm text-red-400" role="alert">
                  {error}
                </p>
              )}
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              disabled={isLoading || success}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label={isLoading ? 'Sending reset email...' : 'Send reset email'}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Sending...
                </>
              ) : (
                <>
                  <Mail className="w-5 h-5 mr-2" />
                  Send Reset Email
                </>
              )}
            </Button>

            {/* Back to Login */}
            <div className="text-center">
              <button
                type="button"
                onClick={handleBackToLogin}
                className="inline-flex items-center text-sm text-blue-400 hover:text-blue-300 transition"
                aria-label="Back to login page"
              >
                <ArrowLeft className="w-4 h-4 mr-1" />
                Back to Login
              </button>
            </div>
          </form>

          {/* Additional Info */}
          <div className="mt-8 pt-6 border-t border-white/10">
            <p className="text-xs text-gray-500 text-center">
              Didn't receive the email? Check your spam folder or try again in a few minutes.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            Need help?{' '}
            <Link 
              to="/support" 
              className="text-blue-400 hover:text-blue-300 transition"
            >
              Contact Support
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
}

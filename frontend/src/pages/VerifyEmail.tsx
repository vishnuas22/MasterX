/**
 * Email Verification Page
 * 
 * Handles email verification when user clicks the verification link from their email.
 * Extracts token from URL query parameters and calls the verification API.
 * 
 * Features:
 * - Automatic verification on page load
 * - Loading state with spinner
 * - Success state with celebration
 * - Error state with retry option
 * - Redirect to dashboard on success
 * - Responsive design
 * - Accessibility compliant
 * 
 * URL Pattern: /verify-email?token=abc123...
 * 
 * @module pages/VerifyEmail
 */

import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { CheckCircle, XCircle, Loader2, Mail, ArrowRight } from 'lucide-react';
import { authAPI } from '../services/api/auth.api';
import { useAuthStore } from '../store/authStore';

type VerificationState = 'loading' | 'success' | 'error' | 'already-verified';

interface VerificationResult {
  state: VerificationState;
  message: string;
  userName?: string;
  userEmail?: string;
}

/**
 * Email Verification Page Component
 * 
 * Automatically verifies email when component mounts using token from URL.
 */
export const VerifyEmail: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { checkAuth } = useAuthStore();
  
  const [result, setResult] = useState<VerificationResult>({
    state: 'loading',
    message: 'Verifying your email...'
  });

  useEffect(() => {
    const verifyEmail = async () => {
      // Get token from URL
      const token = searchParams.get('token');

      if (!token) {
        setResult({
          state: 'error',
          message: 'Verification link is invalid. No token found in URL.'
        });
        return;
      }

      try {
        // Call verification API
        const response = await authAPI.verifyEmail(token);
        
        // Check if already verified
        if (response.message.includes('already verified')) {
          setResult({
            state: 'already-verified',
            message: response.message,
            userName: response.user.name,
            userEmail: response.user.email
          });
        } else {
          // Success - email verified
          setResult({
            state: 'success',
            message: response.message,
            userName: response.user.name,
            userEmail: response.user.email
          });

          // Store tokens if provided (for immediate login)
          if (response.access_token && response.refresh_token) {
            localStorage.setItem('jwt_token', response.access_token);
            localStorage.setItem('refresh_token', response.refresh_token);
            
            // Refresh auth state to update user's verified status
            await checkAuth();
          }
          
          // Clear pending verification email
          localStorage.removeItem('pending_verification_email');

          // Redirect to dashboard after 3 seconds
          setTimeout(() => {
            navigate('/app', { replace: true });
          }, 3000);
        }
      } catch (error: any) {
        console.error('❌ Email verification failed:', error);
        
        // Extract error message
        const message = error.response?.data?.detail || 
                       error.message || 
                       'Failed to verify email. The link may be invalid or expired.';
        
        setResult({
          state: 'error',
          message
        });
      }
    };

    verifyEmail();
  }, [searchParams, navigate, checkAuth]);

  /**
   * Handle retry - redirect to login page
   */
  const handleRetry = () => {
    navigate('/login');
  };

  /**
   * Handle continue - redirect to dashboard
   */
  const handleContinue = () => {
    navigate('/dashboard', { replace: true });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        {/* Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
          {/* Icon */}
          <div className="flex justify-center mb-6">
            {result.state === 'loading' && (
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                <Loader2 
                  className="w-8 h-8 text-blue-600 animate-spin" 
                  data-testid="loading-spinner"
                  aria-label="Loading"
                />
              </div>
            )}
            {result.state === 'success' && (
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center animate-bounce">
                <CheckCircle 
                  className="w-8 h-8 text-green-600" 
                  data-testid="success-icon"
                  aria-label="Success"
                />
              </div>
            )}
            {result.state === 'already-verified' && (
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                <CheckCircle 
                  className="w-8 h-8 text-blue-600" 
                  data-testid="already-verified-icon"
                  aria-label="Already verified"
                />
              </div>
            )}
            {result.state === 'error' && (
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
                <XCircle 
                  className="w-8 h-8 text-red-600" 
                  data-testid="error-icon"
                  aria-label="Error"
                />
              </div>
            )}
          </div>

          {/* Title */}
          <h1 className="text-2xl font-bold text-gray-900 mb-3">
            {result.state === 'loading' && 'Verifying Email'}
            {result.state === 'success' && 'Email Verified! 🎉'}
            {result.state === 'already-verified' && 'Already Verified'}
            {result.state === 'error' && 'Verification Failed'}
          </h1>

          {/* Message */}
          <p className="text-gray-600 mb-6">
            {result.message}
          </p>

          {/* User Info (for success/already-verified) */}
          {(result.state === 'success' || result.state === 'already-verified') && result.userName && (
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-center gap-2 text-sm text-gray-700">
                <Mail className="w-4 h-4" aria-hidden="true" />
                <span>
                  <span className="font-semibold">{result.userName}</span>
                  <span className="mx-1">•</span>
                  <span>{result.userEmail}</span>
                </span>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="space-y-3">
            {result.state === 'loading' && (
              <div className="text-sm text-gray-500">
                Please wait while we verify your email address...
              </div>
            )}

            {result.state === 'success' && (
              <>
                <div className="text-sm text-gray-500 mb-4">
                  Redirecting to dashboard in 3 seconds...
                </div>
                <button
                  onClick={handleContinue}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
                  data-testid="continue-button"
                >
                  <span>Continue to Dashboard</span>
                  <ArrowRight className="w-4 h-4" aria-hidden="true" />
                </button>
              </>
            )}

            {result.state === 'already-verified' && (
              <button
                onClick={handleContinue}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
                data-testid="continue-button"
              >
                <span>Go to Dashboard</span>
                <ArrowRight className="w-4 h-4" aria-hidden="true" />
              </button>
            )}

            {result.state === 'error' && (
              <>
                <button
                  onClick={handleRetry}
                  className="w-full px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
                  data-testid="retry-button"
                >
                  Go to Login
                </button>
                <p className="text-sm text-gray-500">
                  Need a new verification link?{' '}
                  <button
                    onClick={() => navigate('/login')}
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    Login
                  </button>{' '}
                  and request a new one.
                </p>
              </>
            )}
          </div>
        </div>

        {/* Help Text */}
        <div className="mt-6 text-center text-sm text-gray-600">
          <p>
            Need help?{' '}
            <a 
              href="mailto:support@masterx.ai" 
              className="text-blue-600 hover:text-blue-700 font-medium"
            >
              Contact Support
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default VerifyEmail;

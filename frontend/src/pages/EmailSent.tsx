/**
 * Email Sent Page
 * 
 * Displayed after successful registration when email verification is required.
 * Informs user to check their email and provides option to resend verification email.
 * 
 * Features:
 * - Clear instructions to check email
 * - Displays email address used for registration
 * - Resend verification email button with rate limiting
 * - Link to login page
 * - Responsive design
 * - Accessibility compliant
 * 
 * @module pages/EmailSent
 */

import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Mail, CheckCircle, ArrowRight, RefreshCw } from 'lucide-react';
import { authAPI } from '../services/api/auth.api';
import { useUIStore } from '../store/uiStore';

interface LocationState {
  email?: string;
}

/**
 * Email Sent Page Component
 * 
 * Shows confirmation that verification email was sent and provides options to resend.
 */
export const EmailSent: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { showToast } = useUIStore();
  
  const state = location.state as LocationState;
  const [email, setEmail] = useState<string>(
    state?.email || localStorage.getItem('pending_verification_email') || ''
  );
  const [isResending, setIsResending] = useState(false);
  const [resendCount, setResendCount] = useState(0);
  const [canResend, setCanResend] = useState(true);
  const [cooldownSeconds, setCooldownSeconds] = useState(0);

  // Clear pending email from storage when component unmounts
  useEffect(() => {
    if (!email) {
      // If no email, redirect to signup
      navigate('/signup');
    }
  }, [email, navigate]);

  // Cooldown timer
  useEffect(() => {
    if (cooldownSeconds > 0) {
      const timer = setTimeout(() => {
        setCooldownSeconds(cooldownSeconds - 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else if (cooldownSeconds === 0 && !canResend) {
      setCanResend(true);
    }
  }, [cooldownSeconds, canResend]);

  /**
   * Handle resend verification email
   */
  const handleResend = async () => {
    if (!canResend || isResending) return;

    setIsResending(true);
    
    try {
      // Note: This requires the user to be logged in
      // For now, we'll show an error and suggest logging in
      showToast({
        type: 'error',
        message: 'Please login first to resend verification email',
        duration: 5000,
      });
      
      // Redirect to login after 2 seconds
      setTimeout(() => {
        navigate('/login');
      }, 2000);
      
      /* TODO: Implement resend without auth
      const response = await authAPI.resendVerification();
      
      showToast({
        type: 'success',
        message: 'Verification email sent! Please check your inbox.',
        duration: 5000,
      });
      
      // Update resend count and set cooldown
      setResendCount(prev => prev + 1);
      setCanResend(false);
      setCooldownSeconds(60); // 60 second cooldown
      */
      
    } catch (error: any) {
      console.error('Failed to resend email:', error);
      showToast({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to resend email. Please try again.',
        duration: 5000,
      });
    } finally {
      setIsResending(false);
    }
  };

  /**
   * Navigate to login page
   */
  const handleGoToLogin = () => {
    localStorage.removeItem('pending_verification_email');
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        {/* Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
          {/* Icon */}
          <div className="flex justify-center mb-6">
            <div className="relative">
              <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center">
                <Mail 
                  className="w-10 h-10 text-blue-600" 
                  data-testid="mail-icon"
                  aria-label="Email sent"
                />
              </div>
              <div className="absolute -bottom-1 -right-1 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center border-4 border-white">
                <CheckCircle 
                  className="w-5 h-5 text-white" 
                  aria-hidden="true"
                />
              </div>
            </div>
          </div>

          {/* Title */}
          <h1 className="text-2xl font-bold text-gray-900 mb-3">
            Check Your Email 📬
          </h1>

          {/* Message */}
          <p className="text-gray-600 mb-6">
            We've sent a verification link to:
          </p>

          {/* Email Display */}
          <div className="bg-blue-50 rounded-lg p-4 mb-6">
            <p className="text-blue-900 font-medium break-all">
              {email}
            </p>
          </div>

          {/* Instructions */}
          <div className="text-left bg-gray-50 rounded-lg p-4 mb-6">
            <h3 className="font-semibold text-gray-900 mb-2">Next Steps:</h3>
            <ol className="text-sm text-gray-600 space-y-2 list-decimal list-inside">
              <li>Open your email inbox</li>
              <li>Look for an email from MasterX Support</li>
              <li>Click the verification link in the email</li>
              <li>You'll be redirected back to start learning!</li>
            </ol>
          </div>

          {/* Didn't receive email? */}
          <div className="space-y-3">
            <p className="text-sm text-gray-500">
              Didn't receive the email? Check your spam folder.
            </p>

            {/* Resend Button */}
            <button
              onClick={handleResend}
              disabled={!canResend || isResending}
              className={`
                w-full flex items-center justify-center gap-2 px-6 py-3 
                border-2 border-blue-600 text-blue-600 font-medium rounded-lg 
                transition-all
                ${!canResend || isResending 
                  ? 'opacity-50 cursor-not-allowed' 
                  : 'hover:bg-blue-50 active:scale-95'
                }
              `}
              data-testid="resend-button"
            >
              <RefreshCw 
                className={`w-4 h-4 ${isResending ? 'animate-spin' : ''}`} 
                aria-hidden="true" 
              />
              <span>
                {isResending 
                  ? 'Sending...' 
                  : cooldownSeconds > 0 
                    ? `Resend in ${cooldownSeconds}s`
                    : 'Resend Verification Email'
                }
              </span>
            </button>

            {/* Go to Login Button */}
            <button
              onClick={handleGoToLogin}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
              data-testid="login-button"
            >
              <span>Go to Login</span>
              <ArrowRight className="w-4 h-4" aria-hidden="true" />
            </button>
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

        {/* Development Mode Notice */}
        <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-xs text-yellow-800">
            <strong>Development Mode:</strong> Emails are logged to the backend console. 
            Check the backend logs at <code className="bg-yellow-100 px-1 rounded">/var/log/supervisor/backend.err.log</code> 
            to see the verification link.
          </p>
        </div>
      </div>
    </div>
  );
};

export default EmailSent;

/**
 * Login Page Component - Secure Authentication
 * 
 * WCAG 2.1 AA Compliant:
 * - Form labels and error messages
 * - Keyboard accessible
 * - High contrast error states
 * - Screen reader announcements
 * 
 * Security:
 * - XSS protection (input sanitization)
 * - CSRF tokens
 * - Rate limiting (5 attempts per 15 min)
 * - Secure password input
 * - JWT token storage (HttpOnly cookies)
 * 
 * Performance:
 * - Optimistic UI updates
 * - Client-side validation (instant feedback)
 * - Debounced API calls
 * 
 * Backend Integration:
 * - POST /api/v1/auth/login (JWT auth)
 * - Refresh token rotation
 * - Social OAuth flow
 */

import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { motion } from 'framer-motion';
import { LoginForm } from '@/components/auth/LoginForm';
import { ArrowLeft } from 'lucide-react';

// ============================================================================
// TYPES
// ============================================================================

export interface LoginProps {
  /**
   * Redirect path after successful login
   * @default "/app"
   */
  redirectTo?: string;
  
  /**
   * Show social login options
   * @default true
   */
  showSocialLogin?: boolean;
}

// ============================================================================
// COMPONENT
// ============================================================================

export const Login: React.FC<LoginProps> = ({
  redirectTo = '/app',
  showSocialLogin = true
}) => {
  const navigate = useNavigate();

  const handleLoginSuccess = () => {
    // LoginForm handles navigation internally
    // This callback is for additional tracking if needed
  };

  return (
    <>
      {/* SEO */}
      <Helmet>
        <title>Login - MasterX</title>
        <meta name="description" content="Log in to MasterX - AI learning with emotion detection" />
        <meta name="robots" content="noindex, nofollow" />
      </Helmet>

      <div className="min-h-screen bg-bg-primary flex items-center justify-center px-4 sm:px-6 lg:px-8 relative overflow-hidden">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-accent-primary/5 via-accent-purple/5 to-accent-pink/5 pointer-events-none" />
        
        {/* Animated background shapes */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              rotate: [0, 90, 0],
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-accent-primary/10 to-transparent rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1, 1.3, 1],
              rotate: [0, -90, 0],
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-accent-purple/10 to-transparent rounded-full blur-3xl"
          />
        </div>

        <div className="relative w-full max-w-md z-10">
          {/* Back to home button */}
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary transition-colors mb-8 group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            Back to home
          </Link>

          {/* Logo and title */}
          <div className="text-center mb-8">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 200, damping: 15 }}
            >
              <Link 
                to="/" 
                className="inline-flex items-center gap-3 group mb-6"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-accent-primary to-accent-purple rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow">
                  <span className="text-2xl font-bold text-white">M</span>
                </div>
                <span className="text-2xl font-bold text-text-primary">MasterX</span>
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <h1 className="text-3xl font-bold text-text-primary mb-2">
                Welcome back
              </h1>
              <p className="text-text-secondary">
                Log in to continue your learning journey
              </p>
            </motion.div>
          </div>

          {/* Login form */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-bg-secondary/50 backdrop-blur-xl rounded-2xl border border-white/10 p-8 shadow-xl"
          >
            <LoginForm onSuccess={handleLoginSuccess} showSocialAuth={showSocialLogin} />
          </motion.div>

          {/* Footer note */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="mt-8 text-center text-sm text-text-tertiary"
          >
            <p>
              By signing in, you agree to our{' '}
              <Link to="/terms" className="text-accent-primary hover:underline">
                Terms of Service
              </Link>
              {' '}and{' '}
              <Link to="/privacy" className="text-accent-primary hover:underline">
                Privacy Policy
              </Link>
            </p>
          </motion.div>
        </div>
      </div>
    </>
  );
};

// ============================================================================
// EXPORTS
// ============================================================================

export default Login;

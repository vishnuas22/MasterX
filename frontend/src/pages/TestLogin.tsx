/**
 * Test Page for LoginForm Component
 * 
 * Purpose: Test and verify LoginForm functionality in isolation
 */

import React from 'react';
import { LoginForm } from '@/components/auth/LoginForm';
import { ToastContainer } from '@/components/ui/Toast';

export default function TestLogin() {
  return (
    <div className="min-h-screen bg-bg-primary flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-text-primary mb-2">
            LoginForm Test Page
          </h1>
          <p className="text-text-secondary">
            Testing authentication functionality
          </p>
        </div>

        {/* LoginForm Component */}
        <div className="bg-bg-secondary p-8 rounded-2xl shadow-xl">
          <LoginForm
            onSuccess={() => {
              console.log('✅ Login successful callback triggered');
            }}
            showSocialAuth={true}
          />
        </div>

        {/* Debug Info */}
        <div className="bg-bg-secondary p-4 rounded-lg text-sm">
          <h3 className="font-semibold text-text-primary mb-2">Test Credentials:</h3>
          <p className="text-text-secondary mb-1">
            Email: <code className="bg-bg-tertiary px-2 py-0.5 rounded">test@example.com</code>
          </p>
          <p className="text-text-secondary">
            Password: <code className="bg-bg-tertiary px-2 py-0.5 rounded">password123</code>
          </p>
        </div>

        {/* Features to Test */}
        <div className="bg-bg-secondary p-4 rounded-lg text-sm space-y-2">
          <h3 className="font-semibold text-text-primary mb-2">✅ Test Checklist:</h3>
          <ul className="text-text-secondary space-y-1 list-disc list-inside">
            <li>Email validation (invalid format)</li>
            <li>Password validation (min 8 characters)</li>
            <li>Show/hide password toggle</li>
            <li>Remember me checkbox</li>
            <li>Form submission with valid credentials</li>
            <li>Error handling (invalid credentials)</li>
            <li>Loading state during API call</li>
            <li>Toast notifications</li>
            <li>Keyboard navigation (Tab, Enter)</li>
            <li>Forgot password link</li>
            <li>Sign up link navigation</li>
          </ul>
        </div>
      </div>

      {/* Toast Container for notifications */}
      <ToastContainer />
    </div>
  );
}

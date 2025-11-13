/**
 * Signup Form Component
 * 
 * Security (OWASP Compliant):
 * - Password strength meter
 * - Email format validation
 * - Terms of service acceptance
 * - CSRF protection (from backend)
 * - Rate limiting aware
 * 
 * WCAG 2.1 AA Compliant:
 * - Form labels and hints
 * - Error announcements
 * - Keyboard accessible
 * - Password requirements visible
 * 
 * Backend Integration:
 * - POST /api/auth/register
 * - Returns JWT token + user data
 * - Email verification flow (optional)
 */

import React from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Mail, Lock, User, Eye, EyeOff, AlertCircle, CheckCircle2, 
  Loader2, Shield 
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { cn } from '@/utils/cn';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { toast } from '@/components/ui/Toast';

// ============================================================================
// TYPES
// ============================================================================

export interface SignupFormProps {
  onSuccess?: () => void;
  showSocialAuth?: boolean;
  className?: string;
}

interface FormData {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
  acceptTerms: boolean;
}

interface FormErrors {
  name?: string;
  email?: string;
  password?: string;
  confirmPassword?: string;
  acceptTerms?: string;
  general?: string;
}

interface PasswordStrength {
  score: number; // 0-4
  label: string;
  color: string;
  feedback: string[];
}

// ============================================================================
// PASSWORD STRENGTH CHECKER
// ============================================================================

const checkPasswordStrength = (password: string): PasswordStrength => {
  let score = 0;
  const feedback: string[] = [];

  if (password.length >= 8) score++;
  else feedback.push('At least 8 characters');

  if (password.length >= 12) score++;
  else if (password.length >= 8) feedback.push('12+ characters recommended');

  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
  else feedback.push('Mix of uppercase and lowercase');

  if (/\d/.test(password)) score++;
  else feedback.push('Include numbers');

  if (/[^a-zA-Z0-9]/.test(password)) score++;
  else feedback.push('Include special characters');

  const labels = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'];
  const colors = [
    'bg-accent-error',
    'bg-accent-warning',
    'bg-accent-warning',
    'bg-accent-success',
    'bg-accent-success',
  ];

  return {
    score,
    label: labels[score],
    color: colors[score],
    feedback: feedback.length > 0 ? feedback : ['Password is strong'],
  };
};

// ============================================================================
// VALIDATION
// ============================================================================

const validateName = (name: string): string | undefined => {
  if (!name) return 'Name is required';
  if (name.length < 2) return 'Name must be at least 2 characters';
  if (name.length > 50) return 'Name must be less than 50 characters';
  return undefined;
};

const validateEmail = (email: string): string | undefined => {
  if (!email) return 'Email is required';
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) return 'Invalid email format';
  return undefined;
};

const validatePassword = (password: string): string | undefined => {
  if (!password) return 'Password is required';
  if (password.length < 8) return 'Password must be at least 8 characters';
  
  const strength = checkPasswordStrength(password);
  if (strength.score < 2) return 'Password is too weak';
  
  return undefined;
};

// ============================================================================
// PASSWORD STRENGTH METER
// ============================================================================

const PasswordStrengthMeter = React.memo<{ password: string }>(({ password }) => {
  if (!password) return null;

  const strength = checkPasswordStrength(password);

  return (
    <div className="mt-2 space-y-2">
      {/* Strength bar */}
      <div className="flex gap-1">
        {[0, 1, 2, 3, 4].map((index) => (
          <div
            key={index}
            className={cn(
              'h-1 flex-1 rounded-full transition-colors',
              index <= strength.score ? strength.color : 'bg-bg-tertiary'
            )}
          />
        ))}
      </div>

      {/* Strength label */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-text-tertiary">
          Password strength: <span className="font-medium">{strength.label}</span>
        </span>
      </div>

      {/* Feedback */}
      {strength.feedback.length > 0 && strength.score < 4 && (
        <ul className="text-xs text-text-tertiary space-y-1">
          {strength.feedback.map((item, index) => (
            <li key={index} className="flex items-center gap-2">
              <div className="w-1 h-1 rounded-full bg-text-tertiary" />
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
});

PasswordStrengthMeter.displayName = 'PasswordStrengthMeter';

// ============================================================================
// MAIN SIGNUP FORM COMPONENT
// ============================================================================

export const SignupForm = React.memo<SignupFormProps>(({
  onSuccess,
  showSocialAuth = true,
  className,
}) => {
  const navigate = useNavigate();
  const { signup, isLoading } = useAuthStore();
  
  const [formData, setFormData] = React.useState<FormData>({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    acceptTerms: false,
  });
  
  const [errors, setErrors] = React.useState<FormErrors>({});
  const [showPassword, setShowPassword] = React.useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = React.useState(false);

  // Handle input change
  const handleChange = (field: keyof FormData, value: string | boolean) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    
    // Clear error
    if (errors[field as keyof FormErrors]) {
      setErrors((prev) => {
        const newErrors = { ...prev };
        delete newErrors[field as keyof FormErrors];
        return newErrors;
      });
    }
  };

  // Validate on blur
  const handleBlur = (field: keyof FormData) => {
    const newErrors = { ...errors };
    
    switch (field) {
      case 'name':
        const nameError = validateName(formData.name);
        if (nameError) newErrors.name = nameError;
        else delete newErrors.name;
        break;
      
      case 'email':
        const emailError = validateEmail(formData.email);
        if (emailError) newErrors.email = emailError;
        else delete newErrors.email;
        break;
      
      case 'password':
        const passwordError = validatePassword(formData.password);
        if (passwordError) newErrors.password = passwordError;
        else delete newErrors.password;
        
        // Also check confirm password if it's filled
        if (formData.confirmPassword && formData.password !== formData.confirmPassword) {
          newErrors.confirmPassword = 'Passwords do not match';
        }
        break;
      
      case 'confirmPassword':
        if (formData.password !== formData.confirmPassword) {
          newErrors.confirmPassword = 'Passwords do not match';
        } else {
          delete newErrors.confirmPassword;
        }
        break;
    }
    
    setErrors(newErrors);
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate all fields
    const nameError = validateName(formData.name);
    const emailError = validateEmail(formData.email);
    const passwordError = validatePassword(formData.password);
    const confirmError = formData.password !== formData.confirmPassword 
      ? 'Passwords do not match' 
      : undefined;
    const termsError = !formData.acceptTerms 
      ? 'You must accept the terms of service' 
      : undefined;
    
    if (nameError || emailError || passwordError || confirmError || termsError) {
      setErrors({
        name: nameError,
        email: emailError,
        password: passwordError,
        confirmPassword: confirmError,
        acceptTerms: termsError,
      });
      return;
    }

    try {
      await signup({
        name: formData.name,
        email: formData.email,
        password: formData.password,
      });
      
      toast.success('Account created successfully!', {
        description: 'Welcome to MasterX',
      });
      
      if (onSuccess) {
        onSuccess();
      } else {
        navigate('/app');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message;
      
      if (errorMessage.includes('already exists')) {
        setErrors({ email: 'This email is already registered' });
      } else if (errorMessage.includes('rate limit')) {
        setErrors({ general: 'Too many requests. Please wait and try again.' });
      } else {
        setErrors({ general: 'Failed to create account. Please try again.' });
      }
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn('w-full max-w-md', className)}
    >
      <form onSubmit={handleSubmit} className="space-y-5" noValidate>
        {/* General error */}
        {errors.general && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="flex items-start gap-2 p-3 bg-accent-error/10 border border-accent-error/30 rounded-lg"
            role="alert"
          >
            <AlertCircle className="w-5 h-5 text-accent-error flex-shrink-0 mt-0.5" />
            <p className="text-sm text-accent-error">{errors.general}</p>
          </motion.div>
        )}

        {/* Name field */}
        <div>
          <label htmlFor="signup-name" className="block text-sm font-medium text-text-primary mb-2">
            Full Name
          </label>
          <Input
            id="signup-name"
            type="text"
            autoComplete="name"
            placeholder="John Doe"
            value={formData.name}
            onChange={(e) => handleChange('name', e.target.value)}
            onBlur={() => handleBlur('name')}
            error={errors.name}
            disabled={isLoading}
            leftIcon={<User className="w-5 h-5" />}
          />
          {errors.name && (
            <p className="mt-1 text-sm text-accent-error">{errors.name}</p>
          )}
        </div>

        {/* Email field */}
        <div>
          <label htmlFor="signup-email" className="block text-sm font-medium text-text-primary mb-2">
            Email
          </label>
          <Input
            id="signup-email"
            type="email"
            autoComplete="email"
            placeholder="you@example.com"
            value={formData.email}
            onChange={(e) => handleChange('email', e.target.value)}
            onBlur={() => handleBlur('email')}
            error={errors.email}
            disabled={isLoading}
            leftIcon={<Mail className="w-5 h-5" />}
          />
          {errors.email && (
            <p className="mt-1 text-sm text-accent-error">{errors.email}</p>
          )}
        </div>

        {/* Password field */}
        <div>
          <label htmlFor="signup-password" className="block text-sm font-medium text-text-primary mb-2">
            Password
          </label>
          <Input
            id="signup-password"
            type={showPassword ? 'text' : 'password'}
            autoComplete="new-password"
            placeholder="••••••••"
            value={formData.password}
            onChange={(e) => handleChange('password', e.target.value)}
            onBlur={() => handleBlur('password')}
            error={errors.password}
            disabled={isLoading}
            leftIcon={<Lock className="w-5 h-5" />}
            rightIcon={
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="text-text-tertiary hover:text-text-primary transition-colors"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            }
          />
          {errors.password && (
            <p className="mt-1 text-sm text-accent-error">{errors.password}</p>
          )}
          <PasswordStrengthMeter password={formData.password} />
        </div>

        {/* Confirm password field */}
        <div>
          <label htmlFor="signup-confirm-password" className="block text-sm font-medium text-text-primary mb-2">
            Confirm Password
          </label>
          <Input
            id="signup-confirm-password"
            type={showConfirmPassword ? 'text' : 'password'}
            autoComplete="new-password"
            placeholder="••••••••"
            value={formData.confirmPassword}
            onChange={(e) => handleChange('confirmPassword', e.target.value)}
            onBlur={() => handleBlur('confirmPassword')}
            error={errors.confirmPassword}
            disabled={isLoading}
            leftIcon={<Lock className="w-5 h-5" />}
            rightIcon={
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="text-text-tertiary hover:text-text-primary transition-colors"
                aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
              >
                {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            }
          />
          {errors.confirmPassword && (
            <p className="mt-1 text-sm text-accent-error">{errors.confirmPassword}</p>
          )}
          {!errors.confirmPassword && formData.confirmPassword && formData.password === formData.confirmPassword && (
            <div className="mt-1 flex items-center gap-1 text-sm text-accent-success">
              <CheckCircle2 className="w-4 h-4" />
              <span>Passwords match</span>
            </div>
          )}
        </div>

        {/* Terms acceptance */}
        <div>
          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={formData.acceptTerms}
              onChange={(e) => handleChange('acceptTerms', e.target.checked)}
              className="mt-1 w-4 h-4 text-accent-primary bg-bg-tertiary border-white/20 rounded focus:ring-2 focus:ring-accent-primary"
            />
            <span className="text-sm text-text-secondary">
              I agree to the{' '}
              <Link to="/terms" target="_blank" className="text-accent-primary hover:underline">
                Terms of Service
              </Link>{' '}
              and{' '}
              <Link to="/privacy" target="_blank" className="text-accent-primary hover:underline">
                Privacy Policy
              </Link>
            </span>
          </label>
          {errors.acceptTerms && (
            <p className="mt-1 text-sm text-accent-error">{errors.acceptTerms}</p>
          )}
        </div>

        {/* Submit button */}
        <Button
          type="submit"
          variant="primary"
          size="lg"
          fullWidth
          disabled={isLoading}
          leftIcon={isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Shield className="w-5 h-5" />}
        >
          {isLoading ? 'Creating account...' : 'Create account'}
        </Button>

        {/* Divider */}
        {showSocialAuth && (
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/10" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-bg-primary text-text-tertiary">
                Or continue with
              </span>
            </div>
          </div>
        )}

        {/* Social auth placeholder */}
        {showSocialAuth && (
          <div className="text-center">
            <p className="text-sm text-text-tertiary">
              Social authentication coming soon
            </p>
          </div>
        )}

        {/* Login link */}
        <p className="text-center text-sm text-text-secondary">
          Already have an account?{' '}
          <Link to="/login" className="text-accent-primary font-medium hover:underline">
            Sign in
          </Link>
        </p>
      </form>
    </motion.div>
  );
});

SignupForm.displayName = 'SignupForm';

export default SignupForm;

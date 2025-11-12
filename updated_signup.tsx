import React, { useState } from 'react';
import { Eye, EyeOff, ArrowLeft, Check, X } from 'lucide-react';

// ============================================================================
// BORDER BEAM COMPONENT (Correct implementation - rotating beam on border)
// ============================================================================
const BorderBeam = ({
  size = 50,
  duration = 15,
  delay = 0,
  borderWidth = 7.5,
  colorFrom = "#ffaa40",
  colorTo = "#9c40ff",
  anchor = 360,
}) => {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        overflow: "hidden",
        borderRadius: "inherit",
      }}
    >
      {/* The actual rotating beam */}
      <div
        style={{
          position: "absolute",
          inset: `-${borderWidth}px`,
          borderRadius: "inherit",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 1,
            borderRadius: "inherit 10px",
            padding: `${borderWidth}px`,
            WebkitMask: "linear-gradient(white 0 0) content-box, linear-gradient(white 0 0)",
            WebkitMaskComposite: "xor",
            maskComposite: "exclude",
          }}
        >
          {/* Rotating wrapper */}
          <div
            style={{
              width: "100%",
              height: "100%",
              position: "relative",
              animation: `border-beam-rotate ${duration}s linear infinite`,
              animationDelay: `${delay}s`,
            }}
          >
            {/* The beam light */}
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                width: `${size * 3}px`,
                height: "200%",
                background: `linear-gradient(to bottom, transparent 0%, ${colorFrom} 20%, ${colorTo} 50%, ${colorFrom} 80%, transparent 100%)`,
                transform: `translate(-50%, -50%) rotate(${anchor}deg)`,
                opacity: 4,
              }}
            />
          </div>
        </div>
      </div>
      
      <style>{`
        @keyframes border-beam-rotate {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
};

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================
const validateEmail = (email) => {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
};

const validateName = (name) => {
  return name.length >= 2 && /^[a-zA-Z\s]+$/.test(name);
};

const validatePassword = (password) => {
  return {
    length: password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
    special: /[^A-Za-z0-9]/.test(password),
  };
};

const calculatePasswordStrength = (password) => {
  if (!password) return 0;
  
  let strength = 0;
  
  if (password.length >= 8) strength += 25;
  if (password.length >= 12) strength += 15;
  if (password.length >= 16) strength += 10;
  
  if (/[a-z]/.test(password)) strength += 10;
  if (/[A-Z]/.test(password)) strength += 10;
  if (/[0-9]/.test(password)) strength += 10;
  if (/[^A-Za-z0-9]/.test(password)) strength += 15;
  
  if (/(.)\1{2,}/.test(password)) strength -= 10;
  if (/^[a-z]+$/.test(password)) strength -= 10;
  if (/^[A-Z]+$/.test(password)) strength -= 10;
  
  return Math.max(0, Math.min(100, strength));
};

const getPasswordStrengthInfo = (strength) => {
  if (strength < 30) return { label: 'Weak', color: 'bg-red-500', textColor: 'text-red-600' };
  if (strength < 60) return { label: 'Fair', color: 'bg-orange-500', textColor: 'text-orange-600' };
  if (strength < 80) return { label: 'Good', color: 'bg-yellow-500', textColor: 'text-yellow-600' };
  return { label: 'Strong', color: 'bg-green-500', textColor: 'text-green-600' };
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================
export default function SignupPage() {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
    acceptTerms: false,
  });

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const passwordStrength = calculatePasswordStrength(formData.password);
  const strengthInfo = getPasswordStrengthInfo(passwordStrength);
  const passwordChecks = validatePassword(formData.password);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));

    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const handleBlur = (field) => {
    setTouched(prev => ({ ...prev, [field]: true }));
    validateField(field);
  };

  const validateField = (field) => {
    const newErrors = { ...errors };

    switch (field) {
      case 'fullName':
        if (!formData.fullName) {
          newErrors.fullName = 'Name is required';
        } else if (!validateName(formData.fullName)) {
          newErrors.fullName = 'Name must be at least 2 characters and contain only letters';
        } else {
          delete newErrors.fullName;
        }
        break;

      case 'email':
        if (!formData.email) {
          newErrors.email = 'Email is required';
        } else if (!validateEmail(formData.email)) {
          newErrors.email = 'Invalid email address';
        } else {
          delete newErrors.email;
        }
        break;

      case 'password':
        if (!formData.password) {
          newErrors.password = 'Password is required';
        } else if (!Object.values(passwordChecks).every(Boolean)) {
          newErrors.password = 'Password does not meet requirements';
        } else {
          delete newErrors.password;
        }
        break;

      case 'confirmPassword':
        if (!formData.confirmPassword) {
          newErrors.confirmPassword = 'Please confirm your password';
        } else if (formData.password !== formData.confirmPassword) {
          newErrors.confirmPassword = 'Passwords do not match';
        } else {
          delete newErrors.confirmPassword;
        }
        break;

      case 'acceptTerms':
        if (!formData.acceptTerms) {
          newErrors.acceptTerms = 'You must accept the terms and conditions';
        } else {
          delete newErrors.acceptTerms;
        }
        break;
    }

    setErrors(newErrors);
  };

  const handleSubmit = async () => {
    const fieldsToValidate = ['fullName', 'email', 'password', 'confirmPassword', 'acceptTerms'];
    fieldsToValidate.forEach(validateField);
    
    const allTouched = {};
    fieldsToValidate.forEach(field => allTouched[field] = true);
    setTouched(allTouched);

    const hasErrors = fieldsToValidate.some(field => {
      validateField(field);
      return errors[field];
    });

    if (hasErrors || Object.keys(errors).length > 0) {
      return;
    }

    setIsSubmitting(true);

    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      alert('Account created successfully! 🎉');
      setFormData({
        fullName: '',
        email: '',
        password: '',
        confirmPassword: '',
        acceptTerms: false,
      });
      setTouched({});
    } catch (error) {
      setErrors({ submit: 'Something went wrong. Please try again.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Back button */}
        <button
          onClick={() => window.history.back()}
          className="inline-flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors mb-6 group"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
          Back to home
        </button>

        {/* Main Card */}
        <div className="relative bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          {/* Header */}
          <div className="p-6 space-y-1.5">
            <h2 className="text-2xl font-semibold leading-none tracking-tight">Create Account</h2>
            <p className="text-sm text-gray-500">
              Enter your information to create your account.
            </p>
          </div>

          {/* Content */}
          <div className="p-6 pt-0 space-y-4">
            {/* Full Name */}
            <div className="flex flex-col space-y-1.5">
              <label htmlFor="fullName" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Full Name
              </label>
              <input
                id="fullName"
                name="fullName"
                type="text"
                placeholder="John Doe"
                value={formData.fullName}
                onChange={handleChange}
                onBlur={() => handleBlur('fullName')}
                onKeyDown={handleKeyDown}
                className={`flex h-10 w-full rounded-md border ${
                  touched.fullName && errors.fullName ? 'border-red-500' : 'border-gray-300'
                } bg-white px-3 py-2 text-sm ring-offset-white file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50`}
              />
              {touched.fullName && errors.fullName && (
                <p className="text-xs text-red-600">{errors.fullName}</p>
              )}
            </div>

            {/* Email */}
            <div className="flex flex-col space-y-1.5">
              <label htmlFor="email" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                placeholder="you@example.com"
                value={formData.email}
                onChange={handleChange}
                onBlur={() => handleBlur('email')}
                onKeyDown={handleKeyDown}
                className={`flex h-10 w-full rounded-md border ${
                  touched.email && errors.email ? 'border-red-500' : 'border-gray-300'
                } bg-white px-3 py-2 text-sm ring-offset-white file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50`}
              />
              {touched.email && errors.email && (
                <p className="text-xs text-red-600">{errors.email}</p>
              )}
            </div>

            {/* Password */}
            <div className="flex flex-col space-y-1.5">
              <label htmlFor="password" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Create a strong password"
                  value={formData.password}
                  onChange={handleChange}
                  onBlur={() => handleBlur('password')}
                  onKeyDown={handleKeyDown}
                  className={`flex h-10 w-full rounded-md border ${
                    touched.password && errors.password ? 'border-red-500' : 'border-gray-300'
                  } bg-white px-3 py-2 pr-10 text-sm ring-offset-white file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50`}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>

              {/* Password Strength Indicator */}
              {formData.password && (
                <div className="mt-2 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-600">Password strength:</span>
                    <span className={`text-xs font-medium ${strengthInfo.textColor}`}>
                      {strengthInfo.label}
                    </span>
                  </div>
                  <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${strengthInfo.color}`}
                      style={{ width: `${passwordStrength}%` }}
                    />
                  </div>

                  {/* Password Requirements */}
                  <div className="grid grid-cols-2 gap-2 mt-3">
                    {[
                      { key: 'length', label: '8+ characters' },
                      { key: 'uppercase', label: 'Uppercase' },
                      { key: 'lowercase', label: 'Lowercase' },
                      { key: 'number', label: 'Number' },
                      { key: 'special', label: 'Special char' },
                    ].map(({ key, label }) => (
                      <div key={key} className="flex items-center gap-1.5">
                        {passwordChecks[key] ? (
                          <Check className="w-3.5 h-3.5 text-green-600" />
                        ) : (
                          <X className="w-3.5 h-3.5 text-gray-400" />
                        )}
                        <span className={`text-xs ${passwordChecks[key] ? 'text-green-600' : 'text-gray-500'}`}>
                          {label}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {touched.password && errors.password && (
                <p className="text-xs text-red-600">{errors.password}</p>
              )}
            </div>

            {/* Confirm Password */}
            <div className="flex flex-col space-y-1.5">
              <label htmlFor="confirmPassword" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                Confirm Password
              </label>
              <div className="relative">
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  placeholder="Re-enter your password"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  onBlur={() => handleBlur('confirmPassword')}
                  onKeyDown={handleKeyDown}
                  className={`flex h-10 w-full rounded-md border ${
                    touched.confirmPassword && errors.confirmPassword ? 'border-red-500' : 'border-gray-300'
                  } bg-white px-3 py-2 pr-10 text-sm ring-offset-white file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-gray-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50`}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {touched.confirmPassword && errors.confirmPassword && (
                <p className="text-xs text-red-600">{errors.confirmPassword}</p>
              )}
            </div>

            {/* Terms & Conditions */}
            <div className="flex items-start space-x-2">
              <input
                id="acceptTerms"
                name="acceptTerms"
                type="checkbox"
                checked={formData.acceptTerms}
                onChange={handleChange}
                onBlur={() => handleBlur('acceptTerms')}
                className="mt-0.5 h-4 w-4 rounded border-gray-300 text-gray-900 focus:ring-2 focus:ring-gray-950 focus:ring-offset-2"
              />
              <label htmlFor="acceptTerms" className="text-sm text-gray-600 leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                I agree to the{' '}
                <a href="/terms" className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">
                  Terms of Service
                </a>
                {' '}and{' '}
                <a href="/privacy" className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">
                  Privacy Policy
                </a>
              </label>
            </div>
            {touched.acceptTerms && errors.acceptTerms && (
              <p className="text-xs text-red-600">{errors.acceptTerms}</p>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center p-6 pt-0 justify-between">
            <button
              onClick={() => window.location.href = '/login'}
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-gray-300 bg-white hover:bg-gray-100 hover:text-gray-900 h-10 px-4 py-2"
            >
              Login
            </button>
            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-gray-900 text-gray-50 hover:bg-gray-900/90 h-10 px-4 py-2"
            >
              {isSubmitting ? 'Creating...' : 'Create Account'}
            </button>
          </div>

          {/* BorderBeam Effect */}
          <BorderBeam size={70} duration={8} borderWidth={3.5} />
        </div>

        
      </div>
    </div>
  );
}
/**
 * Social Authentication Component
 * 
 * Future Implementation:
 * - Google OAuth 2.0
 * - GitHub OAuth
 * - Microsoft OAuth
 * - Apple Sign In
 * 
 * Security:
 * - CSRF protection via state parameter
 * - PKCE flow for mobile apps
 * - Secure token exchange
 * 
 * Note: Currently a placeholder for future implementation
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Github, Mail, Loader2 } from 'lucide-react';
import { cn } from '@/utils/cn';
import { Button } from '@/components/ui/Button';
import { toast } from '@/components/ui/Toast';

// ============================================================================
// TYPES
// ============================================================================

export interface SocialAuthProps {
  /**
   * Auth mode: login or signup
   */
  mode?: 'login' | 'signup';
  
  /**
   * Callback after successful auth
   */
  onSuccess?: () => void;
  
  /**
   * Enabled providers
   */
  providers?: ('google' | 'github' | 'microsoft' | 'apple')[];
  
  /**
   * Additional CSS classes
   */
  className?: string;
}

interface SocialProvider {
  id: string;
  name: string;
  icon: React.ElementType;
  color: string;
  enabled: boolean;
}

// ============================================================================
// PROVIDER CONFIGURATION
// ============================================================================

const SOCIAL_PROVIDERS: SocialProvider[] = [
  {
    id: 'google',
    name: 'Google',
    icon: Mail, // Replace with Google icon
    color: 'bg-white hover:bg-gray-100 text-gray-900 border border-gray-300',
    enabled: false, // TODO: Enable when backend is ready
  },
  {
    id: 'github',
    name: 'GitHub',
    icon: Github,
    color: 'bg-gray-900 hover:bg-gray-800 text-white',
    enabled: false, // TODO: Enable when backend is ready
  },
  // Add more providers as needed
];

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const SocialAuth = React.memo<SocialAuthProps>(({
  mode = 'login',
  onSuccess,
  providers = ['google', 'github'],
  className,
}) => {
  const [loadingProvider, setLoadingProvider] = React.useState<string | null>(null);

  // Handle social login
  const handleSocialLogin = async (providerId: string) => {
    setLoadingProvider(providerId);
    
    try {
      // TODO: Implement OAuth flow
      // 1. Generate state parameter (CSRF protection)
      // 2. Redirect to OAuth provider
      // 3. Handle callback
      // 4. Exchange code for token
      
      toast.info('Social authentication', {
        description: 'Coming soon! This feature is under development.',
        duration: 3000,
      });
      
      // Simulate delay
      await new Promise((resolve) => setTimeout(resolve, 1000));
      
      // onSuccess?.();
    } catch (error) {
      toast.error('Authentication failed', {
        description: 'Please try again or use email/password',
      });
    } finally {
      setLoadingProvider(null);
    }
  };

  // Filter enabled providers
  const enabledProviders = SOCIAL_PROVIDERS.filter(
    (provider) => providers.includes(provider.id as any) && provider.enabled
  );

  // If no providers enabled, show placeholder
  if (enabledProviders.length === 0) {
    return (
      <div className={cn('text-center', className)}>
        <p className="text-sm text-text-tertiary">
          Social authentication coming soon
        </p>
        <p className="text-xs text-text-tertiary mt-1">
          Google, GitHub, and more providers will be available soon
        </p>
      </div>
    );
  }

  return (
    <div className={cn('space-y-3', className)}>
      {enabledProviders.map((provider) => (
        <motion.div
          key={provider.id}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Button
            type="button"
            variant="secondary"
            size="lg"
            fullWidth
            onClick={() => handleSocialLogin(provider.id)}
            disabled={loadingProvider !== null}
            leftIcon={
              loadingProvider === provider.id ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <provider.icon className="w-5 h-5" />
              )
            }
            className={provider.color}
          >
            {loadingProvider === provider.id
              ? `Connecting to ${provider.name}...`
              : `Continue with ${provider.name}`}
          </Button>
        </motion.div>
      ))}
    </div>
  );
});

SocialAuth.displayName = 'SocialAuth';

// ============================================================================
// EXPORTS
// ============================================================================

export default SocialAuth;

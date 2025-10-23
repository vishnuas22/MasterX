/**
 * Footer Component
 * 
 * WCAG 2.1 AA Compliant:
 * - Landmark <footer> element
 * - Sufficient link contrast
 * - Keyboard navigable
 * 
 * Performance:
 * - Static content (no re-renders)
 * - CSS-only layout
 * - Lazy load social icons
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { Github, Twitter, Linkedin, Mail, Circle } from 'lucide-react';
import { cn } from '@/utils/cn';

// ============================================================================
// TYPES
// ============================================================================

export interface FooterProps {
  /**
   * Show social media links
   * @default true
   */
  showSocial?: boolean;
  
  /**
   * Show version info
   * @default true
   */
  showVersion?: boolean;
  
  /**
   * Show online status
   * @default true
   */
  showStatus?: boolean;
  
  /**
   * Additional CSS classes
   */
  className?: string;
}

// ============================================================================
// FOOTER LINKS
// ============================================================================

const footerLinks = [
  {
    title: 'Legal',
    links: [
      { label: 'Privacy Policy', href: '/privacy' },
      { label: 'Terms of Service', href: '/terms' },
      { label: 'Cookie Policy', href: '/cookies' },
    ],
  },
  {
    title: 'Company',
    links: [
      { label: 'About Us', href: '/about' },
      { label: 'Contact', href: '/contact' },
      { label: 'Careers', href: '/careers' },
    ],
  },
  {
    title: 'Resources',
    links: [
      { label: 'Help Center', href: '/help' },
      { label: 'API Docs', href: '/docs' },
      { label: 'Blog', href: '/blog' },
    ],
  },
];

const socialLinks = [
  { icon: Github, href: 'https://github.com/masterx', label: 'GitHub' },
  { icon: Twitter, href: 'https://twitter.com/masterx', label: 'Twitter' },
  { icon: Linkedin, href: 'https://linkedin.com/company/masterx', label: 'LinkedIn' },
  { icon: Mail, href: 'mailto:hello@masterx.ai', label: 'Email' },
];

// ============================================================================
// STATUS INDICATOR
// ============================================================================

const StatusIndicator = React.memo<{ showStatus: boolean }>(({ showStatus }) => {
  const [isOnline, setIsOnline] = React.useState(typeof navigator !== 'undefined' ? navigator.onLine : true);

  React.useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (!showStatus) return null;

  return (
    <div className="flex items-center gap-2 text-xs text-text-tertiary">
      <Circle
        className={cn(
          'w-2 h-2 fill-current',
          isOnline ? 'text-accent-success' : 'text-accent-error'
        )}
      />
      <span>{isOnline ? 'Online' : 'Offline'}</span>
    </div>
  );
});

StatusIndicator.displayName = 'StatusIndicator';

// ============================================================================
// MAIN FOOTER COMPONENT
// ============================================================================

export const Footer = React.memo<FooterProps>(({
  showSocial = true,
  showVersion = true,
  showStatus = true,
  className,
}) => {
  const currentYear = new Date().getFullYear();
  const appVersion = import.meta.env.VITE_APP_VERSION || '1.0.0';

  return (
    <footer
      className={cn(
        'border-t border-white/10 bg-bg-primary',
        'py-8 px-4',
        className
      )}
      role="contentinfo"
    >
      <div className="max-w-7xl mx-auto">
        {/* Main footer content */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand section */}
          <div className="col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-primary to-accent-purple" />
              <span className="text-lg font-semibold">MasterX</span>
            </div>
            <p className="text-sm text-text-secondary mb-4">
              AI-powered adaptive learning platform with real-time emotion detection
            </p>
            
            {/* Status indicator */}
            <StatusIndicator showStatus={showStatus} />
          </div>

          {/* Link sections */}
          {footerLinks.map((section) => (
            <div key={section.title}>
              <h3 className="text-sm font-semibold text-text-primary mb-3">
                {section.title}
              </h3>
              <ul className="space-y-2">
                {section.links.map((link) => (
                  <li key={link.label}>
                    <Link
                      to={link.href}
                      className="text-sm text-text-secondary hover:text-text-primary transition-colors focus-ring rounded"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom section */}
        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Copyright */}
          <p className="text-sm text-text-tertiary">
            © {currentYear} MasterX. All rights reserved.
          </p>

          {/* Version */}
          {showVersion && (
            <p className="text-xs text-text-tertiary">
              Version {appVersion}
            </p>
          )}

          {/* Social links */}
          {showSocial && (
            <div className="flex items-center gap-4">
              {socialLinks.map((social) => (
                <a
                  key={social.label}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-text-tertiary hover:text-text-primary transition-colors focus-ring rounded"
                  aria-label={social.label}
                >
                  <social.icon className="w-5 h-5" />
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
    </footer>
  );
});

Footer.displayName = 'Footer';

// ============================================================================
// EXPORTS
// ============================================================================

export default Footer;

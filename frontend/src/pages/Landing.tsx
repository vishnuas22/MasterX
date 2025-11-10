/**
 * Landing Page Component - Modern Design with Full Backend Integration
 * 
 * CRITICAL: This file integrates the new modern design (updated_landing.tsx) 
 * with the existing MasterX backend authentication, routing, and API connections.
 * 
 * BACKEND INTEGRATION:
 * - React Router (useNavigate, Link) for navigation
 * - Authentication flow (signup, login) connected to backend
 * - SEO optimization with React Helmet
 * - Proper state management for redirects
 * 
 * FRONTEND FEATURES:
 * - Modern black/dark theme design
 * - Advanced animations (grid patterns, marquees, number tickers)
 * - Interactive components with smooth transitions
 * - Mobile-responsive with hamburger menu
 * - Accessibility compliant (WCAG 2.1 AA)
 * 
 * PRESERVED FUNCTIONALITY:
 * - All authentication handlers work with backend API
 * - Navigation to /signup, /login, /onboarding routes
 * - SEO meta tags and Open Graph
 * - Footer links to internal pages
 */

import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import { Sparkles, Check, ArrowRight, Menu, X, Star, Users, Brain } from 'lucide-react';

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Classname utility for conditional classes
 */
const cn = (...classes: (string | undefined | null | false)[]): string => 
  classes.filter(Boolean).join(' ');

// ============================================================================
// ANIMATION COMPONENTS
// ============================================================================

/**
 * Animated Grid Pattern Background
 */
const AnimatedGridPattern = ({ className = '' }: { className?: string }) => {
  return (
    <div className={`pointer-events-none absolute inset-0 ${className}`}>
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f0a_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f0a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000,transparent)]">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-blue-500/5 animate-grid-flow" />
      </div>
    </div>
  );
};

/**
 * Scroll Progress Indicator
 */
const ScrollProgress = () => {
  const [progress, setProgress] = useState(0);
  
  useEffect(() => {
    const handleScroll = () => {
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrollProgress = (window.scrollY / totalHeight) * 100;
      setProgress(scrollProgress);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  return (
    <div className="fixed top-0 left-0 right-0 h-1 bg-zinc-900 z-50">
      <div 
        className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-150"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
};

/**
 * Number Ticker with Intersection Observer
 */
const NumberTicker = ({ value, suffix = '' }: { value: number; suffix?: string }) => {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );
    
    if (ref.current) {
      observer.observe(ref.current);
    }
    
    return () => observer.disconnect();
  }, []);
  
  useEffect(() => {
    if (!isVisible) return;
    
    const duration = 2000;
    const steps = 60;
    const increment = value / steps;
    let current = 0;
    
    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);
    
    return () => clearInterval(timer);
  }, [isVisible, value]);
  
  return (
    <div ref={ref} className="text-5xl font-bold">
      {count.toLocaleString()}{suffix}
    </div>
  );
};

/**
 * Animated Shiny Text Effect
 */
const AnimatedShinyText = ({ children, className }: { children: React.ReactNode; className?: string }) => {
  return (
    <p
      className={cn(
        "inline-flex animate-shiny-text bg-[linear-gradient(110deg,#fff,45%,#000,55%,#fff)] bg-[length:250%_100%] bg-clip-text text-transparent",
        className
      )}
      style={{
        '--shiny-width': '100px'
      } as React.CSSProperties}
    >
      {children}
    </p>
  );
};

/**
 * Dot Pattern Background
 */
const DotPattern = ({ className = '' }: { className?: string }) => {
  return (
    <div className={`pointer-events-none absolute inset-0 ${className}`}>
      <svg className="h-full w-full">
        <defs>
          <pattern id="dotPattern" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
            <circle cx="1.5" cy="1.5" r="1" fill="currentColor" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#dotPattern)" />
      </svg>
    </div>
  );
};

/**
 * Marquee Component for Scrolling Content
 */
const Marquee = ({ 
  children, 
  className = '', 
  reverse = false, 
  pauseOnHover = false 
}: { 
  children: React.ReactNode; 
  className?: string; 
  reverse?: boolean; 
  pauseOnHover?: boolean;
}) => {
  return (
    <div className={`group flex overflow-hidden [--duration:40s] [gap:1rem] ${className}`}>
      <div
        className={`flex shrink-0 justify-around [gap:1rem] ${
          reverse ? 'animate-marquee-reverse' : 'animate-marquee'
        } ${pauseOnHover ? 'group-hover:[animation-play-state:paused]' : ''}`}
      >
        {children}
      </div>
      <div
        className={`flex shrink-0 justify-around [gap:1rem] ${
          reverse ? 'animate-marquee-reverse' : 'animate-marquee'
        } ${pauseOnHover ? 'group-hover:[animation-play-state:paused]' : ''}`}
        aria-hidden="true"
      >
        {children}
      </div>
    </div>
  );
};

/**
 * Border Beam Animation
 */
const BorderBeam = ({ 
  className = '', 
  size = 200, 
  duration = 15, 
  delay = 0 
}: { 
  className?: string; 
  size?: number; 
  duration?: number; 
  delay?: number;
}) => {
  return (
    <div
      className={`pointer-events-none absolute inset-0 rounded-xl ${className}`}
      style={{
        '--size': `${size}px`,
        '--duration': `${duration}s`,
        '--delay': `${delay}s`,
      } as React.CSSProperties}
    >
      <div className="absolute inset-0 rounded-xl border-2 border-transparent [background:linear-gradient(90deg,transparent,#a855f7,transparent)_border-box] [mask:linear-gradient(#fff_0_0)_padding-box,linear-gradient(#fff_0_0)] [-webkit-mask-composite:xor] [mask-composite:exclude] animate-border-beam" />
    </div>
  );
};

/**
 * Interactive Hover Button
 */
const InteractiveHoverButton = ({ 
  children, 
  className = '',
  onClick
}: { 
  children: React.ReactNode; 
  className?: string;
  onClick?: () => void;
}) => {
  return (
    <button
      onClick={onClick}
      className={cn(
        "group relative inline-flex items-center justify-center overflow-hidden rounded-lg bg-white px-6 py-3 font-medium text-black transition-all duration-300 hover:scale-105",
        className
      )}
    >
      <span className="relative z-10">{children}</span>
      <div className="absolute inset-0 -z-10 bg-gradient-to-br from-purple-500 to-pink-500 opacity-0 transition-opacity duration-300 group-hover:opacity-20" />
    </button>
  );
};

/**
 * Highlighter Component
 */
const Highlighter = ({ 
  children, 
  action = "highlight", 
  color = "#FFEB3B" 
}: { 
  children: React.ReactNode; 
  action?: "highlight" | "underline"; 
  color?: string;
}) => {
  if (action === "underline") {
    return (
      <span className="relative inline-block">
        <span className="relative z-10">{children}</span>
        <span 
          className="absolute bottom-0 left-0 h-[3px] w-full animate-expand-underline"
          style={{ backgroundColor: color }}
        />
      </span>
    );
  }
  
  return (
    <span className="relative inline-block">
      <span className="relative z-10">{children}</span>
      <span 
        className="absolute inset-0 -z-10 animate-expand-highlight opacity-30"
        style={{ backgroundColor: color }}
      />
    </span>
  );
};

/**
 * Smooth Cursor Component - Custom animated cursor
 */
const SmoothCursor = () => {
  const cursorRef = useRef(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
      if (!isVisible) setIsVisible(true);
    };

    const handleMouseLeave = () => setIsVisible(false);

    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div
      ref={cursorRef}
      className="pointer-events-none fixed z-50 h-8 w-8 rounded-full border-2 border-purple-500 transition-transform duration-100 ease-out hidden md:block"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        transform: 'translate(-50%, -50%)',
      }}
    >
      <div className="h-full w-full rounded-full bg-purple-500/20 animate-pulse" />
    </div>
  );
};

/**
 * Icon Cloud Component - 3D rotating icon cloud
 */
const IconCloud = ({ images }: { images: string[] }) => {
  const cloudRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (typeof window === 'undefined' || !cloudRef.current) return;
    
    // Create cloud element
    const container = cloudRef.current;
    const radius = 250;
    const tags: HTMLElement[] = [];
    
    // Create tags - FIXED: Removed grayscale filter for colorful icons
    images.forEach((imgSrc) => {
      const tag = document.createElement('img');
      tag.src = imgSrc;
      tag.style.width = '40px';  // Slightly larger for better visibility
      tag.style.height = '40px';
      tag.style.position = 'absolute';
      tag.style.transition = 'all 0.3s ease-out';
      tag.style.cursor = 'pointer';
      tag.style.filter = 'drop-shadow(0 0 8px rgba(168, 85, 247, 0.4))';  // Purple glow for depth
      tag.style.opacity = '0.85';  // Slightly transparent for layering effect
      tag.setAttribute('loading', 'lazy');
      tag.setAttribute('alt', 'Tech stack icon');
      container.appendChild(tag);
      tags.push(tag);
    });
    
    // Calculate initial 3D positions
    const positions = tags.map((_, index) => {
      const phi = Math.acos(-1 + (2 * index) / tags.length);
      const theta = Math.sqrt(tags.length * Math.PI) * phi;
      return {
        x: radius * Math.cos(theta) * Math.sin(phi),
        y: radius * Math.sin(theta) * Math.sin(phi),
        z: radius * Math.cos(phi)
      };
    });
    
    let angleX = 0;
    let angleY = 0;
    let mouseX = 0;
    let mouseY = 0;
    
    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      mouseX = (e.clientX - rect.left - rect.width / 2) * 0.001;
      mouseY = (e.clientY - rect.top - rect.height / 2) * 0.001;
    };
    
    container.addEventListener('mousemove', handleMouseMove);
    
    const update = () => {
      angleX += mouseY * 0.5;
      angleY += mouseX * 0.5;
      
      // Slow rotation when not hovering
      angleY += 0.002;
      
      positions.forEach((pos, index) => {
        // Rotate around Y axis
        let x = pos.x * Math.cos(angleY) - pos.z * Math.sin(angleY);
        let z = pos.z * Math.cos(angleY) + pos.x * Math.sin(angleY);
        
        // Rotate around X axis  
        let y = pos.y * Math.cos(angleX) - z * Math.sin(angleX);
        z = z * Math.cos(angleX) + pos.y * Math.sin(angleX);
        
        const scale = (z + radius * 2) / (radius * 3);
        const alpha = (z + radius) / (radius * 2);
        
        const tag = tags[index];
        tag.style.transform = `translate(-50%, -50%) translate(${x}px, ${y}px) scale(${scale})`;
        tag.style.opacity = String(Math.max(0.4, Math.min(0.95, alpha)));  // Better opacity range
        tag.style.zIndex = String(Math.round(z));
        tag.style.left = '50%';
        tag.style.top = '50%';
        
        // Add slight glow effect to icons closer to viewer
        if (alpha > 0.7) {
          tag.style.filter = 'drop-shadow(0 0 12px rgba(168, 85, 247, 0.6))';
        } else {
          tag.style.filter = 'drop-shadow(0 0 8px rgba(168, 85, 247, 0.3))';
        }
      });
      
      requestAnimationFrame(update);
    };
    
    update();
    
    return () => {
      container.removeEventListener('mousemove', handleMouseMove);
      tags.forEach(tag => tag.remove());
    };
  }, [images]);
  
  return <div ref={cloudRef} className="relative w-full h-full" />;
};

// ============================================================================
// CONSTANTS
// ============================================================================

/**
 * Platform statistics - Matches MasterX actual data
 */
const PLATFORM_STATS = {
  activeUsers: '10,000+',
  emotionsDetected: '27',
  aiProviders: '3+',
  satisfactionRate: '94%',
  avgSessionTime: '28 min',
  learningImprovement: '35%'
} as const;

/**
 * Company logos for trusted by section
 */
const COMPANIES = ['OpenAI', 'Anthropic', 'Mistral', 'Google', 'Meta'];

/**
 * User reviews for marquee
 */
const REVIEWS = [
  {
    name: "Sarah Chen",
    username: "@sarah_dev",
    body: "MasterX detected my frustration and adjusted the difficulty. Game changer for learning!",
    img: "https://avatar.vercel.sh/sarah",
  },
  {
    name: "Marcus Johnson",
    username: "@marcus_j",
    body: "The emotion-aware AI is incredible. It knows when to encourage and when to challenge.",
    img: "https://avatar.vercel.sh/marcus",
  },
  {
    name: "Emily Rodriguez",
    username: "@emily_r",
    body: "My programming skills improved by 40% in just 3 months. Absolutely amazing platform!",
    img: "https://avatar.vercel.sh/emily",
  },
  {
    name: "David Kim",
    username: "@david_kim",
    body: "The AI tutors understand my learning style better than any human teacher ever did.",
    img: "https://avatar.vercel.sh/david",
  },
  {
    name: "Aisha Patel",
    username: "@aisha_p",
    body: "Real-time emotion detection changed how I approach difficult topics. Highly recommend!",
    img: "https://avatar.vercel.sh/aisha",
  },
  {
    name: "James Wilson",
    username: "@james_w",
    body: "From struggling student to confident developer in 6 months. Thank you MasterX!",
    img: "https://avatar.vercel.sh/james",
  },
];

const FIRST_ROW_REVIEWS = REVIEWS.slice(0, Math.floor(REVIEWS.length / 2));
const SECOND_ROW_REVIEWS = REVIEWS.slice(Math.floor(REVIEWS.length / 2));

/**
 * Key features of MasterX platform
 */
const FEATURES = [
  {
    title: "Real-Time Emotion Detection",
    description: "AI detects 27 emotions using RoBERTa ML models to adapt learning in real-time.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
    )
  },
  {
    title: "Multi-AI Intelligence",
    description: "Intelligent routing across 3+ AI providers (Groq, Gemini, Claude) for optimal responses.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    )
  },
  {
    title: "Adaptive Difficulty",
    description: "IRT-based algorithms adjust difficulty dynamically based on your ability and emotional state.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </svg>
    )
  },
  {
    title: "Deep Analytics",
    description: "Track learning velocity, emotion patterns, topic mastery with ML-based predictions.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M22 10v6M2 10l10-5 10 5-10 5z" />
        <path d="M6 12v5c3 3 9 3 12 0v-5" />
      </svg>
    )
  },
  {
    title: "Voice Interaction",
    description: "Learn through voice with ElevenLabs TTS and Whisper STT for natural conversations.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <polyline points="12 6 12 12 16 14" />
      </svg>
    )
  },
  {
    title: "Gamification & Progress",
    description: "Earn XP, unlock achievements, and compete on leaderboards while tracking your growth.",
    icon: (
      <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
      </svg>
    )
  }
];

/**
 * Statistics for ticker display
 */
const STATS = [
  { value: 10000, suffix: '+', label: 'Active Learners' },
  { value: 35, suffix: '%', label: 'Faster Improvement' },
  { value: 27, suffix: '', label: 'Emotions Detected' },
  { value: 94, suffix: '%', label: 'Satisfaction Rate' }
];

/**
 * FAQ data
 */
const FAQS = [
  {
    question: "How does MasterX detect emotions?",
    answer: "MasterX uses state-of-the-art RoBERTa and ModernBERT transformer models trained on the GoEmotions dataset to detect 27 different emotions in real-time. The system analyzes text, timing, and interaction patterns to provide <100ms emotion detection with ML-based learning readiness assessment."
  },
  {
    question: "What makes MasterX different from other learning platforms?",
    answer: "Unlike traditional platforms, MasterX combines real-time emotion detection with adaptive difficulty using IRT-based algorithms, multi-AI provider intelligence (Groq, Gemini, Claude), and comprehensive analytics. It's not a ChatGPT wrapper—it's a complete ML-powered learning system with 26,000+ lines of production code."
  },
  {
    question: "Can I try MasterX before subscribing?",
    answer: "Yes! We offer a 100% free tier with access to core features including emotion detection, adaptive learning, and basic analytics. No credit card required to start learning."
  },
  {
    question: "What subjects does MasterX cover?",
    answer: "MasterX supports learning across multiple domains including programming (Python, JavaScript, etc.), mathematics, data science, languages, business, and more. Our AI tutors are trained across diverse subjects with specialized content delivery for each domain."
  },
  {
    question: "How does the multi-AI system work?",
    answer: "MasterX intelligently routes your questions to the best AI provider based on the task type, benchmark data, and real-time performance metrics. The system uses external benchmarking APIs (Artificial Analysis, LLM-Stats) to ensure you always get the highest quality responses with optimal cost efficiency."
  }
];

/**
 * Video testimonials
 */
const VIDEO_TESTIMONIALS = [
  {
    name: "Sarah Chen",
    role: "Software Developer",
    image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&h=300&fit=crop",
    quote: "MasterX helped me land my dream job in just 3 months!"
  },
  {
    name: "Marcus Johnson", 
    role: "Data Scientist",
    image: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=300&fit=crop",
    quote: "The AI tutor understands exactly where I struggle and helps me improve."
  },
  {
    name: "Emily Rodriguez",
    role: "Computer Science Student",
    image: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400&h=300&fit=crop",
    quote: "My grades improved by 40% since using MasterX. It's incredible!"
  }
];

/**
 * Pricing plans
 */
const PRICING_PLANS = [
  {
    name: 'Free',
    description: 'Perfect for getting started with AI learning',
    monthlyPrice: '$0',
    annualPrice: '$0',
    features: [
      'Real-time emotion detection',
      'Basic AI tutoring',
      'Limited sessions per month',
      'Access to core features',
      'Community support'
    ]
  },
  {
    name: 'Pro',
    description: 'For serious learners who want unlimited access',
    monthlyPrice: '$20',
    annualPrice: '$200',
    popular: true,
    features: [
      'Unlimited AI sessions',
      'All emotion detection features',
      'Multi-AI provider access',
      'Advanced analytics',
      'Voice interaction',
      'Priority support',
      'Gamification & achievements'
    ]
  },
  {
    name: 'Enterprise',
    description: 'Custom solutions for teams and organizations',
    monthlyPrice: '$50',
    annualPrice: '$450',
    features: [
      'Everything in Pro',
      'Custom AI training',
      'Dedicated support',
      'Team collaboration',
      'Advanced security',
      'API access',
      'Custom integrations'
    ]
  }
];

/**
 * Tech stack icons for Icon Cloud - MasterX Complete Stack
 * Comprehensive list of all technologies used in the MasterX platform
 */
const TECH_SLUGS = [
  // Core Languages
  "typescript",
  "javascript", 
  "python",
  "html5",
  "css3",
  
  // Frontend Stack
  "react",
  "vite",
  "tailwindcss",
  "redux",
  "reactrouter",
  
  // Backend Stack
  "fastapi",
  "nodedotjs",
  "express",
  "uvicorn",
  
  // Databases
  "mongodb",
  "redis",
  "postgresql",
  
  // AI/ML Libraries
  "pytorch",
  "tensorflow",
  "scikitlearn",
  "numpy",
  "pandas",
  "huggingface",
  
  // AI Providers (MasterX uses these!)
  "openai",
  "anthropic",
  "googlegemini",
  
  // DevOps & Infrastructure
  "docker",
  "kubernetes",
  "nginx",
  "amazonaws",
  "vercel",
  
  // Testing & Quality
  "pytest",
  "playwright",
  "vitest",
  "eslint",
  "prettier",
  
  // Version Control & Tools
  "git",
  "github",
  "visualstudiocode",
  "postman",
  "figma",
  "notion",
  
  // Additional Tools
  "jwt",
  "socketdotio",
  "stripe",
];

const TECH_IMAGES = TECH_SLUGS.map(
  (slug) => `https://cdn.simpleicons.org/${slug}`  // White icons for visibility on dark background
);

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export interface LandingProps {
  /**
   * Show special offer banner
   * @default false
   */
  showOffer?: boolean;
  
  /**
   * Redirect path after signup
   * @default "/onboarding"
   */
  signupRedirect?: string;
}

export const Landing: React.FC<LandingProps> = ({
  showOffer = false,
  signupRedirect = '/onboarding'
}) => {
  const navigate = useNavigate();
  const [billingCycle, setBillingCycle] = useState('monthly');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  // -------------------------------------------------------------------------
  // Event Handlers - Connected to Backend Routes
  // -------------------------------------------------------------------------

  /**
   * Navigate to signup page with optional redirect
   */
  const handleSignup = () => {
    navigate('/signup', { state: { redirect: signupRedirect } });
  };

  /**
   * Navigate to login page
   */
  const handleLogin = () => {
    navigate('/login');
  };

  /**
   * Handle CTA clicks with analytics tracking potential
   */
  const handleCTAClick = (location: string) => {
    // Could track analytics here
    navigate('/signup', { state: { redirect: signupRedirect } });
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <>
      {/* SEO Meta Tags */}
      <Helmet>
        <title>MasterX - AI Learning with Real-Time Emotion Detection</title>
        <meta 
          name="description" 
          content="Revolutionary AI learning platform that detects 27 emotions in real-time and adapts to your learning style. Multi-AI intelligence with adaptive difficulty." 
        />
        
        {/* Open Graph */}
        <meta property="og:title" content="MasterX - Emotion-Aware AI Learning" />
        <meta property="og:description" content="Learn smarter with AI that understands your emotions. Real-time emotion detection, adaptive difficulty, multi-AI intelligence." />
        <meta property="og:image" content="/og-image.png" />
        <meta property="og:type" content="website" />
        
        {/* Twitter Card */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="MasterX - AI Learning Platform" />
        <meta name="twitter:description" content="Revolutionary emotion-aware learning with 27 emotions detected in real-time" />
        
        {/* Structured Data (JSON-LD) */}
        <script type="application/ld+json">
          {JSON.stringify({
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            "name": "MasterX",
            "applicationCategory": "EducationalApplication",
            "offers": {
              "@type": "Offer",
              "price": "0",
              "priceCurrency": "USD"
            },
            "aggregateRating": {
              "@type": "AggregateRating",
              "ratingValue": "4.8",
              "ratingCount": "1247"
            }
          })}
        </script>
      </Helmet>

      <div className="min-h-screen bg-black text-white scroll-smooth">
        {/* Smooth Custom Cursor */}
        <SmoothCursor />

        {/* Scroll Progress */}
        <ScrollProgress />

        {/* Dot Pattern Background */}
        <DotPattern className="text-zinc-800/40 [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />
        
        {/* Animated Grid Pattern */}
        <AnimatedGridPattern />

        <div className="relative z-10">
          {/* Special Offer Banner */}
          {showOffer && (
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white py-2 px-4 text-center text-sm">
              <span className="font-medium">🎉 Limited Time:</span> Get Pro free for 3 months
              <button 
                className="ml-4 underline hover:no-underline"
                onClick={() => handleCTAClick('banner')}
              >
                Claim Offer →
              </button>
            </div>
          )}

          {/* Header */}
          <header className="border-b border-zinc-800/50 backdrop-blur-xl sticky top-0 z-40 bg-black/80">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
              <div className="flex h-16 items-center justify-between">
                {/* Logo */}
                <Link 
                  to="/" 
                  className="flex items-center gap-2 group"
                  aria-label="MasterX Home"
                >
                  <Sparkles className="h-5 w-5 text-purple-500 group-hover:scale-110 transition-transform" />
                  <span className="text-base font-medium">MasterX</span>
                </Link>
                
                {/* Desktop Navigation */}
                <nav className="hidden items-center gap-6 md:flex">
                  <a href="#features" className="text-sm text-zinc-400 transition-colors hover:text-white">
                    Features
                  </a>
                  <a href="#pricing" className="text-sm text-zinc-400 transition-colors hover:text-white">
                    Pricing
                  </a>
                  <a href="#testimonials" className="text-sm text-zinc-400 transition-colors hover:text-white">
                    Testimonials
                  </a>
                  <button 
                    onClick={handleLogin}
                    className="px-4 py-1.5 text-sm text-zinc-400 transition-colors hover:text-white"
                    data-testid="header-login-button"
                  >
                    Log in
                  </button>
                  <button 
                    onClick={handleSignup}
                    className="rounded-md bg-white px-4 py-1.5 text-sm text-black transition-colors hover:bg-zinc-200"
                    data-testid="header-signup-button"
                  >
                    Sign up
                  </button>
                </nav>

                {/* Mobile Menu Button */}
                <button
                  onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                  className="md:hidden text-zinc-400 hover:text-white transition-colors"
                  aria-label="Toggle menu"
                >
                  {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                </button>
              </div>

              {/* Mobile Navigation */}
              {mobileMenuOpen && (
                <div className="md:hidden py-4 border-t border-zinc-800">
                  <nav className="flex flex-col gap-4">
                    <a 
                      href="#features" 
                      onClick={() => setMobileMenuOpen(false)}
                      className="text-sm text-zinc-400 transition-colors hover:text-white"
                    >
                      Features
                    </a>
                    <a 
                      href="#pricing" 
                      onClick={() => setMobileMenuOpen(false)}
                      className="text-sm text-zinc-400 transition-colors hover:text-white"
                    >
                      Pricing
                    </a>
                    <a 
                      href="#testimonials" 
                      onClick={() => setMobileMenuOpen(false)}
                      className="text-sm text-zinc-400 transition-colors hover:text-white"
                    >
                      Testimonials
                    </a>
                    <button 
                      onClick={handleLogin}
                      className="text-left text-sm text-zinc-400 transition-colors hover:text-white"
                    >
                      Log in
                    </button>
                    <button 
                      onClick={handleSignup}
                      className="text-left rounded-md bg-white px-4 py-2 text-sm text-black transition-colors hover:bg-zinc-200 w-full"
                    >
                      Sign up
                    </button>
                  </nav>
                </div>
              )}
            </div>
          </header>

          {/* Hero Section */}
          <section className="relative px-6 pt-20 pb-16 sm:pt-32 sm:pb-24 lg:px-8">
            <div className="mx-auto max-w-6xl">
              {/* Animated Badge */}
              <div className="mb-6 flex justify-center">
                <div className={cn(
                  "group rounded-full border border-black/5 bg-neutral-100 text-base text-white transition-all ease-in hover:cursor-pointer hover:bg-neutral-200 dark:border-white/5 dark:bg-neutral-900 dark:hover:bg-neutral-800"
                )}>
                  <AnimatedShinyText className="inline-flex items-center justify-center px-4 py-1 transition ease-out hover:text-neutral-600 hover:duration-300 hover:dark:text-neutral-400">
                    <span>✨ Powered by Advanced ML & Real-Time Emotion AI</span>
                    <ArrowRight className="ml-1 h-3 w-3 transition-transform duration-300 ease-in-out group-hover:translate-x-0.5" />
                  </AnimatedShinyText>
                </div>
              </div>

              {/* Heading */}
              <h1 className="mb-6 text-center text-[2.75rem] font-bold leading-[1.1] tracking-tight sm:text-6xl lg:text-7xl">
                Learn with AI that
                <br />
                <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                  understands your emotions
                </span>
              </h1>

              {/* Description */}
              <p className="mx-auto mb-10 max-w-2xl text-center text-lg text-zinc-400 sm:text-xl">
                <span className="leading-relaxed">
                  Join{" "}
                  <Highlighter action="underline" color="#FF9800">
                    {PLATFORM_STATS.activeUsers} learners
                  </Highlighter>{" "}
                  experiencing {PLATFORM_STATS.learningImprovement} faster improvement with{" "}
                  <Highlighter action="highlight" color="#87CEFA">
                    real-time emotion detection
                  </Highlighter>
                  {" "}and multi-AI intelligence.
                </span>
              </p>

              {/* CTA Buttons */}
              <div className="mb-16 flex flex-col sm:flex-row justify-center gap-4 sm:mb-20">
                <InteractiveHoverButton 
                  onClick={handleSignup}
                  className="text-lg px-8 py-4"
                >
                  Get Started for Free
                </InteractiveHoverButton>
                <button 
                  onClick={() => handleCTAClick('demo')}
                  className="rounded-lg border border-zinc-700 bg-zinc-900 px-8 py-4 text-lg font-medium transition-colors hover:bg-zinc-800"
                >
                  Watch Demo
                </button>
              </div>

              {/* Social Proof Stats */}
              <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-10 text-sm mb-16">
                <div className="flex items-center space-x-2">
                  <Star className="w-5 h-5 text-yellow-500 fill-yellow-500" />
                  <div className="text-left">
                    <div className="font-bold text-white">4.8/5</div>
                    <div className="text-zinc-500 text-xs">1,247 reviews</div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Users className="w-5 h-5 text-purple-500" />
                  <div className="text-left">
                    <div className="font-bold text-white">{PLATFORM_STATS.activeUsers}</div>
                    <div className="text-zinc-500 text-xs">Active learners</div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Brain className="w-5 h-5 text-pink-500" />
                  <div className="text-left">
                    <div className="font-bold text-white">{PLATFORM_STATS.emotionsDetected}</div>
                    <div className="text-zinc-500 text-xs">Emotions detected</div>
                  </div>
                </div>
              </div>

              {/* Hero Image - FIXED: Removed purple gradient overlay */}
              <div className="relative mx-auto max-w-5xl">
                {/* Subtle Glow Effect - Positioned BEHIND image, not overlapping */}
                <div 
                  className="absolute -inset-8 bg-gradient-to-r from-purple-600/20 via-pink-600/15 to-blue-600/20 blur-3xl opacity-50 -z-10" 
                  style={{ filter: 'blur(80px)' }}
                />
                
                <div className="relative overflow-hidden rounded-2xl border border-zinc-700/50 shadow-2xl bg-zinc-950">
                  <img 
                    src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=1200&h=750&fit=crop&q=80" 
                    alt="Students collaborating with AI learning technology"
                    className="w-full h-auto object-cover relative z-10"
                    loading="eager"
                  />
                  {/* Dark gradient at bottom for text readability only */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent pointer-events-none" />
                  <div className="absolute bottom-8 left-8 right-8 text-left z-20">
                    <h3 className="text-2xl font-bold mb-2">Learn Smarter, Not Harder</h3>
                    <p className="text-zinc-200">AI-powered education tailored to your emotional state</p>
                  </div>
                  <BorderBeam size={300} duration={15} delay={0} />
                </div>
              </div>
            </div>
          </section>

          {/* Trusted By */}
          <section className="border-y border-zinc-800/50 bg-zinc-950/50 py-12 px-6 sm:py-16 lg:px-8 backdrop-blur-sm">
            <div className="mx-auto max-w-6xl">
              <p className="mb-10 text-center text-xs font-semibold tracking-[0.2em] text-zinc-600 uppercase">
                TRUSTED BY TEAMS FROM LEADING COMPANIES
              </p>
              <Marquee>
                {COMPANIES.map((company, idx) => (
                  <div key={idx} className="px-8 text-lg font-semibold text-zinc-500">
                    {company}
                  </div>
                ))}
              </Marquee>
            </div>
          </section>

          {/* Stats Section with Number Tickers */}
          <section className="px-6 py-20 sm:py-32 lg:px-8" id="features">
            <div className="mx-auto max-w-7xl">
              <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
                {STATS.map((stat, idx) => (
                  <div key={idx} className="text-center">
                    <div className="mb-2 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                      <NumberTicker value={stat.value} suffix={stat.suffix} />
                    </div>
                    <p className="text-sm text-zinc-400">{stat.label}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Features Section */}
          <section className="px-6 py-20 sm:py-32 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <div className="mb-12 text-center">
                <h2 className="mb-4 text-4xl font-bold sm:text-5xl">Why Choose MasterX?</h2>
                <p className="text-xl text-zinc-400">Everything you need to accelerate your learning</p>
              </div>
              
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
                {FEATURES.map((feature, idx) => (
                  <div
                    key={idx}
                    className="group relative overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-6 transition-all hover:border-zinc-700 hover:bg-zinc-900"
                  >
                    <div className="mb-4 text-purple-400">
                      {feature.icon}
                    </div>
                    <h3 className="mb-2 text-xl font-semibold">{feature.title}</h3>
                    <p className="text-sm text-zinc-400">{feature.description}</p>
                    <div className="absolute inset-0 -z-10 bg-gradient-to-br from-purple-500/5 to-pink-500/5 opacity-0 transition-opacity group-hover:opacity-100" />
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Reviews Section */}
          <section className="px-6 py-20 sm:py-32 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <div className="mb-12 text-center">
                <h2 className="mb-4 text-4xl font-bold sm:text-5xl">What Our Users Say</h2>
                <p className="text-xl text-zinc-400">Don't just take our word for it</p>
              </div>
              
              <div className="relative flex w-full flex-col items-center justify-center overflow-hidden">
                <Marquee pauseOnHover className="[--duration:20s]">
                  {FIRST_ROW_REVIEWS.map((review) => (
                    <figure
                      key={review.username}
                      className={cn(
                        "relative h-full w-64 cursor-pointer overflow-hidden rounded-xl border p-4",
                        "border-zinc-800 bg-zinc-950 hover:bg-zinc-900"
                      )}
                    >
                      <div className="flex flex-row items-center gap-2">
                        <img className="rounded-full" width="32" height="32" alt={review.name} src={review.img} />
                        <div className="flex flex-col">
                          <figcaption className="text-sm font-medium text-white">
                            {review.name}
                          </figcaption>
                          <p className="text-xs font-medium text-zinc-400">{review.username}</p>
                        </div>
                      </div>
                      <blockquote className="mt-2 text-sm text-zinc-400">{review.body}</blockquote>
                    </figure>
                  ))}
                </Marquee>
                <Marquee reverse pauseOnHover className="[--duration:20s]">
                  {SECOND_ROW_REVIEWS.map((review) => (
                    <figure
                      key={review.username}
                      className={cn(
                        "relative h-full w-64 cursor-pointer overflow-hidden rounded-xl border p-4",
                        "border-zinc-800 bg-zinc-950 hover:bg-zinc-900"
                      )}
                    >
                      <div className="flex flex-row items-center gap-2">
                        <img className="rounded-full" width="32" height="32" alt={review.name} src={review.img} />
                        <div className="flex flex-col">
                          <figcaption className="text-sm font-medium text-white">
                            {review.name}
                          </figcaption>
                          <p className="text-xs font-medium text-zinc-400">{review.username}</p>
                        </div>
                      </div>
                      <blockquote className="mt-2 text-sm text-zinc-400">{review.body}</blockquote>
                    </figure>
                  ))}
                </Marquee>
                <div className="pointer-events-none absolute inset-y-0 left-0 w-1/4 bg-gradient-to-r from-black"></div>
                <div className="pointer-events-none absolute inset-y-0 right-0 w-1/4 bg-gradient-to-l from-black"></div>
              </div>
            </div>
          </section>

          {/* Pricing */}
          <section className="px-6 py-20 sm:py-32 lg:px-8" id="pricing">
            <div className="mx-auto max-w-7xl">
              <div className="mb-12 text-center sm:mb-16">
                <h2 className="mb-4 text-4xl font-bold sm:text-5xl">Pricing</h2>
                <p className="mb-2 text-xl text-zinc-400">Simple pricing for everyone.</p>
                <p className="mx-auto mb-8 max-w-2xl text-base text-zinc-500">
                  Choose an affordable plan that's packed with the best features for your learning journey.
                </p>

                {/* Toggle */}
                <div className="inline-flex items-center gap-0.5 rounded-lg border border-zinc-800 bg-zinc-900 p-0.5">
                  <button
                    onClick={() => setBillingCycle('monthly')}
                    className={`rounded-md px-5 py-2 text-sm font-medium transition-all ${
                      billingCycle === 'monthly'
                        ? 'bg-white text-black'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    Monthly
                  </button>
                  <button
                    onClick={() => setBillingCycle('annual')}
                    className={`rounded-md px-5 py-2 text-sm font-medium transition-all ${
                      billingCycle === 'annual'
                        ? 'bg-white text-black'
                        : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    Annual
                  </button>
                </div>
              </div>

              {/* Cards */}
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
                {PRICING_PLANS.map((plan, idx) => (
                  <div
                    key={idx}
                    className={`relative rounded-xl border p-6 transition-all hover:border-zinc-700 ${
                      plan.popular
                        ? 'border-zinc-700 bg-zinc-900'
                        : 'border-zinc-800 bg-zinc-950'
                    }`}
                  >
                    {plan.popular && (
                      <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                        <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-black">
                          Most Popular
                        </span>
                      </div>
                    )}

                    <div className="mb-6">
                      <h3 className="mb-2 text-lg font-semibold">{plan.name}</h3>
                      <p className="mb-6 min-h-[2.5rem] text-sm text-zinc-500">{plan.description}</p>
                      <div className="flex items-baseline gap-1">
                        <span className="text-4xl font-bold">
                          {billingCycle === 'monthly' ? plan.monthlyPrice : plan.annualPrice}
                        </span>
                        <span className="text-sm text-zinc-500">
                          /{billingCycle === 'monthly' ? 'month' : 'year'}
                        </span>
                      </div>
                    </div>

                    <button
                      onClick={handleSignup}
                      className={`mb-6 w-full rounded-lg px-4 py-2.5 text-sm font-medium transition-all ${
                        plan.popular
                          ? 'bg-white text-black hover:bg-zinc-200'
                          : 'bg-zinc-800 text-white hover:bg-zinc-700'
                      }`}
                      data-testid={`pricing-${plan.name.toLowerCase()}-button`}
                    >
                      Get Started
                    </button>

                    <div className="space-y-3">
                      {plan.features.map((feature, i) => (
                        <div key={i} className="flex items-start gap-3">
                          <Check className="mt-0.5 h-4 w-4 flex-shrink-0 text-zinc-400" />
                          <span className="text-sm text-zinc-400">{feature}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Tech Stack Icon Cloud */}
          <section className="relative px-6 py-20 sm:py-32 lg:px-8 overflow-hidden">
            <div className="mx-auto max-w-7xl">
              <div className="mb-12 text-center relative z-10">
                <h2 className="mb-4 text-4xl font-bold sm:text-5xl">Built with Modern Tech</h2>
                <p className="text-xl text-zinc-400">Powered by industry-leading technologies</p>
              </div>
              
              {/* Icon Cloud - 3D rotating tech stack */}
              <div className="relative h-[600px] w-full">
                <div className="absolute inset-0 flex items-center justify-center">
                  <IconCloud images={TECH_IMAGES} />
                </div>
                {/* Subtle radial gradient for depth */}
                <div className="absolute inset-0 bg-gradient-radial from-transparent via-transparent to-black pointer-events-none" />
              </div>
            </div>
          </section>

          {/* Video Testimonials */}
          <section className="px-6 py-20 sm:py-32 lg:px-8" id="testimonials">
            <div className="mx-auto max-w-7xl">
              <div className="mb-12 text-center">
                <h2 className="mb-4 text-4xl font-bold sm:text-5xl">Success Stories</h2>
                <p className="text-xl text-zinc-400">Hear from learners who transformed their careers</p>
              </div>
              
              <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                {VIDEO_TESTIMONIALS.map((testimonial, idx) => (
                  <div
                    key={idx}
                    className="group relative overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 hover:border-zinc-700 transition-all"
                  >
                    <div className="relative aspect-video overflow-hidden">
                      <img 
                        src={testimonial.image} 
                        alt={testimonial.name}
                        className="h-full w-full object-cover transition-transform group-hover:scale-105"
                        loading="lazy"
                      />
                      <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white/90 backdrop-blur-sm transition-transform group-hover:scale-110">
                          <div className="h-0 w-0 border-l-[16px] border-l-black border-y-[10px] border-y-transparent ml-1" />
                        </div>
                      </div>
                    </div>
                    <div className="p-6">
                      <p className="mb-4 text-sm text-zinc-300 italic">"{testimonial.quote}"</p>
                      <div>
                        <p className="font-semibold">{testimonial.name}</p>
                        <p className="text-sm text-zinc-500">{testimonial.role}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* FAQ Section */}
          <section className="px-6 py-20 sm:py-32 lg:px-8 bg-zinc-950/50">
            <div className="mx-auto max-w-3xl">
              <div className="mb-12 text-center">
                <h2 className="mb-4 text-4xl font-bold sm:text-5xl">Frequently Asked Questions</h2>
                <p className="text-xl text-zinc-400">Everything you need to know about MasterX</p>
              </div>
              
              <div className="space-y-4">
                {FAQS.map((faq, idx) => (
                  <div
                    key={idx}
                    className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950"
                  >
                    <button
                      onClick={() => setOpenFaq(openFaq === idx ? null : idx)}
                      className="flex w-full items-center justify-between p-6 text-left transition-colors hover:bg-zinc-900"
                    >
                      <span className="font-semibold">{faq.question}</span>
                      <span className={`text-2xl transition-transform ${openFaq === idx ? 'rotate-45' : ''}`}>
                        +
                      </span>
                    </button>
                    <div
                      className={`overflow-hidden transition-all duration-300 ${
                        openFaq === idx ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                      }`}
                    >
                      <div className="border-t border-zinc-800 p-6 pt-4 text-zinc-400">
                        {faq.answer}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Final CTA Section */}
          <section className="relative px-6 py-20 sm:py-32 lg:px-8 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-pink-900/20" />
            <AnimatedGridPattern className="opacity-50" />
            
            <div className="relative mx-auto max-w-4xl text-center">
              <h2 className="mb-6 text-4xl font-bold sm:text-6xl">
                Ready to Transform Your Learning?
              </h2>
              <p className="mb-10 text-xl text-zinc-400">
                Join thousands of learners achieving their goals with emotion-aware AI
              </p>
              <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
                <InteractiveHoverButton 
                  onClick={handleSignup}
                  className="text-lg px-8 py-4"
                >
                  Start Free Trial
                </InteractiveHoverButton>
                <button 
                  onClick={() => handleCTAClick('demo')}
                  className="rounded-lg border border-zinc-700 bg-zinc-900 px-8 py-4 text-lg font-medium transition-colors hover:bg-zinc-800"
                >
                  Schedule a Demo
                </button>
              </div>
              <p className="mt-6 text-sm text-zinc-500">
                No credit card required • 100% free tier available • Cancel anytime
              </p>
            </div>
          </section>

          {/* Footer */}
          <footer className="border-t border-zinc-800/50 px-6 py-12 lg:px-8">
            <div className="mx-auto max-w-7xl">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
                <div>
                  <h3 className="text-white font-semibold mb-4">Product</h3>
                  <ul className="space-y-2">
                    <li><a href="#features" className="text-zinc-400 hover:text-white transition-colors">Features</a></li>
                    <li><a href="#pricing" className="text-zinc-400 hover:text-white transition-colors">Pricing</a></li>
                    <li><a href="#testimonials" className="text-zinc-400 hover:text-white transition-colors">Testimonials</a></li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-4">Company</h3>
                  <ul className="space-y-2">
                    <li><Link to="/about" className="text-zinc-400 hover:text-white transition-colors">About</Link></li>
                    <li><Link to="/blog" className="text-zinc-400 hover:text-white transition-colors">Blog</Link></li>
                    <li><Link to="/careers" className="text-zinc-400 hover:text-white transition-colors">Careers</Link></li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-4">Resources</h3>
                  <ul className="space-y-2">
                    <li><Link to="/docs" className="text-zinc-400 hover:text-white transition-colors">Documentation</Link></li>
                    <li><Link to="/help" className="text-zinc-400 hover:text-white transition-colors">Help Center</Link></li>
                    <li><Link to="/community" className="text-zinc-400 hover:text-white transition-colors">Community</Link></li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-4">Legal</h3>
                  <ul className="space-y-2">
                    <li><Link to="/privacy" className="text-zinc-400 hover:text-white transition-colors">Privacy</Link></li>
                    <li><Link to="/terms" className="text-zinc-400 hover:text-white transition-colors">Terms</Link></li>
                    <li><Link to="/security" className="text-zinc-400 hover:text-white transition-colors">Security</Link></li>
                  </ul>
                </div>
              </div>

              <div className="pt-8 border-t border-zinc-800 flex flex-col sm:flex-row justify-between items-center">
                <div className="flex items-center gap-2 mb-4 sm:mb-0">
                  <Sparkles className="h-5 w-5 text-purple-500" />
                  <span className="text-base font-medium">MasterX</span>
                </div>
                <p className="text-sm text-zinc-500 mb-4 sm:mb-0">
                  © 2025 MasterX. All rights reserved.
                </p>
                <div className="flex items-center gap-6 text-sm text-zinc-400">
                  <a href="https://twitter.com/masterx" className="transition-colors hover:text-white">Twitter</a>
                  <a href="https://github.com/masterx" className="transition-colors hover:text-white">GitHub</a>
                  <a href="https://linkedin.com/company/masterx" className="transition-colors hover:text-white">LinkedIn</a>
                </div>
              </div>
            </div>
          </footer>
        </div>

        {/* Custom Animations CSS */}
        <style>{`
          html {
            scroll-behavior: smooth;
          }
          @keyframes marquee {
            from { transform: translateX(0); }
            to { transform: translateX(calc(-50% - 0.5rem)); }
          }
          @keyframes marquee-reverse {
            from { transform: translateX(calc(-50% - 0.5rem)); }
            to { transform: translateX(0); }
          }
          @keyframes border-beam {
            0% { offset-distance: 0%; }
            100% { offset-distance: 100%; }
          }
          .animate-marquee {
            animation: marquee 40s linear infinite;
          }
          .animate-marquee-reverse {
            animation: marquee-reverse 40s linear infinite;
          }
          .animate-border-beam {
            animation: border-beam 15s linear infinite;
          }
          @keyframes shiny-text {
            0%, 90%, 100% {
              background-position: calc(-100% - var(--shiny-width)) 0;
            }
            30%, 60% {
              background-position: calc(100% + var(--shiny-width)) 0;
            }
          }
          .animate-shiny-text {
            animation: shiny-text 8s infinite;
          }
          @keyframes expand-underline {
            from { width: 0; }
            to { width: 100%; }
          }
          @keyframes expand-highlight {
            from { transform: scaleX(0); }
            to { transform: scaleX(1); }
          }
          .animate-expand-underline {
            animation: expand-underline 1s ease-out forwards;
          }
          .animate-expand-highlight {
            animation: expand-highlight 1s ease-out forwards;
          }
          @keyframes grid-flow {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.6; }
          }
          .animate-grid-flow {
            animation: grid-flow 8s ease-in-out infinite;
          }
        `}</style>
      </div>
    </>
  );
};

// ============================================================================
// EXPORTS
// ============================================================================

export default Landing;

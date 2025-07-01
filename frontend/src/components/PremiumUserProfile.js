import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GlassCard, GlassButton } from './GlassCard';
import { 
  UserIcon, 
  SettingsIcon, 
  TrophyIcon, 
  BarChartIcon,
  ZapIcon,
  ChevronRightIcon,
  SparkleIcon,
  CrownIcon
} from './PremiumIcons';
import { cn } from '../utils/cn';

// Additional icons for user profile

'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Monitor, 
  Sun, 
  Moon, 
  Palette, 
  Download, 
  Upload, 
  User, 
  Bell, 
  Shield, 
  Database,
  Check,
  ChevronDown,
  Brain,
  Sparkles,
  Zap,
  Settings,
  Volume2,
  VolumeX,
  Globe,
  Key,
  Trash2,
  Archive,
  RefreshCw
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface MasterXSettingsPanelProps {
  isOpen: boolean
  onClose: () => void
}

const settingSections = [
  { id: 'general', label: 'General', icon: Settings, description: 'Basic preferences' },
  { id: 'appearance', label: 'Appearance', icon: Palette, description: 'Visual customization' },
  { id: 'intelligence', label: 'AI Intelligence', icon: Brain, description: 'AI behavior settings' },
  { id: 'notifications', label: 'Notifications', icon: Bell, description: 'Alert preferences' },
  { id: 'data', label: 'Data & Privacy', icon: Database, description: 'Data management' },
  { id: 'account', label: 'Account', icon: User, description: 'Profile settings' },
  { id: 'security', label: 'Security', icon: Shield, description: 'Security options' },
]

const themeOptions = [
  { value: 'quantum', label: 'Quantum Dark', preview: 'from-purple-900 to-cyan-900' },
  { value: 'neural', label: 'Neural Blue', preview: 'from-blue-900 to-indigo-900' },
  { value: 'matrix', label: 'Matrix Green', preview: 'from-green-900 to-emerald-900' },
  { value: 'solar', label: 'Solar Gold', preview: 'from-yellow-900 to-orange-900' },
]

const accentColors = [
  { value: 'purple', label: 'Quantum Purple', color: '#a855f7' },
  { value: 'cyan', label: 'Cyber Cyan', color: '#06b6d4' },
  { value: 'emerald', label: 'Neural Emerald', color: '#10b981' },
  { value: 'amber', label: 'Intelligence Amber', color: '#f59e0b' },
  { value: 'rose', label: 'Plasma Rose', color: '#f43f5e' },
  { value: 'blue', label: 'Photon Blue', color: '#3b82f6' },
]

export function MasterXSettingsPanel({ isOpen, onClose }: MasterXSettingsPanelProps) {
  const [activeSection, setActiveSection] = useState('general')
  const [settings, setSettings] = useState({
    theme: 'quantum',
    accentColor: 'purple',
    language: 'en',
    voiceEnabled: true,
    notifications: true,
    soundEnabled: true,
    autoSave: true,
    dataCollection: false,
    aiProvider: 'auto',
    responseStyle: 'balanced'
  })

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }

  const exportData = () => {
    const data = {
      settings,
      exportDate: new Date().toISOString(),
      version: '3.0'
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `masterx-settings-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const importData = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string)
        if (data.settings) {
          setSettings(data.settings)
        }
      } catch (error) {
        console.error('Failed to import settings:', error)
      }
    }
    reader.readAsText(file)
  }

  const renderSectionContent = () => {
    switch (activeSection) {
      case 'general':
        return <GeneralSettings settings={settings} updateSetting={updateSetting} />
      case 'appearance':
        return <AppearanceSettings settings={settings} updateSetting={updateSetting} themeOptions={themeOptions} accentColors={accentColors} />
      case 'intelligence':
        return <IntelligenceSettings settings={settings} updateSetting={updateSetting} />
      case 'notifications':
        return <NotificationSettings settings={settings} updateSetting={updateSetting} />
      case 'data':
        return <DataSettings onExport={exportData} onImport={importData} settings={settings} updateSetting={updateSetting} />
      case 'account':
        return <AccountSettings />
      case 'security':
        return <SecuritySettings />
      default:
        return <GeneralSettings settings={settings} updateSetting={updateSetting} />
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm"
            style={{ zIndex: 'var(--z-modal-backdrop)' }}
            onClick={onClose}
          />

          {/* Settings Panel */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            className="fixed inset-4 md:inset-8 lg:inset-16 glass-morph-premium rounded-3xl shadow-2xl overflow-hidden border border-purple-500/30"
            style={{ zIndex: 'var(--z-modal)' }}
          >
            <div className="flex h-full">
              {/* Sidebar */}
              <div className="w-80 border-r border-purple-500/20 flex flex-col">
                <div className="p-quantum-6 border-b border-purple-500/20">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-quantum-3">
                      <motion.div
                        className="w-quantum-10 h-quantum-10 rounded-xl bg-gradient-to-r from-purple-600 to-cyan-600 flex items-center justify-center"
                        whileHover={{ scale: 1.05 }}
                      >
                        <Settings className="h-5 w-5 text-white" />
                      </motion.div>
                      <div>
                        <h2 className="text-quantum-xl font-bold text-plasma-white">Quantum Settings</h2>
                        <p className="text-quantum-sm text-plasma-white/60">Configure MasterX</p>
                      </div>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={onClose}
                      className="p-quantum-2 glass-morph rounded-lg hover:bg-red-500/20 transition-all duration-200"
                    >
                      <X className="h-5 w-5 text-plasma-white/70" />
                    </motion.button>
                  </div>
                </div>

                <nav className="flex-1 p-quantum-4 space-y-quantum-2 overflow-y-auto">
                  {settingSections.map((section) => (
                    <motion.button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      whileHover={{ x: 4 }}
                      className={cn(
                        "w-full flex items-center space-x-quantum-3 p-quantum-3 rounded-xl text-left transition-all duration-200",
                        activeSection === section.id
                          ? "glass-morph-premium border border-purple-500/30 text-plasma-white"
                          : "hover:glass-morph text-plasma-white/70 hover:text-plasma-white"
                      )}
                    >
                      <section.icon className={cn(
                        "h-5 w-5",
                        activeSection === section.id ? "text-cyan-400" : "text-plasma-white/50"
                      )} />
                      <div className="flex-1">
                        <div className="text-quantum-sm font-medium">{section.label}</div>
                        <div className="text-quantum-xs text-plasma-white/40">{section.description}</div>
                      </div>
                      {activeSection === section.id && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="w-quantum-2 h-quantum-2 bg-cyan-400 rounded-full"
                        />
                      )}
                    </motion.button>
                  ))}
                </nav>

                <div className="p-4 border-t border-purple-500/20">
                  <div className="text-xs text-plasma-white/40 text-center">
                    MasterX Quantum v3.0
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto">
                <div className="p-8">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={activeSection}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3 }}
                    >
                      {renderSectionContent()}
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// General Settings Component
function GeneralSettings({ settings, updateSetting }: { settings: any, updateSetting: (key: string, value: any) => void }) {
  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-plasma-white mb-2">General Preferences</h3>
        <p className="text-plasma-white/60">Configure basic MasterX settings</p>
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-plasma-white/80 mb-3">Language</label>
          <select
            value={settings.language}
            onChange={(e) => updateSetting('language', e.target.value)}
            className="w-full p-3 glass-morph rounded-xl text-plasma-white bg-transparent border border-purple-500/20 focus:border-cyan-400 focus:outline-none transition-all"
          >
            <option value="en">English (US)</option>
            <option value="en-gb">English (UK)</option>
            <option value="es">Español</option>
            <option value="fr">Français</option>
            <option value="de">Deutsch</option>
            <option value="ja">日本語</option>
            <option value="zh">中文</option>
          </select>
        </div>

        <div className="flex items-center justify-between p-4 glass-morph rounded-xl">
          <div>
            <div className="font-medium text-plasma-white">Voice Input</div>
            <div className="text-sm text-plasma-white/60">Enable voice commands and dictation</div>
          </div>
          <ToggleSwitch
            enabled={settings.voiceEnabled}
            onChange={(enabled) => updateSetting('voiceEnabled', enabled)}
          />
        </div>

        <div className="flex items-center justify-between p-4 glass-morph rounded-xl">
          <div>
            <div className="font-medium text-plasma-white">Auto-save Conversations</div>
            <div className="text-sm text-plasma-white/60">Automatically save chat history</div>
          </div>
          <ToggleSwitch
            enabled={settings.autoSave}
            onChange={(enabled) => updateSetting('autoSave', enabled)}
          />
        </div>
      </div>
    </div>
  )
}

// Appearance Settings Component
function AppearanceSettings({ settings, updateSetting, themeOptions, accentColors }: any) {
  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-plasma-white mb-2">Appearance</h3>
        <p className="text-plasma-white/60">Customize the visual experience</p>
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-plasma-white/80 mb-4">Theme</label>
          <div className="grid grid-cols-2 gap-3">
            {themeOptions.map((theme: any) => (
              <motion.button
                key={theme.value}
                onClick={() => updateSetting('theme', theme.value)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={cn(
                  "p-4 rounded-xl border transition-all duration-200",
                  settings.theme === theme.value
                    ? "border-cyan-400 glass-morph-premium"
                    : "border-purple-500/20 glass-morph hover:border-purple-500/40"
                )}
              >
                <div className={`w-full h-8 bg-gradient-to-r ${theme.preview} rounded-lg mb-3`} />
                <div className="text-sm font-medium text-plasma-white">{theme.label}</div>
                {settings.theme === theme.value && (
                  <Check className="h-4 w-4 text-cyan-400 mx-auto mt-2" />
                )}
              </motion.button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-plasma-white/80 mb-4">Accent Color</label>
          <div className="grid grid-cols-3 gap-3">
            {accentColors.map((color: any) => (
              <motion.button
                key={color.value}
                onClick={() => updateSetting('accentColor', color.value)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={cn(
                  "flex items-center space-x-3 p-3 rounded-xl border transition-all duration-200",
                  settings.accentColor === color.value
                    ? "border-cyan-400 glass-morph-premium"
                    : "border-purple-500/20 glass-morph hover:border-purple-500/40"
                )}
              >
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: color.color }}
                />
                <span className="text-sm text-plasma-white">{color.label}</span>
                {settings.accentColor === color.value && (
                  <Check className="h-4 w-4 text-cyan-400 ml-auto" />
                )}
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// Intelligence Settings Component
function IntelligenceSettings({ settings, updateSetting }: any) {
  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-plasma-white mb-2">AI Intelligence</h3>
        <p className="text-plasma-white/60">Configure AI behavior and preferences</p>
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-plasma-white/80 mb-3">AI Provider</label>
          <select
            value={settings.aiProvider}
            onChange={(e) => updateSetting('aiProvider', e.target.value)}
            className="w-full p-3 glass-morph rounded-xl text-plasma-white bg-transparent border border-purple-500/20 focus:border-cyan-400 focus:outline-none transition-all"
          >
            <option value="auto">Auto-select Best</option>
            <option value="groq">Groq (Fast)</option>
            <option value="gemini">Gemini (Balanced)</option>
            <option value="gpt">GPT (Creative)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-plasma-white/80 mb-3">Response Style</label>
          <div className="grid grid-cols-3 gap-3">
            {['concise', 'balanced', 'detailed'].map((style) => (
              <motion.button
                key={style}
                onClick={() => updateSetting('responseStyle', style)}
                whileHover={{ scale: 1.02 }}
                className={cn(
                  "p-3 rounded-xl border transition-all duration-200 capitalize",
                  settings.responseStyle === style
                    ? "border-cyan-400 glass-morph-premium text-cyan-400"
                    : "border-purple-500/20 glass-morph text-plasma-white/70 hover:text-plasma-white"
                )}
              >
                {style}
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// Notification Settings Component
function NotificationSettings({ settings, updateSetting }: any) {
  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-plasma-white mb-2">Notifications</h3>
        <p className="text-plasma-white/60">Control how you receive alerts</p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between p-4 glass-morph rounded-xl">
          <div>
            <div className="font-medium text-plasma-white">Desktop Notifications</div>
            <div className="text-sm text-plasma-white/60">Show notifications when AI responds</div>
          </div>
          <ToggleSwitch
            enabled={settings.notifications}
            onChange={(enabled) => updateSetting('notifications', enabled)}
          />
        </div>

        <div className="flex items-center justify-between p-4 glass-morph rounded-xl">
          <div>
            <div className="font-medium text-plasma-white">Sound Effects</div>
            <div className="text-sm text-plasma-white/60">Play sounds for interactions</div>
          </div>
          <ToggleSwitch
            enabled={settings.soundEnabled}
            onChange={(enabled) => updateSetting('soundEnabled', enabled)}
          />
        </div>
      </div>
    </div>
  )
}

// Data Settings Component
function DataSettings({ onExport, onImport, settings, updateSetting }: any) {
  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-plasma-white mb-2">Data & Privacy</h3>
        <p className="text-plasma-white/60">Manage your data and privacy settings</p>
      </div>

      <div className="space-y-6">
        <div className="p-6 glass-morph rounded-xl">
          <h4 className="font-semibold text-plasma-white mb-4">Export & Import</h4>
          <div className="flex space-x-3">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onExport}
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg text-white hover:from-purple-700 hover:to-cyan-700 transition-all"
            >
              <Download className="h-4 w-4" />
              <span>Export Settings</span>
            </motion.button>

            <label className="flex items-center space-x-2 px-4 py-2 glass-morph rounded-lg hover:bg-purple-500/20 transition-all cursor-pointer">
              <Upload className="h-4 w-4 text-plasma-white/70" />
              <span className="text-plasma-white/70">Import Settings</span>
              <input
                type="file"
                accept=".json"
                onChange={onImport}
                className="hidden"
              />
            </label>
          </div>
        </div>

        <div className="flex items-center justify-between p-4 glass-morph rounded-xl">
          <div>
            <div className="font-medium text-plasma-white">Data Collection</div>
            <div className="text-sm text-plasma-white/60">Help improve MasterX with usage data</div>
          </div>
          <ToggleSwitch
            enabled={settings.dataCollection}
            onChange={(enabled) => updateSetting('dataCollection', enabled)}
          />
        </div>
      </div>
    </div>
  )
}

// Account Settings Component
function AccountSettings() {
  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-plasma-white mb-2">Account</h3>
        <p className="text-plasma-white/60">Manage your account information</p>
      </div>

      <div className="space-y-6">
        <div className="p-6 glass-morph rounded-xl">
          <div className="flex items-center space-x-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-full flex items-center justify-center">
              <User className="h-8 w-8 text-white" />
            </div>
            <div>
              <h4 className="text-lg font-semibold text-plasma-white">Quantum User</h4>
              <p className="text-plasma-white/60">Premium Account</p>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-plasma-white/80 mb-2">Display Name</label>
              <input
                type="text"
                placeholder="Your name"
                className="w-full p-3 glass-morph rounded-xl text-plasma-white bg-transparent border border-purple-500/20 focus:border-cyan-400 focus:outline-none transition-all"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-plasma-white/80 mb-2">Email</label>
              <input
                type="email"
                placeholder="your.email@example.com"
                className="w-full p-3 glass-morph rounded-xl text-plasma-white bg-transparent border border-purple-500/20 focus:border-cyan-400 focus:outline-none transition-all"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Security Settings Component
function SecuritySettings() {
  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-2xl font-bold text-plasma-white mb-2">Security</h3>
        <p className="text-plasma-white/60">Keep your account secure</p>
      </div>

      <div className="space-y-4">
        <motion.button
          whileHover={{ scale: 1.01 }}
          className="w-full p-4 glass-morph rounded-xl text-left hover:bg-purple-500/10 transition-all"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-plasma-white">Two-Factor Authentication</div>
              <div className="text-sm text-plasma-white/60">Add extra security to your account</div>
            </div>
            <ChevronDown className="h-5 w-5 text-plasma-white/40" />
          </div>
        </motion.button>

        <motion.button
          whileHover={{ scale: 1.01 }}
          className="w-full p-4 glass-morph rounded-xl text-left hover:bg-purple-500/10 transition-all"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-plasma-white">Change Password</div>
              <div className="text-sm text-plasma-white/60">Update your account password</div>
            </div>
            <Key className="h-5 w-5 text-plasma-white/40" />
          </div>
        </motion.button>

        <motion.button
          whileHover={{ scale: 1.01 }}
          className="w-full p-4 glass-morph rounded-xl text-left hover:bg-purple-500/10 transition-all"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-plasma-white">Active Sessions</div>
              <div className="text-sm text-plasma-white/60">Manage your login sessions</div>
            </div>
            <Monitor className="h-5 w-5 text-plasma-white/40" />
          </div>
        </motion.button>
      </div>
    </div>
  )
}

// Toggle Switch Component
function ToggleSwitch({ enabled, onChange }: { enabled: boolean, onChange: (enabled: boolean) => void }) {
  return (
    <motion.button
      onClick={() => onChange(!enabled)}
      className={cn(
        "relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200",
        enabled ? "bg-gradient-to-r from-purple-600 to-cyan-600" : "bg-plasma-white/20"
      )}
      whileTap={{ scale: 0.95 }}
    >
      <motion.span
        className="inline-block h-4 w-4 transform rounded-full bg-white shadow-lg transition-transform duration-200"
        animate={{ x: enabled ? 24 : 4 }}
      />
    </motion.button>
  )
}

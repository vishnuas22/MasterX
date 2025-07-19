'use client'

import { useState, useRef, useEffect } from 'react'
import { ChevronDown, Check, Zap, Brain, Code, Palette, Eye, Target } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DropdownOption {
  value: string
  label: string
  description?: string
  icon?: React.ComponentType<any>
  badge?: string
  color?: string
}

interface QuantumDropdownProps {
  options: DropdownOption[]
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  disabled?: boolean
  label?: string
}

export function QuantumDropdown({
  options,
  value,
  onChange,
  placeholder = 'Select option...',
  className = '',
  disabled = false,
  label
}: QuantumDropdownProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [highlightedIndex, setHighlightedIndex] = useState(-1)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const selectedOption = options.find(option => option.value === value)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
        setHighlightedIndex(-1)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    if (!isOpen) {
      setHighlightedIndex(-1)
    }
  }, [isOpen])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return

    switch (e.key) {
      case 'Enter':
      case ' ':
        e.preventDefault()
        if (isOpen && highlightedIndex >= 0) {
          onChange(options[highlightedIndex].value)
          setIsOpen(false)
        } else {
          setIsOpen(!isOpen)
        }
        break
      case 'ArrowDown':
        e.preventDefault()
        if (!isOpen) {
          setIsOpen(true)
        } else {
          setHighlightedIndex(prev => 
            prev < options.length - 1 ? prev + 1 : 0
          )
        }
        break
      case 'ArrowUp':
        e.preventDefault()
        if (isOpen) {
          setHighlightedIndex(prev => 
            prev > 0 ? prev - 1 : options.length - 1
          )
        }
        break
      case 'Escape':
        setIsOpen(false)
        setHighlightedIndex(-1)
        break
    }
  }

  const handleOptionClick = (optionValue: string) => {
    onChange(optionValue)
    setIsOpen(false)
    setHighlightedIndex(-1)
  }

  return (
    <div className={cn('relative', className)}>
      {label && (
        <label className="block text-sm font-medium text-purple-300 mb-2">
          {label}
        </label>
      )}
      
      <div
        ref={dropdownRef}
        className={cn(
          'relative w-full',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        {/* Trigger Button */}
        <button
          type="button"
          onClick={() => !disabled && setIsOpen(!isOpen)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          className={cn(
            'w-full flex items-center justify-between px-4 py-3 text-left',
            'glass-morph border border-purple-500/30 rounded-lg',
            'hover:border-purple-400/50 focus:border-purple-400 focus:outline-none',
            'transition-all duration-200 group',
            'text-white bg-slate-800/50',
            isOpen && 'border-purple-400 ring-2 ring-purple-400/20',
            disabled && 'cursor-not-allowed'
          )}
        >
          <div className="flex items-center space-x-3 flex-1 min-w-0">
            {selectedOption ? (
              <>
                {selectedOption.icon && (
                  <selectedOption.icon className={cn(
                    'h-4 w-4 flex-shrink-0',
                    selectedOption.color || 'text-purple-400'
                  )} />
                )}
                <div className="flex-1 min-w-0">
                  <span className="font-medium text-white truncate">
                    {selectedOption.label}
                  </span>
                  {selectedOption.description && (
                    <p className="text-xs text-purple-300 truncate">
                      {selectedOption.description}
                    </p>
                  )}
                </div>
                {selectedOption.badge && (
                  <span className="px-2 py-1 text-xs bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
                    {selectedOption.badge}
                  </span>
                )}
              </>
            ) : (
              <span className="text-gray-400">{placeholder}</span>
            )}
          </div>
          
          <ChevronDown className={cn(
            'h-4 w-4 text-purple-300 transition-transform duration-200 flex-shrink-0 ml-2',
            isOpen && 'rotate-180'
          )} />
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <div className="absolute top-full left-0 right-0 mt-2 z-50">
            <div className="glass-morph border border-purple-500/30 rounded-lg shadow-xl overflow-hidden">
              <div className="max-h-64 overflow-y-auto">
                {options.map((option, index) => {
                  const Icon = option.icon
                  const isSelected = option.value === value
                  const isHighlighted = index === highlightedIndex

                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => handleOptionClick(option.value)}
                      className={cn(
                        'w-full flex items-center space-x-3 px-4 py-3 text-left',
                        'hover:bg-purple-500/20 transition-all duration-150',
                        'border-b border-purple-500/10 last:border-b-0',
                        isSelected && 'bg-purple-500/20 border-purple-500/30',
                        isHighlighted && 'bg-purple-500/10'
                      )}
                    >
                      {Icon && (
                        <Icon className={cn(
                          'h-4 w-4 flex-shrink-0',
                          option.color || 'text-purple-400'
                        )} />
                      )}
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <span className={cn(
                            'font-medium truncate',
                            isSelected ? 'text-white' : 'text-purple-100'
                          )}>
                            {option.label}
                          </span>
                          {option.badge && (
                            <span className="px-2 py-1 text-xs bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
                              {option.badge}
                            </span>
                          )}
                        </div>
                        {option.description && (
                          <p className="text-xs text-purple-300 truncate mt-1">
                            {option.description}
                          </p>
                        )}
                      </div>

                      {isSelected && (
                        <Check className="h-4 w-4 text-purple-400 flex-shrink-0" />
                      )}
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Predefined option sets for common use cases
export const TASK_TYPE_OPTIONS: DropdownOption[] = [
  {
    value: 'general',
    label: 'General',
    description: 'Balanced AI assistance for everyday tasks',
    icon: Brain,
    color: 'text-purple-400'
  },
  {
    value: 'reasoning',
    label: 'Reasoning',
    description: 'Deep analysis and logical problem solving',
    icon: Target,
    color: 'text-cyan-400'
  },
  {
    value: 'coding',
    label: 'Coding',
    description: 'Programming help and code generation',
    icon: Code,
    color: 'text-green-400'
  },
  {
    value: 'creative',
    label: 'Creative',
    description: 'Creative writing and artistic tasks',
    icon: Palette,
    color: 'text-pink-400'
  },
  {
    value: 'fast',
    label: 'Fast',
    description: 'Quick responses for simple queries',
    icon: Zap,
    color: 'text-yellow-400'
  },
  {
    value: 'multimodal',
    label: 'Multimodal',
    description: 'Advanced vision and multimodal AI',
    icon: Eye,
    color: 'text-amber-400'
  }
]

export const PROVIDER_OPTIONS: DropdownOption[] = [
  {
    value: '',
    label: 'Auto-Select',
    description: 'Intelligent model selection based on task',
    icon: Brain,
    color: 'text-purple-400',
    badge: 'Smart'
  },
  {
    value: 'groq',
    label: 'Groq',
    description: 'Ultra-fast inference with Llama models',
    icon: Zap,
    color: 'text-orange-400',
    badge: 'Fast'
  },
  {
    value: 'gemini',
    label: 'Gemini',
    description: 'Google\'s advanced reasoning AI',
    icon: Target,
    color: 'text-blue-400',
    badge: 'Smart'
  },
  {
    value: 'openai',
    label: 'OpenAI',
    description: 'GPT-4 for premium AI capabilities',
    icon: Brain,
    color: 'text-green-400',
    badge: 'Premium'
  },
  {
    value: 'anthropic',
    label: 'Claude',
    description: 'Anthropic\'s helpful and harmless AI',
    icon: Code,
    color: 'text-purple-400',
    badge: 'Coding'
  }
]

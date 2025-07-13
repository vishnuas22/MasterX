'use client'

import { useState, useEffect } from 'react'
import { Brain, MessageSquare, Zap, Target, BookOpen, Users } from 'lucide-react'
import { ConnectivityCheck } from '@/components/connectivity-check'
import { ChatInterface } from '@/components/chat-interface'
import { LearningDashboard } from '@/components/learning-dashboard'

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'home' | 'chat' | 'dashboard'>('home')

  return (
    <main className="min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-black/20 backdrop-blur-lg border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-purple-400 quantum-pulse" />
              <span className="ml-2 text-xl font-bold quantum-text-glow">MasterX</span>
            </div>
            <div className="flex space-x-4">
              <button
                onClick={() => setActiveTab('home')}
                className={`px-4 py-2 rounded-lg transition-all ${
                  activeTab === 'home'
                    ? 'bg-purple-600 text-white quantum-glow'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                Home
              </button>
              <button
                onClick={() => setActiveTab('chat')}
                className={`px-4 py-2 rounded-lg transition-all ${
                  activeTab === 'chat'
                    ? 'bg-purple-600 text-white quantum-glow'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                AI Chat
              </button>
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-lg transition-all ${
                  activeTab === 'dashboard'
                    ? 'bg-purple-600 text-white quantum-glow'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                Dashboard
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className="pt-16">
        {activeTab === 'home' && <HomeContent />}
        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'dashboard' && <LearningDashboard />}
      </div>

      {/* Connectivity Check */}
      <ConnectivityCheck />
    </main>
  )
}

function HomeContent() {
  return (
    <div className="relative">
      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-6xl font-bold mb-6 quantum-text-glow">
            Revolutionary AI Learning Platform
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Experience the future of education with our Quantum Intelligence Engine. 
            Adaptive learning, real-time mentorship, and personalized growth paths.
          </p>
          <div className="flex justify-center space-x-4">
            <button className="bg-purple-600 hover:bg-purple-700 text-white px-8 py-3 rounded-lg font-semibold quantum-glow transition-all">
              Start Learning
            </button>
            <button className="border border-purple-500 text-purple-400 hover:bg-purple-500 hover:text-white px-8 py-3 rounded-lg font-semibold transition-all">
              Learn More
            </button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-black/20">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12 quantum-text-glow">
            Quantum-Powered Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="bg-slate-800/50 p-6 rounded-xl border border-purple-500/20 hover:border-purple-500/50 transition-all quantum-glow"
              >
                <feature.icon className="h-12 w-12 text-purple-400 mb-4" />
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-300">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quantum Engine Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-6 quantum-text-glow">
            Quantum Intelligence Engine
          </h2>
          <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Our revolutionary 27,000+ line AI system combines multiple learning modes, 
            emotional intelligence, and real-time adaptation for an unparalleled learning experience.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {quantumFeatures.map((feature, index) => (
              <div
                key={index}
                className="bg-gradient-to-b from-purple-900/50 to-slate-900/50 p-6 rounded-xl border border-purple-500/30"
              >
                <h3 className="text-lg font-semibold mb-2 text-purple-300">{feature.title}</h3>
                <p className="text-sm text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

const features = [
  {
    icon: MessageSquare,
    title: 'AI Mentorship',
    description: 'Real-time conversations with advanced AI mentors that adapt to your learning style and emotional state.'
  },
  {
    icon: Brain,
    title: 'Quantum Intelligence',
    description: 'Our revolutionary AI engine with 27,000+ lines of sophisticated learning algorithms and neural architectures.'
  },
  {
    icon: Zap,
    title: 'Real-time Adaptation',
    description: 'Dynamic difficulty adjustment and personalized learning paths that evolve with your progress.'
  },
  {
    icon: Target,
    title: 'Precision Learning',
    description: 'Metacognitive training, memory palaces, and advanced psychology-based learning techniques.'
  },
  {
    icon: BookOpen,
    title: 'Multimodal AI',
    description: 'Text, voice, image, and video processing for rich, interactive learning experiences.'
  },
  {
    icon: Users,
    title: 'Collaborative Learning',
    description: 'Study groups, peer mentorship, and social learning features with AI facilitation.'
  }
]

const quantumFeatures = [
  {
    title: 'Learning Modes',
    description: 'Socratic, Debug, Challenge, Mentor, Creative, and Analytical learning approaches'
  },
  {
    title: 'Emotional AI',
    description: 'Mood-based adaptation and emotional intelligence for personalized experiences'
  },
  {
    title: 'Real-time Analytics',
    description: 'Live learning velocity optimization and performance prediction'
  },
  {
    title: 'Neural Networks',
    description: 'Advanced neural architectures and self-evolving AI systems'
  }
]
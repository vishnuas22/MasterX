'use client'

import { useState } from 'react'
import {
  MessageSquare,
  Plus,
  Settings,
  History,
  Menu,
  X,
  Send,
  Paperclip,
  Mic,
  Brain,
  Code,
  BarChart3,
  Lightbulb,
  ArrowRight
} from 'lucide-react'

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [activeView, setActiveView] = useState('chat')

  const navigationItems = [
    { id: 'chat', label: 'Chat', icon: MessageSquare },
    { id: 'history', label: 'History', icon: History },
    { id: 'settings', label: 'Settings', icon: Settings }
  ]

  const recentChats = [
    { id: '1', title: 'Python debugging help', time: '2m ago' },
    { id: '2', title: 'React component design', time: '1h ago' },
    { id: '3', title: 'Database optimization', time: '3h ago' },
    { id: '4', title: 'API integration guide', time: '1d ago' }
  ]

  const suggestions = [
    {
      icon: Brain,
      title: "Explain quantum computing",
      description: "Learn about quantum mechanics and computing principles"
    },
    {
      icon: Code,
      title: "Write Python code",
      description: "Generate and optimize code solutions"
    },
    {
      icon: BarChart3,
      title: "Analyze data",
      description: "Extract insights from complex datasets"
    },
    {
      icon: Lightbulb,
      title: "Brainstorm ideas",
      description: "Generate innovative solutions"
    }
  ]

  const handleSuggestionClick = (suggestion: any) => {
    console.log('Suggestion clicked:', suggestion.title)
    // In a real app, this would populate the input field or send the message
  }

  const handleSendMessage = () => {
    console.log('Send message clicked')
    // In a real app, this would send the message to the backend
  }

  const handleNewChat = () => {
    console.log('New chat clicked')
    // In a real app, this would create a new conversation
  }

  const handleRecentChatClick = (chat: any) => {
    console.log('Recent chat clicked:', chat.title)
    // In a real app, this would load the conversation
  }

  const renderMainContent = () => {
    switch (activeView) {
      case 'history':
        return (
          <div className="flex flex-col items-center justify-center h-full p-8">
            <div className="text-center">
              <History className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Chat History</h2>
              <p className="text-gray-600">Your conversation history will appear here.</p>
            </div>
          </div>
        )
      case 'settings':
        return (
          <div className="flex flex-col items-center justify-center h-full p-8">
            <div className="text-center">
              <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Settings</h2>
              <p className="text-gray-600">Configure your preferences and account settings.</p>
            </div>
          </div>
        )
      default:
        return (
          <div className="flex flex-col h-full bg-white">
            {/* Main Content */}
            <div className="flex-1 flex flex-col items-center justify-center p-8">
              {/* Welcome Section */}
              <div className="text-center mb-12 max-w-2xl">
                <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center mx-auto mb-6">
                  <Brain className="h-6 w-6 text-white" />
                </div>

                <h1 className="text-4xl font-bold text-gray-900 mb-4">
                  How can I help you today?
                </h1>

                <p className="text-lg text-gray-600">
                  I'm your AI assistant. Ask me anything or choose a suggestion below.
                </p>
              </div>

              {/* Suggestions */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl mb-12">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="p-4 border border-gray-200 rounded-xl text-left hover:border-gray-300 hover:shadow-sm transition-all duration-200 group"
                  >
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center group-hover:bg-gray-200 transition-colors">
                        <suggestion.icon className="h-4 w-4 text-gray-600" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900 mb-1">
                          {suggestion.title}
                        </h3>
                        <p className="text-sm text-gray-600">
                          {suggestion.description}
                        </p>
                      </div>
                      <ArrowRight className="h-4 w-4 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Input Area */}
            <div className="border-t border-gray-200 p-4">
              <div className="max-w-4xl mx-auto">
                <div className="relative">
                  <div className="flex items-end space-x-3 bg-gray-50 rounded-2xl p-3">
                    <button
                      onClick={() => console.log('File attachment clicked')}
                      className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                    >
                      <Paperclip className="h-4 w-4 text-gray-500" />
                    </button>

                    <div className="flex-1">
                      <textarea
                        placeholder="Message MasterX..."
                        className="w-full bg-transparent resize-none border-0 outline-none text-gray-900 placeholder-gray-500 text-sm leading-6"
                        rows={1}
                        style={{ minHeight: '24px', maxHeight: '200px' }}
                      />
                    </div>

                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => console.log('Voice input clicked')}
                        className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
                      >
                        <Mic className="h-4 w-4 text-gray-500" />
                      </button>
                      <button
                        onClick={handleSendMessage}
                        className="p-2 bg-black hover:bg-gray-800 rounded-lg transition-colors"
                      >
                        <Send className="h-4 w-4 text-white" />
                      </button>
                    </div>
                  </div>
                </div>

                <p className="text-xs text-gray-500 text-center mt-3">
                  MasterX can make mistakes. Consider checking important information.
                </p>
              </div>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="h-screen bg-gray-50 overflow-hidden">
      <div className="flex h-full">
        {/* Sidebar */}
        <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed left-0 top-0 h-full w-80 bg-white border-r border-gray-200 z-50 lg:relative lg:translate-x-0 lg:z-auto flex flex-col transition-transform duration-300`}>
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-100">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-black rounded-lg flex items-center justify-center">
                <MessageSquare className="h-4 w-4 text-white" />
              </div>
              <span className="font-semibold text-gray-900">MasterX</span>
            </div>

            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg lg:hidden"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* New Chat Button */}
          <div className="p-4">
            <button
              onClick={handleNewChat}
              className="w-full flex items-center justify-center space-x-2 bg-black text-white rounded-lg py-3 px-4 hover:bg-gray-800 transition-colors"
            >
              <Plus className="h-4 w-4" />
              <span className="font-medium">New chat</span>
            </button>
          </div>

          {/* Navigation */}
          <div className="px-4 pb-4">
            <nav className="space-y-1">
              {navigationItems.map((item) => {
                const isActive = activeView === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveView(item.id)}
                    className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                      isActive
                        ? "bg-gray-100 text-gray-900"
                        : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                    }`}
                  >
                    <item.icon className="h-4 w-4" />
                    <span className="font-medium">{item.label}</span>
                  </button>
                )
              })}
            </nav>
          </div>

          {/* Recent Chats */}
          <div className="flex-1 px-4 pb-4">
            <div className="mb-3">
              <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                Recent
              </h3>
            </div>

            <div className="space-y-1">
              {recentChats.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => handleRecentChatClick(chat)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-left hover:bg-gray-50 transition-colors group"
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-900 truncate">
                      {chat.title}
                    </div>
                    <div className="text-xs text-gray-500">
                      {chat.time}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-100">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-gray-600">U</span>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-gray-900">User</div>
                <div className="text-xs text-gray-500">Free plan</div>
              </div>
            </div>
          </div>
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="fixed top-4 left-4 z-50 p-2 bg-white border border-gray-200 rounded-lg shadow-sm lg:hidden"
        >
          <Menu className="h-4 w-4" />
        </button>

        {/* Main Content */}
        <main className="flex-1 relative overflow-hidden">
          {renderMainContent()}
        </main>

        {/* Mobile Overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </div>
    </div>
  )
}

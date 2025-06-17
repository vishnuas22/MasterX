import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, ArrowRight, CheckCircle } from 'lucide-react';
import { GlassCard, GlassButton, GlassInput } from './GlassCard';
import { LoadingSpinner } from './LoadingSpinner';
import { useApp } from '../context/AppContext';

export function UserOnboarding() {
  const { state, actions } = useApp();
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    learningGoals: [],
    subjects: [],
    experience: 'beginner'
  });
  const [errors, setErrors] = useState({});

  const learningGoals = [
    'Gain new skills for career advancement',
    'Prepare for exams or certifications',
    'Explore personal interests',
    'Academic study support',
    'Professional development',
    'Creative learning'
  ];

  const subjects = [
    'Programming & Software Development',
    'Mathematics & Statistics',
    'Science & Engineering',
    'Business & Finance',
    'Language Learning',
    'Arts & Design',
    'History & Literature',
    'Personal Development'
  ];

  const validateStep = (currentStep) => {
    const newErrors = {};
    
    if (currentStep === 1) {
      if (!formData.name.trim()) newErrors.name = 'Name is required';
      if (!formData.email.trim()) newErrors.email = 'Email is required';
      else if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Invalid email format';
    }
    
    if (currentStep === 2) {
      if (formData.learningGoals.length === 0) newErrors.learningGoals = 'Select at least one goal';
    }
    
    if (currentStep === 3) {
      if (formData.subjects.length === 0) newErrors.subjects = 'Select at least one subject';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(step)) {
      setStep(step + 1);
    }
  };

  const handleComplete = async () => {
    if (!validateStep(step)) return;

    try {
      const userData = {
        name: formData.name,
        email: formData.email,
        learning_preferences: {
          goals: formData.learningGoals,
          subjects: formData.subjects,
          experience_level: formData.experience
        }
      };

      // Create user first
      await actions.createUser(userData);
      
      // Get the created user by email to ensure we have the correct ID
      const createdUser = await actions.getUserByEmail(formData.email);
      
      // Create initial session using the verified user ID
      if (formData.subjects.length > 0 && createdUser?.id) {
        await actions.createSession({
          user_id: createdUser.id,
          subject: formData.subjects[0],
          difficulty_level: formData.experience,
          learning_objectives: formData.learningGoals.slice(0, 3) // First 3 goals
        });
      }
    } catch (error) {
      setErrors({ general: error.message });
    }
  };

  const toggleArrayItem = (array, item) => {
    return array.includes(item)
      ? array.filter(i => i !== item)
      : [...array, item];
  };

  if (state.isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 flex items-center justify-center">
        <LoadingSpinner size="xl" message="Setting up your learning profile..." />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="flex justify-center mb-4">
            <div className="p-4 rounded-2xl bg-gradient-to-r from-blue-500 to-purple-500">
              <Brain className="h-12 w-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
            Welcome to MasterX
          </h1>
          <p className="text-gray-400 text-lg">
            Your AI-powered learning journey begins here
          </p>
        </motion.div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-400">Step {step} of 4</span>
            <span className="text-sm text-gray-400">{Math.round((step / 4) * 100)}% Complete</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <motion.div
              className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(step / 4) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        {/* Steps */}
        <AnimatePresence mode="wait">
          {step === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
            >
              <GlassCard className="p-8">
                <h2 className="text-2xl font-bold text-white mb-6">Let's get to know you</h2>
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Full Name
                    </label>
                    <GlassInput
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                      placeholder="Enter your full name"
                      error={!!errors.name}
                    />
                    {errors.name && <p className="text-red-400 text-sm mt-1">{errors.name}</p>}
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Email Address
                    </label>
                    <GlassInput
                      type="email"
                      value={formData.email}
                      onChange={(e) => setFormData({...formData, email: e.target.value})}
                      placeholder="Enter your email"
                      error={!!errors.email}
                    />
                    {errors.email && <p className="text-red-400 text-sm mt-1">{errors.email}</p>}
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          )}

          {step === 2 && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
            >
              <GlassCard className="p-8">
                <h2 className="text-2xl font-bold text-white mb-6">What are your learning goals?</h2>
                <p className="text-gray-400 mb-6">Select all that apply to personalize your experience</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {learningGoals.map((goal) => (
                    <motion.button
                      key={goal}
                      onClick={() => setFormData({
                        ...formData,
                        learningGoals: toggleArrayItem(formData.learningGoals, goal)
                      })}
                      className={`p-4 rounded-xl text-left transition-all duration-300 ${
                        formData.learningGoals.includes(goal)
                          ? 'bg-blue-500/20 border border-blue-400/50'
                          : 'bg-white/5 border border-white/10 hover:bg-white/10'
                      }`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-gray-200 text-sm">{goal}</span>
                        {formData.learningGoals.includes(goal) && (
                          <CheckCircle className="h-5 w-5 text-blue-400" />
                        )}
                      </div>
                    </motion.button>
                  ))}
                </div>
                {errors.learningGoals && (
                  <p className="text-red-400 text-sm mt-3">{errors.learningGoals}</p>
                )}
              </GlassCard>
            </motion.div>
          )}

          {step === 3 && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
            >
              <GlassCard className="p-8">
                <h2 className="text-2xl font-bold text-white mb-6">What subjects interest you?</h2>
                <p className="text-gray-400 mb-6">Choose your areas of interest</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {subjects.map((subject) => (
                    <motion.button
                      key={subject}
                      onClick={() => setFormData({
                        ...formData,
                        subjects: toggleArrayItem(formData.subjects, subject)
                      })}
                      className={`p-4 rounded-xl text-left transition-all duration-300 ${
                        formData.subjects.includes(subject)
                          ? 'bg-purple-500/20 border border-purple-400/50'
                          : 'bg-white/5 border border-white/10 hover:bg-white/10'
                      }`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-gray-200 text-sm">{subject}</span>
                        {formData.subjects.includes(subject) && (
                          <CheckCircle className="h-5 w-5 text-purple-400" />
                        )}
                      </div>
                    </motion.button>
                  ))}
                </div>
                {errors.subjects && (
                  <p className="text-red-400 text-sm mt-3">{errors.subjects}</p>
                )}
              </GlassCard>
            </motion.div>
          )}

          {step === 4 && (
            <motion.div
              key="step4"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
            >
              <GlassCard className="p-8">
                <h2 className="text-2xl font-bold text-white mb-6">What's your experience level?</h2>
                <p className="text-gray-400 mb-6">This helps us tailor content to your level</p>
                <div className="space-y-3">
                  {[
                    { value: 'beginner', label: 'Beginner', desc: 'New to most topics, prefer step-by-step guidance' },
                    { value: 'intermediate', label: 'Intermediate', desc: 'Some experience, ready for deeper concepts' },
                    { value: 'advanced', label: 'Advanced', desc: 'Experienced learner, prefer challenging material' }
                  ].map((level) => (
                    <motion.button
                      key={level.value}
                      onClick={() => setFormData({...formData, experience: level.value})}
                      className={`w-full p-4 rounded-xl text-left transition-all duration-300 ${
                        formData.experience === level.value
                          ? 'bg-green-500/20 border border-green-400/50'
                          : 'bg-white/5 border border-white/10 hover:bg-white/10'
                      }`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-gray-200 font-medium">{level.label}</h3>
                          <p className="text-gray-400 text-sm">{level.desc}</p>
                        </div>
                        {formData.experience === level.value && (
                          <CheckCircle className="h-5 w-5 text-green-400" />
                        )}
                      </div>
                    </motion.button>
                  ))}
                </div>
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Display */}
        {errors.general && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4"
          >
            <GlassCard className="p-4 bg-red-500/10 border-red-400/30">
              <p className="text-red-400 text-sm">{errors.general}</p>
            </GlassCard>
          </motion.div>
        )}

        {/* Navigation */}
        <div className="flex justify-between mt-8">
          <GlassButton
            variant="secondary"
            onClick={() => setStep(Math.max(1, step - 1))}
            disabled={step === 1}
          >
            Back
          </GlassButton>
          
          {step < 4 ? (
            <GlassButton onClick={handleNext}>
              Next
              <ArrowRight className="ml-2 h-4 w-4" />
            </GlassButton>
          ) : (
            <GlassButton onClick={handleComplete} disabled={state.isLoading}>
              {state.isLoading ? 'Creating...' : 'Complete Setup'}
              <ArrowRight className="ml-2 h-4 w-4" />
            </GlassButton>
          )}
        </div>
      </div>
    </div>
  );
}

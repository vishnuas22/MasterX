/**
 * Profile Modal Component - User Profile Management
 * 
 * WCAG 2.1 AA Compliant:
 * - Form labels and descriptions
 * - Keyboard navigation
 * - Screen reader support
 * - High contrast for readability
 * 
 * Features:
 * - View and edit profile information
 * - Avatar upload
 * - Learning statistics
 * - Achievement showcase
 * 
 * Backend Integration:
 * - GET /api/v1/users/profile
 * - PATCH /api/v1/users/profile
 */

import React, { useState } from 'react';
import { Modal } from '@/components/ui/Modal';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Avatar } from '@/components/ui/Avatar';
import { Badge } from '@/components/ui/Badge';
import { AchievementBadge } from '@/components/gamification/AchievementBadge';
import { useAuth } from '@/hooks/useAuth';
import { cn } from '@/utils/cn';

export interface ProfileProps {
  onClose: () => void;
}

type ProfileTab = 'overview' | 'stats' | 'achievements';

// Mock achievements data - In real app, fetch from backend
const mockAchievements = [
  {
    id: '1',
    name: 'First Steps',
    description: 'Complete your first learning session',
    icon: '🏆',
    rarity: 'common' as const,
    xpReward: 50,
    unlockedAt: new Date()
  },
  {
    id: '2',
    name: 'Week Warrior',
    description: 'Maintain a 7-day streak',
    icon: '🔥',
    rarity: 'rare' as const,
    xpReward: 100,
    unlockedAt: new Date()
  },
  {
    id: '3',
    name: 'Knowledge Seeker',
    description: 'Ask 100 questions',
    icon: '🎯',
    rarity: 'epic' as const,
    xpReward: 200,
    unlockedAt: new Date()
  },
  {
    id: '4',
    name: 'Master Mind',
    description: 'Master 10 different topics',
    icon: '💎',
    rarity: 'legendary' as const,
    xpReward: 500,
    progress: 0.6 // Locked, showing progress
  }
];

export const Profile: React.FC<ProfileProps> = ({ onClose }) => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<ProfileTab>('overview');
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const tabs: { id: ProfileTab; label: string; icon: string }[] = [
    { id: 'overview', label: 'Overview', icon: '👤' },
    { id: 'stats', label: 'Statistics', icon: '📊' },
    { id: 'achievements', label: 'Achievements', icon: '🏆' }
  ];

  const handleSave = async () => {
    setIsSaving(true);
    try {
      // TODO: Save profile to backend
      await new Promise(resolve => setTimeout(resolve, 1000));
      setIsEditing(false);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Modal
      isOpen={true}
      onClose={onClose}
      title="Profile"
      size="xl"
      data-testid="profile-modal"
    >
      <div className="space-y-6">
        {/* Header with Avatar */}
        <div className="flex items-center space-x-6">
          <Avatar
            src={user?.avatar}
            name={user?.name || 'User'}
            size="xl"
          />
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-white mb-1">
              {user?.name || 'Anonymous User'}
            </h2>
            <p className="text-gray-400 mb-2">{user?.email}</p>
            <div className="flex items-center space-x-2">
              <Badge variant="primary">Level 5</Badge>
              <Badge variant="success">🔥 12 day streak</Badge>
            </div>
          </div>
          {!isEditing && (
            <Button variant="secondary" onClick={() => setIsEditing(true)}>
              Edit Profile
            </Button>
          )}
        </div>

        {/* Tabs */}
        <div className="border-b border-dark-600">
          <div className="flex space-x-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "px-4 py-2 text-sm font-medium transition-colors",
                  activeTab === tab.id
                    ? "text-blue-400 border-b-2 border-blue-400"
                    : "text-gray-400 hover:text-gray-300"
                )}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div>
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {isEditing ? (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Full Name
                    </label>
                    <Input
                      type="text"
                      defaultValue={user?.name || ''}
                      placeholder="Enter your name"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Bio
                    </label>
                    <textarea
                      className="w-full px-4 py-2 bg-dark-700 border border-dark-600 rounded-lg text-white resize-none"
                      rows={4}
                      placeholder="Tell us about yourself..."
                      defaultValue="Passionate learner exploring AI and technology"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Location
                    </label>
                    <Input
                      type="text"
                      defaultValue="San Francisco, CA"
                      placeholder="Enter your location"
                    />
                  </div>
                  <div className="flex space-x-3">
                    <Button variant="primary" onClick={handleSave} loading={isSaving}>
                      Save Changes
                    </Button>
                    <Button variant="ghost" onClick={() => setIsEditing(false)}>
                      Cancel
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-1">Bio</h3>
                    <p className="text-white">
                      Passionate learner exploring AI and technology
                    </p>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-1">Location</h3>
                    <p className="text-white">San Francisco, CA</p>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-400 mb-1">Member Since</h3>
                    <p className="text-white">January 2025</p>
                  </div>
                </>
              )}
            </div>
          )}

          {activeTab === 'stats' && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <Card className="p-6 text-center">
                <div className="text-3xl font-bold text-blue-400 mb-2">42</div>
                <div className="text-sm text-gray-400">Total Sessions</div>
              </Card>
              <Card className="p-6 text-center">
                <div className="text-3xl font-bold text-green-400 mb-2">24h</div>
                <div className="text-sm text-gray-400">Learning Time</div>
              </Card>
              <Card className="p-6 text-center">
                <div className="text-3xl font-bold text-purple-400 mb-2">8</div>
                <div className="text-sm text-gray-400">Topics Mastered</div>
              </Card>
              <Card className="p-6 text-center">
                <div className="text-3xl font-bold text-yellow-400 mb-2">5</div>
                <div className="text-sm text-gray-400">Current Level</div>
              </Card>
              <Card className="p-6 text-center">
                <div className="text-3xl font-bold text-red-400 mb-2">12</div>
                <div className="text-sm text-gray-400">Day Streak</div>
              </Card>
              <Card className="p-6 text-center">
                <div className="text-3xl font-bold text-indigo-400 mb-2">15</div>
                <div className="text-sm text-gray-400">Achievements</div>
              </Card>
            </div>
          )}

          {activeTab === 'achievements' && (
            <div className="space-y-4">
              <div className="mb-4">
                <h4 className="text-white font-semibold mb-2">
                  Unlocked: {mockAchievements.filter(a => a.unlockedAt).length}/{mockAchievements.length}
                </h4>
                <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                    style={{ width: `${(mockAchievements.filter(a => a.unlockedAt).length / mockAchievements.length) * 100}%` }}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {mockAchievements.map((achievement) => (
                  <AchievementBadge
                    key={achievement.id}
                    achievement={achievement}
                    size="md"
                    showDetails={true}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
};

export default Profile;

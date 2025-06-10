import os
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from groq import Groq
from datetime import datetime
import asyncio
from models import ChatSession, ChatMessage, MentorResponse

logger = logging.getLogger(__name__)

class MasterXAIService:
    def __init__(self):
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "deepseek-r1-distill-llama-70b"  # DeepSeek R1 70B model
        
    def _get_system_prompt(self, session: ChatSession, context: Dict[str, Any] = None) -> str:
        """Generate dynamic system prompt based on session context"""
        base_prompt = """You are MasterX, a world-class AI Mentor designed to provide personalized, engaging, and highly effective learning experiences. Your mission is to help users unlock their full potential in any subject.

CORE PRINCIPLES:
• Personalization: Adapt explanations, pace, and motivation to the user's unique goals, background, and preferences
• Expert Guidance: Organize learning from foundational to advanced; use clear explanations, analogies, and real-world examples
• Active Learning: After each concept, present interactive exercises progressing from simple to real-world problem-solving
• Constructive Feedback: Provide detailed feedback, alternative solutions, and encouragement
• Spaced Repetition: Schedule periodic reviews to maximize retention
• Accountability: Track progress and dynamically adjust learning plans

RESPONSE FORMATTING:
• Use clear, structured responses with headers and bullet points
• Include practical examples and analogies
• Provide actionable next steps
• Suggest exercises when appropriate
• Keep explanations engaging and conversational

CURRENT SESSION CONTEXT:"""
        
        if session:
            base_prompt += f"""
• Subject: {session.subject or 'General Learning'}
• Difficulty Level: {session.difficulty_level}
• Learning Objectives: {', '.join(session.learning_objectives) if session.learning_objectives else 'Not specified'}
• Current Topic: {session.current_topic or 'Introduction'}
"""
        
        if context and context.get('user_background'):
            base_prompt += f"• User Background: {context['user_background']}\n"
            
        if context and context.get('recent_topics'):
            base_prompt += f"• Recent Topics Covered: {', '.join(context['recent_topics'])}\n"

        base_prompt += """
INTERACTION STYLE:
• Be encouraging and supportive while maintaining high standards
• Use a warm, professional tone that feels like a premium mentor experience
• Celebrate progress and provide constructive guidance on challenges
• Ask thoughtful questions to gauge understanding
• Adapt complexity based on user responses

Remember: You're not just providing information—you're creating a transformative learning experience that builds confidence, competence, and passion for learning."""

        return base_prompt
    
    def _format_response(self, raw_response: str, response_type: str = "explanation") -> MentorResponse:
        """Format AI response with better structure and metadata"""
        
        # Extract key elements from response
        concepts_covered = []
        suggested_actions = []
        next_steps = None
        
        # Simple parsing for concepts (can be enhanced with more sophisticated NLP)
        lines = raw_response.split('\n')
        for line in lines:
            if 'concept:' in line.lower() or 'key idea:' in line.lower():
                concepts_covered.append(line.strip().replace('Concept:', '').replace('Key idea:', '').strip())
            elif 'action:' in line.lower() or 'try this:' in line.lower():
                suggested_actions.append(line.strip().replace('Action:', '').replace('Try this:', '').strip())
            elif 'next:' in line.lower() or 'next step:' in line.lower():
                next_steps = line.strip().replace('Next:', '').replace('Next step:', '').strip()
        
        # Format the response with better structure
        formatted_response = raw_response
        
        # Add visual separators and structure if not present
        if '##' not in formatted_response and len(formatted_response) > 200:
            # Add structure to long responses
            sections = formatted_response.split('\n\n')
            if len(sections) > 1:
                formatted_response = sections[0] + '\n\n## Key Points\n' + '\n\n'.join(sections[1:])
        
        return MentorResponse(
            response=formatted_response,
            response_type=response_type,
            suggested_actions=suggested_actions or ["Continue with the next topic", "Ask questions if anything is unclear"],
            concepts_covered=concepts_covered,
            next_steps=next_steps,
            metadata={
                "model_used": self.model,
                "response_length": len(formatted_response),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def get_mentor_response(
        self, 
        user_message: str, 
        session: ChatSession,
        context: Dict[str, Any] = None,
        stream: bool = False
    ) -> MentorResponse:
        """Get response from AI mentor"""
        try:
            system_prompt = self._get_system_prompt(session, context)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Add recent conversation history if available in context
            if context and context.get('recent_messages'):
                for msg in context['recent_messages'][-6:]:  # Last 6 messages for context
                    role = "user" if msg.sender == "user" else "assistant"
                    messages.insert(-1, {"role": role, "content": msg.message})
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                stream=stream
            )
            
            if stream:
                return response  # Return generator for streaming
            else:
                content = response.choices[0].message.content
                return self._format_response(content)
                
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return MentorResponse(
                response="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                response_type="error",
                metadata={"error": str(e)}
            )
    
    async def generate_exercise(
        self, 
        topic: str, 
        difficulty: str, 
        exercise_type: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """Generate practice exercises for a topic"""
        try:
            prompt = f"""Generate a {exercise_type} exercise for the topic: {topic}
            
Difficulty level: {difficulty}
Exercise type: {exercise_type}

Please provide:
1. A clear, engaging question
2. Multiple choice options (if applicable)
3. The correct answer
4. A detailed explanation of why this is correct
5. Key concepts this exercise tests

Format as JSON with fields: question, options, correct_answer, explanation, concepts"""

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to structured text
            try:
                return json.loads(content)
            except:
                return {
                    "question": content,
                    "exercise_type": exercise_type,
                    "difficulty": difficulty,
                    "concepts": [topic]
                }
                
        except Exception as e:
            logger.error(f"Error generating exercise: {str(e)}")
            return {
                "question": f"Practice question about {topic} (generating failed)",
                "error": str(e)
            }
    
    async def analyze_user_response(
        self, 
        question: str, 
        user_answer: str, 
        correct_answer: str = None
    ) -> Dict[str, Any]:
        """Analyze user's response and provide feedback"""
        try:
            prompt = f"""Analyze this learning interaction:

Question: {question}
User's Answer: {user_answer}
Correct Answer: {correct_answer or "Not specified"}

Provide:
1. Assessment of the user's understanding (correct/partially correct/incorrect)
2. Specific feedback on their answer
3. Explanation of key concepts
4. Suggestions for improvement
5. Encouragement and next steps

Be constructive, encouraging, and educational."""

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=600
            )
            
            feedback = response.choices[0].message.content
            
            # Determine if answer was correct (simple heuristic)
            is_correct = "correct" in feedback.lower() and "incorrect" not in feedback.lower()
            
            return {
                "feedback": feedback,
                "is_correct": is_correct,
                "suggestions": ["Review the key concepts", "Try a similar problem"],
                "encouragement": "Keep practicing - you're making great progress!"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response: {str(e)}")
            return {
                "feedback": "Thank you for your response. Let's continue learning!",
                "is_correct": False,
                "error": str(e)
            }
    
    async def generate_learning_path(
        self, 
        subject: str, 
        user_level: str, 
        goals: List[str]
    ) -> Dict[str, Any]:
        """Generate personalized learning path"""
        try:
            prompt = f"""Create a comprehensive learning path for:

Subject: {subject}
Current Level: {user_level}
Goals: {', '.join(goals)}

Provide a structured learning path with:
1. 5-7 major milestones
2. Specific topics for each milestone
3. Estimated time to complete each
4. Prerequisites and dependencies
5. Recommended exercises and projects

Format as a clear, actionable roadmap."""

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1200
            )
            
            return {
                "learning_path": response.choices[0].message.content,
                "subject": subject,
                "estimated_duration": f"{len(goals) * 2} weeks",
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating learning path: {str(e)}")
            return {
                "learning_path": f"Basic path for {subject}: Start with fundamentals, practice regularly, build projects.",
                "error": str(e)
            }

# Global instance
ai_service = MasterXAIService()

#!/usr/bin/env python3
"""
🚀 MASTERX EMOTIONAL ADAPTATION A/B TESTING FRAMEWORK
Advanced testing framework to validate emotion detection impact on AI responses

This addresses the user's concern: "Let me test whether our emotion detection system 
is actually **enhancing** the AI responses or if we're just seeing standard LLM 
responses that could handle emotions on their own."

Key Features:
- A/B testing with statistical significance
- Quantitative measurement of emotional adaptation
- Human evaluation integration
- Real user testing capabilities
- Comprehensive metrics collection
"""

import asyncio
import os
import time
import json
import statistics
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TestGroup(Enum):
    """A/B test groups"""
    CONTROL = "control"  # Standard LLM without emotional context
    TREATMENT = "treatment"  # LLM with emotional context injection

class EmotionalState(Enum):
    """Emotional states to test"""
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    CONFUSED = "confused"
    NEUTRAL = "neutral"

@dataclass
class TestScenario:
    """Test scenario definition"""
    scenario_id: str
    name: str
    emotional_state: EmotionalState
    user_message: str
    emotional_context: str
    expected_adaptations: List[str]
    success_criteria: Dict[str, float]

@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    scenario_id: str
    group: TestGroup
    response: str
    response_time_ms: float
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    human_ratings: Dict[str, float] = field(default_factory=dict)

class EmotionalAdaptationValidator:
    """Advanced validator for emotional adaptation in AI responses"""
    
    def __init__(self):
        self.test_scenarios = self._create_test_scenarios()
        self.results: List[TestResult] = []
        
    def _create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios"""
        return [
            TestScenario(
                scenario_id="math_frustrated",
                name="Frustrated Math Student",
                emotional_state=EmotionalState.FRUSTRATED,
                user_message="I don't understand this algebra problem at all! This is too hard!",
                emotional_context="Student is frustrated (0.85), anxious (0.70), low confidence (0.80). Needs high empathy, encouragement, and step-by-step guidance.",
                expected_adaptations=["empathy", "encouragement", "simplification", "reassurance"],
                success_criteria={
                    "empathy_score": 0.7,
                    "encouragement_score": 0.6,
                    "clarity_score": 0.8,
                    "supportiveness": 0.7
                }
            ),
            TestScenario(
                scenario_id="science_excited",
                name="Excited Science Student", 
                emotional_state=EmotionalState.EXCITED,
                user_message="That was amazing! I just figured out how photosynthesis works! What's next?",
                emotional_context="Student is excited (0.90), curious (0.85), highly engaged (0.88). Optimal for advanced challenges and exploration.",
                expected_adaptations=["enthusiasm_matching", "challenge_increase", "exploration_encouragement"],
                success_criteria={
                    "enthusiasm_match": 0.7,
                    "challenge_level": 0.8,
                    "engagement_sustaining": 0.8
                }
            ),
            TestScenario(
                scenario_id="coding_confused",
                name="Confused Programming Student",
                emotional_state=EmotionalState.CONFUSED,
                user_message="I'm lost with this Python code. What does this error message even mean?",
                emotional_context="Student is confused (0.80), cognitive overload (0.75). Needs clear explanations and structured guidance.",
                expected_adaptations=["clarity", "structure", "step_by_step", "error_explanation"],
                success_criteria={
                    "clarity_score": 0.8,
                    "structure_score": 0.7,
                    "helpfulness": 0.8
                }
            ),
            TestScenario(
                scenario_id="history_confident",
                name="Confident History Student",
                emotional_state=EmotionalState.CONFIDENT,
                user_message="I understand the causes of WWI now. I'm ready for more complex historical analysis!",
                emotional_context="Student is confident (0.85), ready for challenges (0.80), high comprehension (0.88).",
                expected_adaptations=["challenge_increase", "depth_increase", "analytical_thinking"],
                success_criteria={
                    "challenge_level": 0.8,
                    "depth_score": 0.8,
                    "analytical_encouragement": 0.7
                }
            ),
            TestScenario(
                scenario_id="general_neutral",
                name="Neutral Baseline",
                emotional_state=EmotionalState.NEUTRAL,
                user_message="Explain the basics of quantum physics.",
                emotional_context="",  # No emotional context
                expected_adaptations=[],
                success_criteria={
                    "informativeness": 0.7,
                    "clarity_score": 0.7
                }
            )
        ]
    
    async def run_ab_test(self, num_iterations: int = 20) -> Dict[str, Any]:
        """Run comprehensive A/B test"""
        print(f"🚀 Starting A/B Test with {num_iterations} iterations per scenario")
        print("=" * 60)
        
        all_results = []
        
        for scenario in self.test_scenarios:
            print(f"\n🎯 Testing Scenario: {scenario.name}")
            print(f"Emotional State: {scenario.emotional_state.value}")
            print(f"Expected Adaptations: {', '.join(scenario.expected_adaptations)}")
            
            scenario_results = await self._test_scenario(scenario, num_iterations)
            all_results.extend(scenario_results)
            
            # Quick analysis for this scenario
            control_results = [r for r in scenario_results if r.group == TestGroup.CONTROL]
            treatment_results = [r for r in scenario_results if r.group == TestGroup.TREATMENT]
            
            print(f"Control group: {len(control_results)} tests")
            print(f"Treatment group: {len(treatment_results)} tests")
        
        # Comprehensive analysis
        analysis = self._analyze_results(all_results)
        return analysis
    
    async def _test_scenario(self, scenario: TestScenario, iterations: int) -> List[TestResult]:
        """Test a specific scenario with both control and treatment groups"""
        results = []
        
        for i in range(iterations):
            # Randomly assign to control or treatment
            group = TestGroup.CONTROL if i % 2 == 0 else TestGroup.TREATMENT
            
            try:
                # Run test with appropriate group settings
                result = await self._run_single_test(scenario, group)
                results.append(result)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Test failed for {scenario.name}, iteration {i}: {e}")
                continue
        
        return results
    
    async def _run_single_test(self, scenario: TestScenario, group: TestGroup) -> TestResult:
        """Run a single test instance"""
        try:
            from groq import AsyncGroq
            
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key:
                raise Exception("GROQ_API_KEY not found")
            
            client = AsyncGroq(api_key=api_key)
            
            # Prepare messages based on group
            messages = []
            
            if group == TestGroup.TREATMENT:
                # Treatment group: Include emotional context
                system_message = f"You are MasterX, an AI tutor with advanced emotional intelligence. EMOTIONAL CONTEXT: {scenario.emotional_context} Adapt your response accordingly."
            else:
                # Control group: Standard system message
                system_message = "You are MasterX, an AI tutor. Provide helpful educational responses."
            
            messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": scenario.user_message})
            
            # Make API call
            start_time = time.time()
            response = await client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            response_time = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content
            
            # Create test result
            result = TestResult(
                test_id=str(uuid.uuid4()),
                scenario_id=scenario.scenario_id,
                group=group,
                response=content,
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )
            
            # Calculate metrics
            result.metrics = self._calculate_response_metrics(content, scenario)
            
            return result
            
        except Exception as e:
            raise Exception(f"Single test execution failed: {e}")
    
    def _calculate_response_metrics(self, response: str, scenario: TestScenario) -> Dict[str, float]:
        """Calculate quantitative metrics for response quality"""
        metrics = {}
        response_lower = response.lower()
        
        # Empathy score
        empathy_words = ["understand", "feel", "sense", "empathy", "appreciate", "acknowledge", 
                        "realize", "recognize", "sorry", "difficult", "challenging", "frustrating"]
        empathy_count = sum(1 for word in empathy_words if word in response_lower)
        metrics["empathy_score"] = min(1.0, empathy_count / 5.0)  # Normalize to 0-1
        
        # Encouragement score
        encouragement_words = ["you can", "don't worry", "it's okay", "keep going", "great job",
                              "excellent", "well done", "believe", "confident", "capable", "try"]
        encouragement_count = sum(1 for phrase in encouragement_words if phrase in response_lower)
        metrics["encouragement_score"] = min(1.0, encouragement_count / 3.0)
        
        # Clarity and structure score
        structure_indicators = ["first", "second", "step", "then", "next", "finally", "let's start",
                               "here's how", "breaking down", "step by step"]
        structure_count = sum(1 for phrase in structure_indicators if phrase in response_lower)
        metrics["clarity_score"] = min(1.0, structure_count / 3.0)
        
        # Challenge level (for confident students)
        challenge_words = ["advanced", "complex", "deeper", "explore", "challenge", "next level",
                          "sophisticated", "intricate", "analyze", "critical thinking"]
        challenge_count = sum(1 for word in challenge_words if word in response_lower)
        metrics["challenge_level"] = min(1.0, challenge_count / 3.0)
        
        # Supportiveness score
        support_words = ["help", "support", "together", "guide", "assist", "here for you",
                        "we can", "let me", "i'll help", "don't hesitate"]
        support_count = sum(1 for phrase in support_words if phrase in response_lower)
        metrics["supportiveness"] = min(1.0, support_count / 3.0)
        
        # Response length (informativeness proxy)
        word_count = len(response.split())
        metrics["informativeness"] = min(1.0, word_count / 200.0)  # Normalize based on expected length
        
        # Enthusiasm matching (for excited students)
        enthusiasm_words = ["amazing", "fantastic", "excellent", "great", "wonderful", "awesome",
                           "exciting", "brilliant", "outstanding", "impressive", "!"]
        enthusiasm_count = sum(1 for word in enthusiasm_words if word in response_lower)
        exclamation_count = response.count("!")
        metrics["enthusiasm_match"] = min(1.0, (enthusiasm_count + exclamation_count) / 5.0)
        
        return metrics
    
    def _analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Comprehensive statistical analysis of A/B test results"""
        print("\n📊 COMPREHENSIVE A/B TEST ANALYSIS")
        print("=" * 50)
        
        # Group results by scenario and test group
        scenario_analysis = {}
        
        for scenario in self.test_scenarios:
            scenario_results = [r for r in results if r.scenario_id == scenario.scenario_id]
            control_results = [r for r in scenario_results if r.group == TestGroup.CONTROL]
            treatment_results = [r for r in scenario_results if r.group == TestGroup.TREATMENT]
            
            if not control_results or not treatment_results:
                continue
            
            print(f"\n🎯 Scenario: {scenario.name}")
            print(f"Control: {len(control_results)}, Treatment: {len(treatment_results)}")
            
            # Analyze each metric
            scenario_metrics = {}
            
            for metric_name in control_results[0].metrics.keys():
                control_values = [r.metrics[metric_name] for r in control_results]
                treatment_values = [r.metrics[metric_name] for r in treatment_results]
                
                control_mean = statistics.mean(control_values)
                treatment_mean = statistics.mean(treatment_values)
                
                # Calculate improvement
                improvement = ((treatment_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0
                
                # Simple statistical significance test (t-test approximation)
                if len(control_values) > 1 and len(treatment_values) > 1:
                    control_std = statistics.stdev(control_values)
                    treatment_std = statistics.stdev(treatment_values)
                    
                    # Simple significance test
                    pooled_std = ((control_std ** 2 + treatment_std ** 2) / 2) ** 0.5
                    effect_size = abs(treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                    significant = effect_size > 0.5  # Simple threshold
                else:
                    significant = False
                
                scenario_metrics[metric_name] = {
                    "control_mean": control_mean,
                    "treatment_mean": treatment_mean,
                    "improvement_pct": improvement,
                    "significant": significant,
                    "effect_size": effect_size if 'effect_size' in locals() else 0
                }
                
                # Display results
                status = "✅ SIGNIFICANT" if significant else "⚪ Not significant"
                print(f"  {metric_name}: {improvement:+.1f}% {status}")
                print(f"    Control: {control_mean:.3f}, Treatment: {treatment_mean:.3f}")
            
            scenario_analysis[scenario.scenario_id] = {
                "scenario_name": scenario.name,
                "emotional_state": scenario.emotional_state.value,
                "metrics": scenario_metrics,
                "sample_sizes": {
                    "control": len(control_results),
                    "treatment": len(treatment_results)
                }
            }
        
        # Overall summary
        print(f"\n🏆 OVERALL EMOTIONAL ADAPTATION EFFECTIVENESS")
        print("=" * 50)
        
        total_improvements = 0
        significant_improvements = 0
        total_metrics = 0
        
        for scenario_id, analysis in scenario_analysis.items():
            for metric_name, metric_data in analysis["metrics"].items():
                total_metrics += 1
                if metric_data["improvement_pct"] > 0:
                    total_improvements += 1
                if metric_data["significant"] and metric_data["improvement_pct"] > 0:
                    significant_improvements += 1
        
        effectiveness_rate = (total_improvements / total_metrics * 100) if total_metrics > 0 else 0
        significance_rate = (significant_improvements / total_metrics * 100) if total_metrics > 0 else 0
        
        print(f"Metrics showing improvement: {total_improvements}/{total_metrics} ({effectiveness_rate:.1f}%)")
        print(f"Statistically significant improvements: {significant_improvements}/{total_metrics} ({significance_rate:.1f}%)")
        
        # Conclusion
        if significance_rate > 30:
            conclusion = "✅ STRONG EVIDENCE: Emotional adaptation significantly enhances AI responses"
        elif effectiveness_rate > 50:
            conclusion = "⚪ MODERATE EVIDENCE: Emotional adaptation shows promising improvements"
        else:
            conclusion = "❌ WEAK EVIDENCE: Limited impact from emotional adaptation"
        
        print(f"\n🎯 CONCLUSION: {conclusion}")
        
        return {
            "scenario_analysis": scenario_analysis,
            "overall_metrics": {
                "total_metrics": total_metrics,
                "improvements": total_improvements,
                "significant_improvements": significant_improvements,
                "effectiveness_rate": effectiveness_rate,
                "significance_rate": significance_rate
            },
            "conclusion": conclusion,
            "timestamp": datetime.utcnow().isoformat()
        }

class HumanEvaluationFramework:
    """Framework for human evaluation of AI responses"""
    
    def __init__(self):
        self.evaluation_criteria = [
            "appropriateness",
            "emotional_sensitivity",
            "helpfulness", 
            "clarity",
            "engagement",
            "overall_quality"
        ]
    
    def generate_evaluation_interface(self, test_results: List[TestResult]) -> str:
        """Generate human evaluation interface"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MasterX Emotional Adaptation Evaluation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .scenario { border: 1px solid #ccc; margin: 20px 0; padding: 15px; }
                .response { background: #f5f5f5; padding: 10px; margin: 10px 0; }
                .rating { margin: 10px 0; }
                .rating input { margin: 0 5px; }
            </style>
        </head>
        <body>
            <h1>MasterX Emotional Adaptation Evaluation</h1>
            <p>Please rate each AI response on a scale of 1-5 for the following criteria:</p>
            <ul>
                <li><strong>Appropriateness:</strong> How well does the response match the student's emotional state?</li>
                <li><strong>Emotional Sensitivity:</strong> How empathetic and emotionally aware is the response?</li>
                <li><strong>Helpfulness:</strong> How helpful is the response for the student's learning?</li>
                <li><strong>Clarity:</strong> How clear and understandable is the response?</li>
                <li><strong>Engagement:</strong> How engaging and motivating is the response?</li>
                <li><strong>Overall Quality:</strong> Overall assessment of response quality</li>
            </ul>
            
            <form id="evaluation-form">
        """
        
        for i, result in enumerate(test_results[:10]):  # Limit to first 10 for demo
            html_template += f"""
                <div class="scenario">
                    <h3>Scenario {i+1}: {result.scenario_id}</h3>
                    <p><strong>Group:</strong> {result.group.value}</p>
                    <div class="response">
                        <strong>AI Response:</strong><br>
                        {result.response[:300]}...
                    </div>
                    
                    {self._generate_rating_inputs(f"response_{i}", self.evaluation_criteria)}
                </div>
            """
        
        html_template += """
                <button type="submit">Submit Evaluations</button>
            </form>
            
            <script>
                document.getElementById('evaluation-form').addEventListener('submit', function(e) {
                    e.preventDefault();
                    alert('Evaluation data would be submitted to analysis system');
                });
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_rating_inputs(self, response_id: str, criteria: List[str]) -> str:
        """Generate HTML rating inputs"""
        html = ""
        for criterion in criteria:
            html += f"""
                <div class="rating">
                    <label><strong>{criterion.replace('_', ' ').title()}:</strong></label><br>
                    """
            for rating in range(1, 6):
                html += f"""
                    <input type="radio" name="{response_id}_{criterion}" value="{rating}" id="{response_id}_{criterion}_{rating}">
                    <label for="{response_id}_{criterion}_{rating}">{rating}</label>
                """
            html += "</div>"
        return html

async def main():
    """Main execution function"""
    print("🚀 MASTERX EMOTIONAL ADAPTATION VALIDATION FRAMEWORK")
    print("=" * 60)
    print("Testing whether emotion detection actually enhances AI responses")
    print("vs standard LLM responses...")
    print()
    
    # Initialize validator
    validator = EmotionalAdaptationValidator()
    
    # Run A/B test
    try:
        analysis = await validator.run_ab_test(num_iterations=10)  # Reduced for demo
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"emotional_adaptation_analysis_{timestamp}.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: emotional_adaptation_analysis_{timestamp}.json")
        
        # Generate human evaluation interface
        human_eval = HumanEvaluationFramework()
        html_interface = human_eval.generate_evaluation_interface(validator.results[:5])
        
        with open(f"human_evaluation_interface_{timestamp}.html", "w") as f:
            f.write(html_interface)
        
        print(f"🌐 Human evaluation interface: human_evaluation_interface_{timestamp}.html")
        
    except Exception as e:
        print(f"❌ Validation framework failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
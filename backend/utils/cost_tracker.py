"""
Cost Monitoring System
Following specifications from 2.CRITICAL_INITIAL_SETUP.md Section 2
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from utils.database import get_cost_tracking_collection
from utils.errors import CostThresholdError

logger = logging.getLogger(__name__)


class CostTracker:
    """Monitor and alert on API costs"""
    
    # Provider pricing (per 1M tokens) - Updated regularly
    PRICING = {
        'openai': {
            'gpt-4o': {'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000}
        },
        'anthropic': {
            'claude-sonnet-4': {'input': 3.00 / 1_000_000, 'output': 15.00 / 1_000_000}
        },
        'gemini': {
            'gemini-2.0-flash-exp': {'input': 0.075 / 1_000_000, 'output': 0.30 / 1_000_000}
        },
        'groq': {
            'llama-3.3-70b-versatile': {'input': 0.05 / 1_000_000, 'output': 0.08 / 1_000_000}
        },
        'emergent': {
            'gpt-4o': {'input': 2.50 / 1_000_000, 'output': 10.00 / 1_000_000}
        }
    }
    
    # Cost thresholds for alerts
    DAILY_THRESHOLD = 100.00  # Alert if > $100/day
    HOURLY_THRESHOLD = 10.00  # Alert if > $10/hour
    
    async def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost of single request"""
        
        pricing = self.PRICING.get(provider, {}).get(model, {
            'input': 0.00001,  # Default fallback
            'output': 0.00003
        })
        
        cost = (
            input_tokens * pricing['input'] +
            output_tokens * pricing['output']
        )
        
        return cost
    
    async def track_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str,
        category: str = "general"
    ) -> float:
        """Track cost of single request"""
        
        cost = await self.calculate_cost(provider, model, input_tokens, output_tokens)
        
        # Save to database
        await self.save_cost_record(
            provider=provider,
            model=model,
            user_id=user_id,
            category=category,
            cost=cost,
            tokens=input_tokens + output_tokens
        )
        
        # Check thresholds
        await self.check_cost_thresholds()
        
        return cost
    
    async def save_cost_record(
        self,
        provider: str,
        model: str,
        user_id: str,
        category: str,
        cost: float,
        tokens: int
    ):
        """Save cost record to database"""
        collection = get_cost_tracking_collection()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        # Update or insert cost record for today
        await collection.update_one(
            {
                "date": today,
                "provider": provider,
                "user_id": user_id
            },
            {
                "$inc": {
                    "total_requests": 1,
                    "total_tokens": tokens,
                    "total_cost": cost,
                    f"category_breakdown.{category}": cost
                },
                "$set": {
                    "date": today,
                    "provider": provider,
                    "user_id": user_id
                }
            },
            upsert=True
        )
    
    async def check_cost_thresholds(self):
        """Alert if costs exceed thresholds"""
        try:
            today_cost = await self.get_daily_cost()
            hour_cost = await self.get_hourly_cost()
            
            if today_cost > self.DAILY_THRESHOLD:
                logger.error(f"🚨 DAILY COST ALERT: ${today_cost:.2f} (threshold: ${self.DAILY_THRESHOLD})")
                # Could send alert email/Slack here
            
            if hour_cost > self.HOURLY_THRESHOLD:
                logger.warning(f"⚠️ HOURLY COST ALERT: ${hour_cost:.2f} (threshold: ${self.HOURLY_THRESHOLD})")
        except Exception as e:
            logger.error(f"Error checking cost thresholds: {e}")
    
    async def get_daily_cost(self) -> float:
        """Get total cost for today"""
        collection = get_cost_tracking_collection()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        pipeline = [
            {"$match": {"date": today}},
            {"$group": {"_id": None, "total": {"$sum": "$total_cost"}}}
        ]
        
        result = await collection.aggregate(pipeline).to_list(1)
        return result[0]["total"] if result else 0.0
    
    async def get_hourly_cost(self) -> float:
        """Get total cost for last hour"""
        # Simplified: return daily cost / 24 as approximation
        # In production, would track timestamp-based costs
        daily = await self.get_daily_cost()
        return daily / 24
    
    async def get_weekly_cost(self) -> float:
        """Get total cost for last 7 days"""
        collection = get_cost_tracking_collection()
        
        # Get dates for last 7 days
        dates = []
        for i in range(7):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(date)
        
        pipeline = [
            {"$match": {"date": {"$in": dates}}},
            {"$group": {"_id": None, "total": {"$sum": "$total_cost"}}}
        ]
        
        result = await collection.aggregate(pipeline).to_list(1)
        return result[0]["total"] if result else 0.0
    
    async def get_cost_breakdown(self, days: int = 7) -> Dict:
        """Get cost breakdown by provider, category, user"""
        collection = get_cost_tracking_collection()
        
        # Get dates
        dates = []
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(date)
        
        # By provider
        provider_pipeline = [
            {"$match": {"date": {"$in": dates}}},
            {"$group": {"_id": "$provider", "total": {"$sum": "$total_cost"}}}
        ]
        provider_result = await collection.aggregate(provider_pipeline).to_list(100)
        by_provider = {r["_id"]: r["total"] for r in provider_result}
        
        # By category (aggregate all category breakdowns)
        by_category = {
            "coding": 0.0,
            "math": 0.0,
            "empathy": 0.0,
            "research": 0.0,
            "general": 0.0
        }
        
        docs = await collection.find({"date": {"$in": dates}}).to_list(1000)
        for doc in docs:
            breakdown = doc.get("category_breakdown", {})
            for cat, cost in breakdown.items():
                if cat in by_category:
                    by_category[cat] += cost
        
        return {
            "by_provider": by_provider,
            "by_category": by_category
        }
    
    async def get_top_users(self, days: int = 7, limit: int = 10) -> list:
        """Get top users by cost"""
        collection = get_cost_tracking_collection()
        
        # Get dates
        dates = []
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(date)
        
        pipeline = [
            {"$match": {"date": {"$in": dates}, "user_id": {"$ne": None}}},
            {"$group": {"_id": "$user_id", "total_cost": {"$sum": "$total_cost"}}},
            {"$sort": {"total_cost": -1}},
            {"$limit": limit}
        ]
        
        result = await collection.aggregate(pipeline).to_list(limit)
        return result


# Global instance
cost_tracker = CostTracker()

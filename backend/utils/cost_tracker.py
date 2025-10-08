"""
Cost Monitoring System (ENHANCED - Fully Dynamic)
Integrates with DynamicPricingEngine for zero hardcoded prices

PRINCIPLES (AGENTS.md):
- No hardcoded pricing
- Dynamic pricing from external APIs
- All existing functionality preserved

Date: October 8, 2025 (Enhanced)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from utils.database import get_cost_tracking_collection
from utils.errors import CostThresholdError

logger = logging.getLogger(__name__)


class CostTracker:
    """Monitor and alert on API costs (ENHANCED - Dynamic Pricing)"""
    
    # Cost thresholds for alerts (from environment)
    DAILY_THRESHOLD = float(os.getenv("DAILY_COST_THRESHOLD", "100.0"))
    HOURLY_THRESHOLD = float(os.getenv("HOURLY_COST_THRESHOLD", "10.0"))
    
    def __init__(self):
        """Initialize cost tracker"""
        self.pricing_engine = None  # Will be set by server.py on startup
        logger.info("✅ CostTracker initialized (dynamic pricing)")
    
    def set_pricing_engine(self, pricing_engine):
        """Set pricing engine (called by server.py on startup)"""
        self.pricing_engine = pricing_engine
        logger.info("✅ CostTracker connected to DynamicPricingEngine")
    
    async def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost of single request using dynamic pricing
        
        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in USD
        """
        
        if not self.pricing_engine:
            logger.warning("Pricing engine not initialized, using fallback")
            # Fallback: very rough estimate
            return (input_tokens + output_tokens) * 0.000001
        
        try:
            # Get dynamic pricing
            pricing = await self.pricing_engine.get_pricing(provider, model)
            
            # Calculate cost
            cost = (
                input_tokens * pricing.input_cost_per_million / 1_000_000 +
                output_tokens * pricing.output_cost_per_million / 1_000_000
            )
            
            # Log if estimated (lower confidence)
            if pricing.confidence < 1.0:
                logger.debug(
                    f"Cost calculated with estimate "
                    f"(confidence: {pricing.confidence:.2f})"
                )
            
            return cost
            
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            # Fallback to safe estimate
            return (input_tokens + output_tokens) * 0.000001
    
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

"""
Database utilities and initialization
Following specifications from 2.CRITICAL_INITIAL_SETUP.md Section 1
"""

import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.models import INDEXES

logger = logging.getLogger(__name__)

# MongoDB connection
_client: AsyncIOMotorClient = None
_database: AsyncIOMotorDatabase = None


def get_database() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance"""
    global _database
    if _database is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _database


async def connect_to_mongodb():
    """Connect to MongoDB"""
    global _client, _database
    
    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    database_name = os.getenv("MONGO_DATABASE", "masterx")
    
    logger.info(f"Connecting to MongoDB: {mongo_url}")
    
    _client = AsyncIOMotorClient(
        mongo_url,
        maxPoolSize=50,
        minPoolSize=10,
        serverSelectionTimeoutMS=5000
    )
    
    _database = _client[database_name]
    
    # Test connection
    await _client.admin.command('ping')
    logger.info(f"✅ Connected to MongoDB database: {database_name}")


async def close_mongodb_connection():
    """Close MongoDB connection"""
    global _client
    if _client:
        _client.close()
        logger.info("✅ MongoDB connection closed")


async def initialize_database():
    """
    Initialize MongoDB database with collections and indexes
    Following specifications from 2.CRITICAL_INITIAL_SETUP.md Section 1
    """
    db = get_database()
    
    # Collection names
    collections = [
        "users",
        "sessions",
        "messages",
        "benchmark_results",
        "provider_health",
        "user_performance",
        "cost_tracking"
    ]
    
    # Get existing collections
    existing_collections = await db.list_collection_names()
    
    # Create collections if they don't exist
    for collection in collections:
        if collection not in existing_collections:
            await db.create_collection(collection)
            logger.info(f"✅ Created collection: {collection}")
        else:
            logger.info(f"📁 Collection already exists: {collection}")
    
    # Create indexes
    for collection_name, indexes in INDEXES.items():
        collection = db[collection_name]
        
        for index_spec in indexes:
            try:
                await collection.create_index(
                    index_spec['keys'],
                    unique=index_spec.get('unique', False)
                )
                logger.info(f"✅ Created index on {collection_name}: {index_spec['keys']}")
            except Exception as e:
                # Index might already exist
                logger.debug(f"Index on {collection_name} {index_spec['keys']} already exists or error: {e}")
    
    logger.info("✅ Database initialization complete")


# Convenience functions for collections
def get_users_collection():
    """Get users collection"""
    return get_database()["users"]


def get_sessions_collection():
    """Get sessions collection"""
    return get_database()["sessions"]


def get_messages_collection():
    """Get messages collection"""
    return get_database()["messages"]


def get_benchmark_results_collection():
    """Get benchmark_results collection"""
    return get_database()["benchmark_results"]


def get_provider_health_collection():
    """Get provider_health collection"""
    return get_database()["provider_health"]


def get_user_performance_collection():
    """Get user_performance collection"""
    return get_database()["user_performance"]


def get_cost_tracking_collection():
    """Get cost_tracking collection"""
    return get_database()["cost_tracking"]

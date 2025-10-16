"""
Performance optimization utilities.

This module provides caching, connection pooling, and query optimization
capabilities for the Marketing Project API.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger("marketing_project.performance.optimization")


class CacheManager:
    """In-memory cache manager with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = asyncio.Lock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry is expired."""
        if key not in self.expiry_times:
            return True
        return time.time() > self.expiry_times[key]
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self.expiry_times.items()
            if current_time > expiry_time
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.expiry_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        while len(self.cache) >= self.max_size:
            if self.cache:
                # Remove the least recently used item
                key, _ = self.cache.popitem(last=False)
                self.expiry_times.pop(key, None)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self.lock:
            if key in self.cache and not self._is_expired(key):
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hit_count += 1
                return value
            
            self.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        async with self.lock:
            # Cleanup expired entries first
            self._cleanup_expired()
            
            # Evict LRU if needed
            self._evict_lru()
            
            # Set value
            self.cache[key] = value
            ttl = ttl or self.default_ttl
            self.expiry_times[key] = time.time() + ttl
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.expiry_times.pop(key, None)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.expiry_times.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "expired_entries": len([
                    key for key in self.expiry_times
                    if self._is_expired(key)
                ])
            }
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


class ConnectionPool:
    """Database connection pool manager."""
    
    def __init__(self, max_connections: int = 10, min_connections: int = 2):
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connections: List[Any] = []
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.used_connections: Dict[Any, float] = {}
        self.connection_timeout = 300  # 5 minutes
        self.lock = asyncio.Lock()
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
            except RuntimeError:
                # No event loop running, will start when needed
                self._cleanup_task = None
    
    async def _cleanup_loop(self):
        """Background cleanup loop for expired connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
    
    async def _cleanup_expired_connections(self):
        """Remove expired connections."""
        current_time = time.time()
        expired_connections = []
        
        async with self.lock:
            for conn, last_used in list(self.used_connections.items()):
                if current_time - last_used > self.connection_timeout:
                    expired_connections.append(conn)
                    del self.used_connections[conn]
        
        # Close expired connections
        for conn in expired_connections:
            try:
                if hasattr(conn, 'close'):
                    await conn.close()
            except Exception as e:
                logger.warning(f"Error closing expired connection: {e}")
    
    async def get_connection(self) -> Any:
        """Get a connection from the pool."""
        # Try to get from available connections
        try:
            connection = self.available_connections.get_nowait()
            async with self.lock:
                self.used_connections[connection] = time.time()
            return connection
        except asyncio.QueueEmpty:
            pass
        
        # Create new connection if under limit
        async with self.lock:
            if len(self.connections) < self.max_connections:
                connection = await self._create_connection()
                self.connections.append(connection)
                self.used_connections[connection] = time.time()
                return connection
        
        # Wait for available connection
        connection = await self.available_connections.get()
        async with self.lock:
            self.used_connections[connection] = time.time()
        return connection
    
    async def return_connection(self, connection: Any):
        """Return a connection to the pool."""
        async with self.lock:
            if connection in self.used_connections:
                del self.used_connections[connection]
        
        await self.available_connections.put(connection)
    
    async def _create_connection(self) -> Any:
        """Create a new database connection."""
        # This would be implemented based on the specific database
        # For now, return a placeholder
        return {"id": len(self.connections), "created_at": time.time()}
    
    async def close_all(self):
        """Close all connections in the pool."""
        async with self.lock:
            # Close all connections
            for connection in self.connections:
                try:
                    if hasattr(connection, 'close'):
                        await connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            
            self.connections.clear()
            self.used_connections.clear()
            
            # Clear queue
            while not self.available_connections.empty():
                try:
                    self.available_connections.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        async with self.lock:
            return {
                "total_connections": len(self.connections),
                "available_connections": self.available_connections.qsize(),
                "used_connections": len(self.used_connections),
                "max_connections": self.max_connections,
                "min_connections": self.min_connections
            }


class QueryOptimizer:
    """Query optimization utilities."""
    
    def __init__(self):
        self.query_cache = {}
        self.query_stats = {}
    
    def optimize_sql_query(self, query: str) -> str:
        """Optimize SQL query."""
        # Basic query optimization
        query = query.strip()
        
        # Remove unnecessary whitespace
        query = ' '.join(query.split())
        
        # Add LIMIT if not present and query is SELECT
        if query.upper().startswith('SELECT') and 'LIMIT' not in query.upper():
            query += ' LIMIT 1000'
        
        return query
    
    def optimize_mongodb_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize MongoDB query."""
        optimized = query.copy()
        
        # Add limit if not present
        if 'limit' not in optimized:
            optimized['limit'] = 1000
        
        # Optimize projection
        if 'projection' in optimized:
            # Ensure _id is excluded if not explicitly included
            if '_id' not in optimized['projection']:
                optimized['projection']['_id'] = 0
        
        return optimized
    
    def get_query_plan(self, query: str, query_type: str = "sql") -> Dict[str, Any]:
        """Get query execution plan."""
        if query_type == "sql":
            return self._analyze_sql_query(query)
        elif query_type == "mongodb":
            return self._analyze_mongodb_query(query)
        else:
            return {"error": "Unsupported query type"}
    
    def _analyze_sql_query(self, query: str) -> Dict[str, Any]:
        """Analyze SQL query for optimization opportunities."""
        analysis = {
            "query": query,
            "type": "SELECT",
            "has_where": "WHERE" in query.upper(),
            "has_order_by": "ORDER BY" in query.upper(),
            "has_limit": "LIMIT" in query.upper(),
            "has_join": "JOIN" in query.upper(),
            "estimated_complexity": "low"
        }
        
        # Determine complexity
        if analysis["has_join"]:
            analysis["estimated_complexity"] = "high"
        elif analysis["has_where"] and analysis["has_order_by"]:
            analysis["estimated_complexity"] = "medium"
        
        return analysis
    
    def _analyze_mongodb_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MongoDB query for optimization opportunities."""
        analysis = {
            "query": query,
            "has_filter": bool(query.get("filter", {})),
            "has_sort": bool(query.get("sort", {})),
            "has_limit": "limit" in query,
            "has_projection": bool(query.get("projection", {})),
            "estimated_complexity": "low"
        }
        
        # Determine complexity
        filter_keys = len(query.get("filter", {}))
        if filter_keys > 5:
            analysis["estimated_complexity"] = "high"
        elif filter_keys > 2:
            analysis["estimated_complexity"] = "medium"
        
        return analysis
    
    def suggest_indexes(self, query: str, query_type: str = "sql") -> List[str]:
        """Suggest indexes for query optimization."""
        suggestions = []
        
        if query_type == "sql":
            # Extract WHERE conditions
            if "WHERE" in query.upper():
                where_part = query.upper().split("WHERE")[1].split("ORDER BY")[0].split("LIMIT")[0]
                # Simple column extraction (would be more sophisticated in practice)
                columns = []
                for word in where_part.split():
                    if word.isalpha() and word not in ["AND", "OR", "=", ">", "<", "LIKE"]:
                        columns.append(word)
                
                if columns:
                    suggestions.append(f"CREATE INDEX idx_{'_'.join(columns[:3])} ON table_name ({', '.join(columns[:3])})")
        
        return suggestions


# Global instances (lazy initialization)
cache_manager = None
connection_pool = None
query_optimizer = None

def get_cache_manager():
    """Get or create cache manager instance."""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
    return cache_manager

def get_connection_pool():
    """Get or create connection pool instance."""
    global connection_pool
    if connection_pool is None:
        connection_pool = ConnectionPool()
    return connection_pool

def get_query_optimizer():
    """Get or create query optimizer instance."""
    global query_optimizer
    if query_optimizer is None:
        query_optimizer = QueryOptimizer()
    return query_optimizer

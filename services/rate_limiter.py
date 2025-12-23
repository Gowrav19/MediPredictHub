"""
Rate Limiting Service for API endpoints
"""

import os
import time
import json
from typing import Dict, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self):
        self.requests_per_minute = int(os.getenv('RATE_LIMIT_PER_MINUTE', 60))
        self.requests_per_hour = int(os.getenv('RATE_LIMIT_PER_HOUR', 1000))
        
        # Store request timestamps for each IP
        self.minute_requests = defaultdict(deque)
        self.hour_requests = defaultdict(deque)
        
        # Cleanup old entries every 5 minutes
        self.last_cleanup = time.time()

    def is_allowed(self, ip_address: str) -> Dict[str, any]:
        """Check if request is allowed based on rate limits"""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > 300:  # 5 minutes
            self._cleanup_old_entries()
            self.last_cleanup = current_time
        
        # Check minute limit
        minute_requests = self.minute_requests[ip_address]
        self._remove_old_requests(minute_requests, current_time - 60)
        
        if len(minute_requests) >= self.requests_per_minute:
            return {
                'allowed': False,
                'error': 'Rate limit exceeded. Too many requests per minute.',
                'retry_after': 60 - (current_time - minute_requests[0])
            }
        
        # Check hour limit
        hour_requests = self.hour_requests[ip_address]
        self._remove_old_requests(hour_requests, current_time - 3600)
        
        if len(hour_requests) >= self.requests_per_hour:
            return {
                'allowed': False,
                'error': 'Rate limit exceeded. Too many requests per hour.',
                'retry_after': 3600 - (current_time - hour_requests[0])
            }
        
        # Add current request
        minute_requests.append(current_time)
        hour_requests.append(current_time)
        
        return {
            'allowed': True,
            'remaining_minute': self.requests_per_minute - len(minute_requests),
            'remaining_hour': self.requests_per_hour - len(hour_requests)
        }

    def _remove_old_requests(self, requests: deque, cutoff_time: float):
        """Remove requests older than cutoff_time"""
        while requests and requests[0] < cutoff_time:
            requests.popleft()

    def _cleanup_old_entries(self):
        """Clean up old entries from memory"""
        current_time = time.time()
        
        # Clean up minute requests
        for ip in list(self.minute_requests.keys()):
            self._remove_old_requests(self.minute_requests[ip], current_time - 60)
            if not self.minute_requests[ip]:
                del self.minute_requests[ip]
        
        # Clean up hour requests
        for ip in list(self.hour_requests.keys()):
            self._remove_old_requests(self.hour_requests[ip], current_time - 3600)
            if not self.hour_requests[ip]:
                del self.hour_requests[ip]

    def get_rate_limit_headers(self, ip_address: str) -> Dict[str, str]:
        """Get rate limit headers for response"""
        current_time = time.time()
        
        minute_remaining = max(0, self.requests_per_minute - len(self.minute_requests[ip_address]))
        hour_remaining = max(0, self.requests_per_hour - len(self.hour_requests[ip_address]))
        
        return {
            'X-RateLimit-Limit-Minute': str(self.requests_per_minute),
            'X-RateLimit-Remaining-Minute': str(minute_remaining),
            'X-RateLimit-Limit-Hour': str(self.requests_per_hour),
            'X-RateLimit-Remaining-Hour': str(hour_remaining)
        }

# Global rate limiter instance
rate_limiter = RateLimiter()

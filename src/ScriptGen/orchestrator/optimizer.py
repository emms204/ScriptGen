"""
LLM Optimizer for efficient LLM interaction.

This module provides optimization components for reducing LLM usage and costs
through techniques such as caching, batching, and deduplication.
"""

import os
import json
import hashlib
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict
import threading

from ..llm.llm_wrapper import LLMWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('llm_optimizer')


@dataclass
class CacheConfig:
    """Configuration for LLM caching"""
    enable_cache: bool = True
    cache_dir: str = '.cache'
    max_cache_entries: int = 1000
    max_cache_age_days: int = 30
    deduplication_window: int = 60  # seconds


@dataclass
class OptimizationStats:
    """Statistics for LLM optimization"""
    cache_hits: int = 0
    cache_misses: int = 0
    deduplicated_calls: int = 0
    batched_calls: int = 0
    calls: int = 0
    tokens: int = 0
    duration: float = 0.0


class LLMOptimizer:
    """
    LLM Optimizer for efficient LLM usage.
    
    This class provides methods for reducing LLM usage and costs through:
    1. Response caching to avoid duplicate calls
    2. Call batching to reduce overhead
    3. Prompt deduplication to avoid redundant calls
    4. Strategic token usage optimization
    
    It acts as a wrapper around LLMWrapper, interceping calls and applying
    optimization strategies automatically.
    """
    
    def __init__(
        self,
        llm_wrapper: Optional[LLMWrapper] = None,
        config: Optional[CacheConfig] = None
    ):
        """
        Initialize LLM optimizer.
        
        Args:
            llm_wrapper: LLM wrapper to optimize
            config: Cache configuration
        """
        self.llm_wrapper = llm_wrapper or LLMWrapper()
        self.config = config or CacheConfig()
        
        # Initialize cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.recent_calls: Dict[str, Any] = {}
        self.call_durations: List[float] = []
        
        # Initialize stats
        self.stats = OptimizationStats()
        
        # Initialize batching
        self.batch_mode = False
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_results: Dict[str, Any] = {}
        self.batch_lock = threading.Lock()
        
        # Load cache from disk if enabled
        if self.config.enable_cache:
            self._ensure_cache_dir()
            self._load_cache()
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists"""
        if not os.path.exists(self.config.cache_dir):
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def _load_cache(self) -> None:
        """Load cache from disk"""
        cache_file = os.path.join(self.config.cache_dir, 'llm_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cache entries")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk"""
        if not self.config.enable_cache:
            return
            
        cache_file = os.path.join(self.config.cache_dir, 'llm_cache.json')
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} cache entries")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries"""
        if not self.config.enable_cache:
            return
            
        # Check if we need to clean up
        if len(self.cache) <= self.config.max_cache_entries:
            return
            
        # Sort by timestamp and keep newest entries
        entries = [(k, v.get('timestamp', 0)) for k, v in self.cache.items()]
        entries.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only max_cache_entries
        keep_entries = entries[:self.config.max_cache_entries]
        keep_keys = set(k for k, _ in keep_entries)
        
        # Remove old entries
        self.cache = {k: v for k, v in self.cache.items() if k in keep_keys}
        logger.info(f"Cleaned up cache, {len(self.cache)} entries remain")
    
    def _compute_cache_key(
        self,
        prompt: str,
        model: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compute cache key for LLM call.
        
        Args:
            prompt: Prompt text
            model: LLM model name
            params: Call parameters
            
        Returns:
            Cache key
        """
        # Create key components
        components = [prompt]
        if model:
            components.append(str(model))
        if params:
            # Sort params to ensure consistent keys
            param_str = json.dumps(params, sort_keys=True)
            components.append(param_str)
        
        # Join components and hash
        key_str = '|'.join(components)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def call_llm(
        self,
        prompt: str,
        model: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call LLM with optimization.
        
        Args:
            prompt: Prompt text
            model: LLM model name
            params: Call parameters
            
        Returns:
            LLM response
        """
        # If in batch mode, add to queue and return placeholder
        if self.batch_mode:
            return self._add_to_batch(prompt, model, params)
        
        # Apply optimizations
        return self._optimized_call(prompt, model, params)
    
    def _optimized_call(
        self,
        prompt: str,
        model: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply optimizations to LLM call.
        
        Args:
            prompt: Prompt text
            model: LLM model name
            params: Call parameters
            
        Returns:
            LLM response
        """
        start_time = time.time()
        
        # Try to get from cache
        if self.config.enable_cache:
            cache_key = self._compute_cache_key(prompt, model, params)
            cached = self.cache.get(cache_key)
            
            if cached:
                self.stats.cache_hits += 1
                logger.debug(f"Cache hit for key {cache_key[:8]}...")
                return cached['response']
            
            # Check for deduplication within time window
            current_time = time.time()
            for call_key, call_data in list(self.recent_calls.items()):
                if current_time - call_data['timestamp'] > self.config.deduplication_window:
                    # Remove expired calls
                    self.recent_calls.pop(call_key, None)
                    continue
                
                if call_key == cache_key and 'response' in call_data:
                    self.stats.deduplicated_calls += 1
                    logger.debug(f"Deduplicated call for key {cache_key[:8]}...")
                    return call_data['response']
            
            # Add to recent calls
            self.recent_calls[cache_key] = {
                'timestamp': current_time,
                'prompt': prompt,
                'model': model,
                'params': params
            }
        
        # Cache miss - make actual call
        self.stats.cache_misses += 1
        self.stats.calls += 1
        
        try:
            response = self.llm_wrapper.call_llm(prompt=prompt, model=model, params=params)
            
            # Add tokens to stats
            if 'usage' in response and 'total_tokens' in response['usage']:
                self.stats.tokens += response['usage']['total_tokens']
            
            # Add to cache
            if self.config.enable_cache:
                self.cache[cache_key] = {
                    'timestamp': time.time(),
                    'response': response
                }
                self.recent_calls[cache_key]['response'] = response
                
                # Periodically clean up cache
                if len(self.cache) > self.config.max_cache_entries:
                    self._cleanup_cache()
                    self._save_cache()
            
            duration = time.time() - start_time
            self.call_durations.append(duration)
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            raise
    
    def start_batch(self) -> None:
        """Start batch mode for collecting multiple LLM calls"""
        with self.batch_lock:
            self.batch_mode = True
            self.batch_queue = []
            self.batch_results = {}
    
    def _add_to_batch(
        self,
        prompt: str,
        model: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add call to batch queue.
        
        Args:
            prompt: Prompt text
            model: LLM model name
            params: Call parameters
            
        Returns:
            Placeholder response
        """
        with self.batch_lock:
            # Generate a unique ID for this call
            call_id = f"batch_{len(self.batch_queue)}"
            
            # Add to queue
            self.batch_queue.append({
                'id': call_id,
                'prompt': prompt,
                'model': model,
                'params': params
            })
            
            # Return placeholder that will be resolved later
            return {
                'text': f"[BATCH_PLACEHOLDER:{call_id}]",
                'batch_id': call_id,
                'is_batch_placeholder': True
            }
    
    def end_batch(self) -> OptimizationStats:
        """
        End batch mode and process all queued calls.
        
        Returns:
            Optimization statistics
        """
        with self.batch_lock:
            if not self.batch_mode:
                logger.warning("end_batch called when not in batch mode")
                return OptimizationStats()
            
            # Process queue
            batch_start_time = time.time()
            queue = self.batch_queue.copy()
            self.batch_mode = False
            
            # Reset stats for this batch
            batch_stats = OptimizationStats()
            
            # Group by model and params for efficient batching
            call_groups = defaultdict(list)
            for call in queue:
                # Create a key for grouping similar calls
                model_key = call.get('model', 'default')
                params_key = json.dumps(call.get('params', {}), sort_keys=True)
                group_key = f"{model_key}|{params_key}"
                call_groups[group_key].append(call)
            
            # Process each group
            for group_key, calls in call_groups.items():
                logger.debug(f"Processing batch group of {len(calls)} calls")
                
                # Process calls in this group
                for call in calls:
                    result = self._optimized_call(
                        prompt=call['prompt'],
                        model=call['model'],
                        params=call['params']
                    )
                    
                    # Store result
                    self.batch_results[call['id']] = result
                    
                    # Update batch stats
                    if result.get('is_cached', False):
                        batch_stats.cache_hits += 1
                    else:
                        batch_stats.cache_misses += 1
                        batch_stats.calls += 1
                    
                    if result.get('is_deduplicated', False):
                        batch_stats.deduplicated_calls += 1
                    
                    # Update token usage
                    if 'usage' in result and 'total_tokens' in result['usage']:
                        batch_stats.tokens += result['usage']['total_tokens']
                
                # Count as batched calls if more than one per group
                if len(calls) > 1:
                    batch_stats.batched_calls += len(calls) - 1
            
            # Update batch duration
            batch_stats.duration = time.time() - batch_start_time
            
            # Update overall stats
            self.stats.cache_hits += batch_stats.cache_hits
            self.stats.cache_misses += batch_stats.cache_misses
            self.stats.deduplicated_calls += batch_stats.deduplicated_calls
            self.stats.batched_calls += batch_stats.batched_calls
            self.stats.calls += batch_stats.calls
            self.stats.tokens += batch_stats.tokens
            
            return batch_stats
    
    def resolve_batch_placeholder(
        self,
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve batch placeholder to actual response.
        
        Args:
            response: Placeholder response from batch call
            
        Returns:
            Actual response
        """
        if not response.get('is_batch_placeholder', False):
            return response
            
        batch_id = response.get('batch_id')
        if batch_id in self.batch_results:
            return self.batch_results[batch_id]
            
        logger.warning(f"Batch placeholder {batch_id} not found in results")
        return response
    
    def get_stats(self) -> OptimizationStats:
        """
        Get current optimization statistics.
        
        Returns:
            Optimization statistics
        """
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset optimization statistics"""
        self.stats = OptimizationStats()
    
    def estimate_cost(self, model: str, tokens: int) -> float:
        """
        Estimate cost of LLM usage.
        
        Args:
            model: LLM model name
            tokens: Number of tokens
            
        Returns:
            Estimated cost in USD
        """
        # Simplified cost estimates for common models
        cost_per_1k_tokens = {
            'gpt-4': 0.03,
            'gpt-3.5-turbo': 0.002,
            'claude-2': 0.0113,
            'mistral-7b': 0.0,  # Open source
            'default': 0.01  # Default fallback
        }
        
        # Get cost rate for model or use default
        rate = cost_per_1k_tokens.get(model, cost_per_1k_tokens['default'])
        
        # Calculate and return cost
        return (tokens / 1000) * rate
    
    def optimize_prompt(
        self,
        prompt: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Optimize prompt for token efficiency.
        
        Args:
            prompt: Original prompt
            max_length: Maximum prompt length in tokens
            
        Returns:
            Optimized prompt
        """
        # Simple optimization for now - truncate if needed
        if not max_length:
            return prompt
            
        # Estimate token count (4 chars ~= 1 token)
        estimated_tokens = len(prompt) // 4
        if estimated_tokens <= max_length:
            return prompt
            
        # Simple truncation - in a real implementation would be more sophisticated
        truncate_chars = (estimated_tokens - max_length) * 4
        return prompt[:-truncate_chars] 
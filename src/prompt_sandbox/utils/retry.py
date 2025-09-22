"""
Retry logic with exponential backoff for async operations
"""

import asyncio
import time
from typing import Callable, Any
from functools import wraps


class AsyncRetryError(Exception):
    """Raised when all retry attempts are exhausted"""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying async functions with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Usage:
        @retry_with_backoff(max_retries=3)
        async def flaky_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        raise AsyncRetryError(
                            f"Failed after {max_retries} retries: {e}"
                        ) from e

                    print(f"⚠️  Attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                    print(f"   Retrying in {delay:.1f}s...")

                    await asyncio.sleep(delay)
                    delay *= backoff_factor

            # Should never reach here, but just in case
            raise AsyncRetryError(
                f"Unexpected retry failure"
            ) from last_exception

        return wrapper
    return decorator

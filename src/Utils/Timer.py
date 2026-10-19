from collections import defaultdict
from functools import wraps
from time import perf_counter

class Timer:
    def __init__(self):
        self.counters = defaultdict(float)

    @staticmethod
    def measure(counter_name):
        """Decorator that accumulates execution time in self.timer."""

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                start = perf_counter()
                try:
                    return func(self, *args, **kwargs)
                finally:
                    self.timer.counters[counter_name] += perf_counter() - start

            return wrapper

        return decorator

    def reset(self):
        self.counters.clear()

    def report(self):
        for name, elapsed in self.counters.items():
            print(f"{name}: {elapsed:.3f}s")
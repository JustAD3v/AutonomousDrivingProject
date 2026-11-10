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

    def from_second(self, seconds):
        hours = 0
        minutes = 0
        time_str = ""
        if seconds >= 3600:
            hours = seconds // 3600
            time_str += f"{hours:.0f}h "
            seconds %= 3600
        if seconds >= 60:
            minutes = seconds // 60
            time_str += f"{minutes:.0f}min "
            seconds %= 60
        time_str += f"{seconds:.0f}s"
        return time_str
    
    def report(self):
        for name, elapsed in self.counters.items():
            time = self.from_second(elapsed)
            print(f"{name}: {time} ({elapsed:.3f}s)")
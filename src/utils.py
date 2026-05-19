import time


class TokenTimer:
    def __init__(self):
        self.tokens = 0
        self.elapsed = 0.0
        self._start: float | None = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object):
        assert self._start is not None
        self.elapsed += time.perf_counter() - self._start
        self._start = None

    def add_tokens(self, count):
        self.tokens += count

    def to_dict(self):
        elapsed = self.elapsed
        if self._start is not None:
            elapsed += time.perf_counter() - self._start
        return {
            "generated_tokens": self.tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self.tokens / elapsed if elapsed > 0 else 0.0,
        }

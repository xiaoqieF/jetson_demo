from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Callable, Optional


def stamp_to_nanoseconds(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass(frozen=True)
class MatchedFrame:
    stamp_ns: int
    image: Any
    result: Any


class FrameSynchronizer:
    """Bounded exact-timestamp matcher independent of ROS and Qt."""

    def __init__(self, capacity: int = 8, timeout_sec: float = 1.5,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if capacity < 1 or timeout_sec <= 0:
            raise ValueError("capacity and timeout_sec must be positive")
        self.capacity = capacity
        self.timeout_sec = timeout_sec
        self._clock = clock
        self._images: OrderedDict[int, tuple[float, Any]] = OrderedDict()
        self._results: OrderedDict[int, tuple[float, Any]] = OrderedDict()

    def insert_image(self, stamp_ns: int, image: Any) -> Optional[MatchedFrame]:
        return self._insert(stamp_ns, image, self._images, self._results, True)

    def insert_result(self, stamp_ns: int, result: Any) -> Optional[MatchedFrame]:
        return self._insert(stamp_ns, result, self._results, self._images, False)

    def _insert(self, stamp_ns: int, value: Any, own: OrderedDict,
                other: OrderedDict, is_image: bool) -> Optional[MatchedFrame]:
        now = self._clock()
        self.cleanup(now)
        if stamp_ns in other:
            _, counterpart = other.pop(stamp_ns)
            return MatchedFrame(stamp_ns, value, counterpart) if is_image else MatchedFrame(
                stamp_ns, counterpart, value)
        own[stamp_ns] = (now, value)
        own.move_to_end(stamp_ns)
        while len(own) > self.capacity:
            own.popitem(last=False)
        return None

    def cleanup(self, now: Optional[float] = None) -> None:
        cutoff = (self._clock() if now is None else now) - self.timeout_sec
        for cache in (self._images, self._results):
            expired = [key for key, (arrival, _) in cache.items() if arrival < cutoff]
            for key in expired:
                del cache[key]

    @property
    def cache_sizes(self) -> tuple[int, int]:
        return len(self._images), len(self._results)

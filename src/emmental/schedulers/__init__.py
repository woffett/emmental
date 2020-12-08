"""Emmental scheduler module."""
from emmental.schedulers.mixed_scheduler import MixedScheduler
from emmental.schedulers.round_robin_scheduler import RoundRobinScheduler
from emmental.schedulers.sequential_scheduler import SequentialScheduler
from emmental.schedulers.leep_scheduler import LEEPScheduler

SCHEDULERS = {
    "mixed": MixedScheduler,
    "round_robin": RoundRobinScheduler,
    "sequential": SequentialScheduler,
    "leep": LEEPScheduler
}

__all__ = ["MixedScheduler", "RoundRobinScheduler", "SequentialScheduler", "LEEPScheduler"]

"""L0 Kernel — assemble, configure, observe, and keep alive everything else.

**The kernel law.** Every L0 module is depended upon by the flow layers and
depends on none of them. No kernel module knows what a frame, detection, track,
or observation is. This is not stylistic purity: it is what allows the kernel to
be reused unchanged by a future UnityWorks Audio OS or Sensor OS, and what
prevents the kernel from becoming the place where layering rules go to die.

    M15 VisionRuntime          make the platform exist and keep it running
    M16 ConfigurationManager   resolve and validate configuration
    M17 PluginManager          make swappable code loadable and safe
    M19 EventBus               deliver typed notifications
    M20 HealthMonitor          know what is working
    M21 MetricsEngine          count things accurately and cheaply

M18 Model Manager is **not** implemented in Flow 1: its first consumer is the
Detection Engine (Flow 2), and implementing it now would be speculative.
"""

from __future__ import annotations

from .clock import ScaledClock, SystemClock, VirtualClock
from .config import ConfigLayer, ConfigurationManager
from .events import EventBus
from .health import HealthMonitor
from .metrics import MetricName, MetricsEngine
from .plugins import PluginManager, PortCatalogue
from .runtime import RuntimeState, VisionRuntime

__all__ = [
    "ConfigLayer",
    "ConfigurationManager",
    "EventBus",
    "HealthMonitor",
    "MetricName",
    "MetricsEngine",
    "PluginManager",
    "PortCatalogue",
    "RuntimeState",
    "ScaledClock",
    "SystemClock",
    "VirtualClock",
    "VisionRuntime",
]

"""Canonical class IDs and ROS encodings. Cost intent is documentation; Dev 3 inflates."""

# 0 unknown — never free (inflate). Not traversable.
UNKNOWN = 0
# 1 traversable — free / low cost.
TRAVERSABLE = 1
# 2 hazard — lethal / inscribed.
HAZARD = 2

CANONICAL = frozenset({UNKNOWN, TRAVERSABLE, HAZARD})

MASK_ENCODING = "mono8"
CONF_ENCODING = "32FC1"

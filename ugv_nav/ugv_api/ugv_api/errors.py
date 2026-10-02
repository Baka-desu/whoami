"""Errors shared by the ROS side and the HTTP layer (no ROS imports)."""


class ServiceUnavailable(RuntimeError):
    """A ROS service or action server the request needs is not on the graph."""

"""
TEST-ONLY closed-loop scenarios for the Dev 4 stack (not product maps).

Each scenario is a synthetic OccupancyGrid in the `map` frame, following the
Dev 3 -> Dev 4 hand-off test cases (DEV3_DEV4_INTERFACE.md §9): open space, a wall
with an opening, a corridor, an unknown block, and an obstacle that appears on the
path while the robot drives. The robot always starts at (0, 0, yaw 0).

OccupancyGrid values: 0 free, 100 lethal, -1 unknown. All sizes and positions are
test values, not project values.
"""

from dataclasses import dataclass, field

RESOLUTION = 0.05
ORIGIN = (-1.0, -2.5)
SIZE_M = (7.0, 5.0)
WIDTH = round(SIZE_M[0] / RESOLUTION)
HEIGHT = round(SIZE_M[1] / RESOLUTION)

FREE, LETHAL, UNKNOWN = 0, 100, -1


@dataclass(frozen=True)
class Box:
    """Axis-aligned map-frame rectangle [x0, x1] x [y0, y1] in metres."""

    x0: float
    x1: float
    y0: float
    y1: float
    value: int = LETHAL


@dataclass(frozen=True)
class Scenario:
    name: str
    goal: tuple                      # (x, y, yaw) in map
    boxes: tuple = ()                # present from the start
    # Appears once the robot's x passes `trigger_x` (dynamic hazard, arch §11).
    late_boxes: tuple = ()
    trigger_x: float = 0.0
    timeout_s: float = 90.0
    notes: str = field(default='', compare=False)


SCENARIOS = {
    s.name: s for s in (
        Scenario('open', goal=(4.0, 0.0, 0.0), notes='open space'),
        Scenario(
            'wall_gap', goal=(4.0, 0.0, 0.0),
            boxes=(Box(1.9, 2.1, -2.5, 1.0), Box(1.9, 2.1, 1.8, 2.5)),
            notes='wall across the direct path, 0.8 m opening at y 1.0..1.8'),
        Scenario(
            'corridor', goal=(4.0, 0.0, 0.0),
            boxes=(Box(1.0, 3.0, -2.5, -0.6), Box(1.0, 3.0, 0.6, 2.5)),
            notes='only way through is a 1.2 m corridor'),
        Scenario(
            'unknown_block', goal=(4.0, 0.0, 0.0),
            boxes=(Box(1.5, 2.5, -1.0, 1.0, UNKNOWN),),
            notes='unknown on the direct path; unknown != free'),
        Scenario(
            'dynamic_obstacle', goal=(4.0, 0.0, 0.0),
            late_boxes=(Box(2.0, 2.3, -0.7, 0.7),), trigger_x=0.8,
            notes='hazard appears on the path while driving'),
    )
}


def cell_of(x, y):
    return int((x - ORIGIN[0]) / RESOLUTION), int((y - ORIGIN[1]) / RESOLUTION)


def cell_center(i, j):
    return ORIGIN[0] + (i + 0.5) * RESOLUTION, ORIGIN[1] + (j + 0.5) * RESOLUTION


def build_grid(boxes):
    """Row-major OccupancyGrid data (index = j * WIDTH + i)."""
    data = [FREE] * (WIDTH * HEIGHT)
    for box in boxes:
        i0, j0 = cell_of(box.x0, box.y0)
        i1, j1 = cell_of(box.x1 - 1e-9, box.y1 - 1e-9)
        for j in range(max(j0, 0), min(j1, HEIGHT - 1) + 1):
            for i in range(max(i0, 0), min(i1, WIDTH - 1) + 1):
                data[j * WIDTH + i] = box.value
    return data


def cells_with(data, value):
    """Centres of all cells holding `value` (LETHAL or UNKNOWN)."""
    return [cell_center(k % WIDTH, k // WIDTH) for k, v in enumerate(data) if v == value]


def min_clearance(x, y, cells):
    """Distance from (x, y) to the nearest blocked cell centre (inf if none)."""
    return min((((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for cx, cy in cells),
               default=float('inf'))

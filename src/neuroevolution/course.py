"""Immutable 2D courses, exact ray queries, and swept circular collisions.

Coordinates use a Cartesian plane: x points right, y points up. Angles are
radians, with zero facing right and positive rotation counterclockwise.
Touching a wall or the arena boundary counts as a collision.
"""

import math
from dataclasses import dataclass

from neuroevolution._validation import (
    Point,
    finite_float,
    nonnegative_float,
    nonnegative_int,
    point,
    positive_float,
)


def _box_interval(
    origin: Point, direction: Point, bounds: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    """Intersect the two axis slabs; parameters may be distances or fractions."""
    near, far = -math.inf, math.inf
    for start, delta, low, high in (
        (origin[0], direction[0], bounds[0], bounds[2]),
        (origin[1], direction[1], bounds[1], bounds[3]),
    ):
        if delta == 0:
            if not low <= start <= high:
                return None
            continue
        first, last = sorted(((low - start) / delta, (high - start) / delta))
        near, far = max(near, first), min(far, last)
        if near > far:
            return None
    return near, far


def _circle_entry(
    start: Point, end: Point, center: Point, radius: float
) -> float | None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    ox, oy = start[0] - center[0], start[1] - center[1]
    c = ox * ox + oy * oy - radius * radius
    if c <= 0:
        return 0.0
    a = dx * dx + dy * dy
    if a == 0:
        return None
    b = 2 * (ox * dx + oy * dy)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    fraction = (-b - math.sqrt(discriminant)) / (2 * a)
    return fraction if 0 <= fraction <= 1 else None


@dataclass(frozen=True, slots=True)
class Circle:
    center: Point
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", point(self.center, "Circle center"))
        object.__setattr__(self, "radius", positive_float(self.radius, "Circle radius"))

    def contains(self, position: Point) -> bool:
        x, y = point(position)
        return math.hypot(x - self.center[0], y - self.center[1]) <= self.radius

    def entry_fraction(self, start: Point, end: Point) -> float | None:
        """First entry along a movement segment, including tangential contact."""
        return _circle_entry(point(start), point(end), self.center, self.radius)


@dataclass(frozen=True, slots=True)
class Rectangle:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("x", "y"):
            object.__setattr__(self, name, finite_float(getattr(self, name), name))
        for name in ("width", "height"):
            object.__setattr__(self, name, positive_float(getattr(self, name), name))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    def intersects_circle(self, position: Point, radius: float) -> bool:
        x, y = point(position)
        radius = nonnegative_float(radius, "Radius")
        left, bottom, right, top = self.bounds
        nearest_x, nearest_y = min(max(x, left), right), min(max(y, bottom), top)
        return math.hypot(x - nearest_x, y - nearest_y) <= radius

    def collision_fraction(
        self, start: Point, end: Point, radius: float
    ) -> float | None:
        """First contact of a moving circle with this rectangle.

        The expanded collision shape has rounded corners: two rectangular
        strips plus four corner disks. Expanding only a bounding box would
        incorrectly block paths that pass safely beside a corner.
        """
        start, end = point(start), point(end)
        radius = nonnegative_float(radius, "Radius")
        left, bottom, right, top = self.bounds
        direction = (end[0] - start[0], end[1] - start[1])
        contacts = []
        for bounds in (
            (left - radius, bottom, right + radius, top),
            (left, bottom - radius, right, top + radius),
        ):
            interval = _box_interval(start, direction, bounds)
            if interval is not None and interval[1] >= 0 and interval[0] <= 1:
                contacts.append(max(0.0, interval[0]))
        for corner in ((left, bottom), (left, top), (right, bottom), (right, top)):
            fraction = _circle_entry(start, end, corner, radius)
            if fraction is not None:
                contacts.append(fraction)
        return min(contacts) if contacts else None


@dataclass(frozen=True, slots=True)
class Course:
    width: float
    height: float
    spawn: Point
    goal: Circle
    obstacles: tuple[Rectangle, ...] = ()
    checkpoints: tuple[Circle, ...] = ()
    spawn_heading: float = 0.0

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            object.__setattr__(self, name, positive_float(getattr(self, name), name))
        object.__setattr__(self, "spawn", point(self.spawn, "Spawn"))
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        object.__setattr__(
            self, "spawn_heading", finite_float(self.spawn_heading, "Spawn heading")
        )
        for obstacle in self.obstacles:
            left, bottom, right, top = obstacle.bounds
            if left < 0 or bottom < 0 or right > self.width or top > self.height:
                raise ValueError("Obstacles must fit inside the arena.")
        if self.collides_circle(self.spawn, 0):
            raise ValueError("Spawn must be inside the arena and outside obstacles.")
        for region in (self.goal, *self.checkpoints):
            if self.collides_circle(region.center, region.radius):
                raise ValueError("Goal and checkpoint regions must be clear of walls.")

    def collides_circle(self, position: Point, radius: float) -> bool:
        x, y = point(position)
        radius = nonnegative_float(radius, "Radius")
        return (
            x <= radius
            or y <= radius
            or x >= self.width - radius
            or y >= self.height - radius
            or any(wall.intersects_circle((x, y), radius) for wall in self.obstacles)
        )

    def first_collision(self, start: Point, end: Point, radius: float) -> float | None:
        """Return the earliest collision fraction in [0, 1], or None."""
        start, end = point(start), point(end)
        radius = nonnegative_float(radius, "Radius")
        if self.collides_circle(start, radius):
            return 0.0
        direction = end[0] - start[0], end[1] - start[1]
        inner = (radius, radius, self.width - radius, self.height - radius)
        interval = _box_interval(start, direction, inner)
        contacts = []
        if interval is not None and 0 <= interval[1] <= 1:
            contacts.append(interval[1])
        for wall in self.obstacles:
            fraction = wall.collision_fraction(start, end, radius)
            if fraction is not None:
                contacts.append(fraction)
        return min(contacts) if contacts else None

    def raycast(self, origin: Point, angle: float, max_distance: float) -> float:
        """Distance from the ray origin to the nearest wall, capped at its range."""
        origin = point(origin, "Ray origin")
        angle = finite_float(angle, "Ray angle")
        max_distance = positive_float(max_distance, "Ray range")
        if not (0 <= origin[0] <= self.width and 0 <= origin[1] <= self.height):
            raise ValueError("Ray origin must be inside the arena.")
        if self.collides_circle(origin, 0):
            return 0.0
        direction = math.cos(angle), math.sin(angle)
        interval = _box_interval(origin, direction, (0, 0, self.width, self.height))
        distance = min(max_distance, interval[1])
        for wall in self.obstacles:
            hit = _box_interval(origin, direction, wall.bounds)
            if hit is not None and hit[1] >= 0:
                distance = min(distance, max(0.0, hit[0]))
        return distance

    def advance_checkpoints(self, start: Point, end: Point, passed: int) -> int:
        """Count checkpoint entries in order along the actual movement segment."""
        start, end = point(start), point(end)
        passed = nonnegative_int(passed, "Checkpoints passed")
        if passed > len(self.checkpoints):
            raise ValueError("Checkpoints passed exceeds the course checkpoint count.")
        while passed < len(self.checkpoints):
            fraction = self.checkpoints[passed].entry_fraction(start, end)
            if fraction is None:
                break
            start = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            passed += 1
        return passed


def slalom_course() -> Course:
    """A fixed, traversable starting course with three alternating openings."""
    return Course(
        width=900,
        height=600,
        spawn=(70, 300),
        goal=Circle((830, 300), 25),
        obstacles=(
            Rectangle(240, 0, 20, 420),
            Rectangle(440, 180, 20, 420),
            Rectangle(640, 0, 20, 420),
        ),
        checkpoints=(
            Circle((250, 510), 60),
            Circle((450, 90), 60),
            Circle((650, 510), 60),
        ),
    )

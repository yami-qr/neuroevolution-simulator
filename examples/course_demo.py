"""Exercise the course with a scripted reference and an untrained controller."""

import math

from neuroevolution import (
    Action,
    AgentConfig,
    AgentState,
    Course,
    FeedForwardNetwork,
    observe,
    run_episode,
    slalom_course,
    spawn_agent,
    step_agent,
)


def scripted_reference(course: Course, config: AgentConfig) -> AgentState:
    """Follow known waypoints to demonstrate traversability, without learning."""
    waypoints = [
        (198, 300),
        (198, 508),
        (382, 508),
        (382, 92),
        (582, 92),
        (582, 508),
        (742, 508),
        (742, 300),
        course.goal.center,
    ]
    state = spawn_agent(course, config)
    index = 0
    while state.status == "active":
        target = waypoints[index]
        dx, dy = target[0] - state.position[0], target[1] - state.position[1]
        if math.hypot(dx, dy) <= 4 and index < len(waypoints) - 1:
            index += 1
            continue
        error = (math.atan2(dy, dx) - state.heading + math.pi) % math.tau - math.pi
        steering = max(-1.0, min(1.0, error / (config.turn_rate * config.timestep)))
        throttle = 1.0 if abs(error) < 0.05 else 0.0
        state = step_agent(course, state, Action(steering, throttle), config)
    return state


def main() -> None:
    course, config = slalom_course(), AgentConfig()
    observation = observe(course, spawn_agent(course, config), config)
    print(
        f"Course: {course.width:g} x {course.height:g}, {len(course.obstacles)} walls"
    )
    print(f"Initial observation ({observation.size} inputs): {observation.round(3)}")

    reference = scripted_reference(course, config)
    if reference.status != "reached" or reference.checkpoints_passed != 3:
        raise RuntimeError("Scripted course reference failed.")
    print(
        f"Scripted reference: {reference.status}, "
        f"{reference.steps * config.timestep:.2f}s, "
        f"{reference.checkpoints_passed}/3 checkpoints"
    )

    controller = FeedForwardNetwork((10, 12, 2), seed=42)
    result = run_episode(course, controller, agent_config=config)
    print(
        f"Untrained network (seed 42): {result.final_state.status}, "
        f"{result.elapsed_seconds:.2f}s, "
        f"{result.final_state.checkpoints_passed}/3 checkpoints"
    )
    print("No evolution or training has been performed.")


if __name__ == "__main__":
    main()

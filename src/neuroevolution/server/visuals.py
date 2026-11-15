"""Bounded replay inspection and standalone Matplotlib fitness exports."""

import io
import threading
from dataclasses import asdict

from neuroevolution import (
    AgentConfig,
    FeedForwardNetwork,
    FitnessConfig,
    SensorConfig,
    observe,
    run_episode,
    score_episode,
)
from neuroevolution.server.repository import Conflict
from neuroevolution.server.schemas import ExperimentSettings

_PLOT_LOCK = threading.Lock()


def generation_replay(source, *, max_frames=1200):
    settings_data, controller, summary, _, _ = source
    settings = ExperimentSettings.model_validate(settings_data)
    course = settings.course.domain()
    agent = AgentConfig(**settings.agent.model_dump())
    sensors = SensorConfig(**settings.sensors.model_dump())
    network = FeedForwardNetwork.from_dict(controller)
    result = run_episode(
        course, network, agent_config=agent, sensor_config=sensors, record=True
    )
    score = score_episode(
        course,
        result,
        agent_config=agent,
        config=FitnessConfig(**settings.fitness.model_dump()),
    )
    if score.total != summary["fitness"]:
        raise Conflict("Stored controller replay differs from its original fitness.")
    count = len(result.trajectory)
    indices = (
        list(range(count))
        if count <= max_frames
        else [
            round(index * (count - 1) / (max_frames - 1)) for index in range(max_frames)
        ]
    )
    frames = []
    for index in indices:
        state = result.trajectory[index]
        inputs = observe(course, state, agent, sensors)
        frames.append(
            {
                "time": state.steps * agent.timestep,
                "state": asdict(state),
                "inputs": inputs.tolist(),
                "outputs": network.forward(inputs).tolist()
                if state.status == "active"
                else None,
            }
        )
    return {
        "agent": summary,
        "controller": controller,
        "course": asdict(course),
        "agent_config": asdict(agent),
        "sensor_config": asdict(sensors),
        "frames": frames,
        "sampled": len(frames) < count,
        "total_frames": count,
        "elapsed_seconds": result.elapsed_seconds,
        "final_state": asdict(result.final_state),
        "fitness": {**asdict(score), "total": score.total},
    }


def fitness_png(generations):
    if not generations:
        raise Conflict("A saved generation is needed to export a fitness chart.")
    # Matplotlib has shared font/rendering state; serialize threaded exports.
    with _PLOT_LOCK:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator, PercentFormatter

        figure = Figure(
            figsize=(10, 4.8), dpi=150, facecolor="#101516", layout="constrained"
        )
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.set_facecolor("#181e20")
        x = [row.generation for row in generations]
        (first,) = axis.plot(
            x,
            [row.best_fitness for row in generations],
            color="#a2e3ba",
            label="Best fitness",
            linewidth=2,
            marker="o" if len(x) == 1 else None,
        )
        (second,) = axis.plot(
            x,
            [row.mean_fitness for row in generations],
            color="#9badcc",
            label="Mean fitness",
            linewidth=1.5,
            marker="o" if len(x) == 1 else None,
        )
        right = axis.twinx()
        (third,) = right.plot(
            x,
            [row.completed_count / row.population_size for row in generations],
            color="#e6c489",
            label="Completion rate",
            linestyle="--",
            linewidth=1.5,
            marker="o" if len(x) == 1 else None,
        )
        right.set_ylim(0, 1.05)
        right.yaxis.set_major_formatter(PercentFormatter(1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("Generation", color="#b1c1b9")
        axis.set_ylabel("Fitness", color="#b1c1b9")
        right.set_ylabel("Agents reaching the goal", color="#b1c1b9")
        axis.set_title(
            "Neuroevolution · Fitness progression", loc="left", color="#edf2f0", pad=18
        )
        axis.grid(alpha=0.12, color="#899995")
        for panel in (axis, right):
            panel.tick_params(colors="#b1c1b9", labelsize=9)
            for spine in panel.spines.values():
                spine.set_color("#34423c")
        axis.legend(
            handles=[first, second, third],
            facecolor="#181e20",
            edgecolor="#34423c",
            labelcolor="#edf2f0",
            fontsize=8,
            loc="upper left",
        )
        output = io.BytesIO()
        figure.savefig(output, format="png", facecolor=figure.get_facecolor())
        figure.clear()
        return output.getvalue()

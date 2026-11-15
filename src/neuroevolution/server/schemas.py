"""Validated, immutable API settings and response contracts."""

import math
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from neuroevolution import (
    AgentConfig,
    Circle,
    Course,
    EvolutionConfig,
    EvolutionRunner,
    FitnessConfig,
    Rectangle,
    SensorConfig,
    slalom_course,
    spawn_agent,
)
from neuroevolution.network import ActivationName

Seed = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class Schema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
        validate_default=True,
    )


class RectangleSpec(Schema):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    def domain(self) -> Rectangle:
        return Rectangle(**self.model_dump())


class CircleSpec(Schema):
    center: tuple[float, float]
    radius: float = Field(gt=0)

    def domain(self) -> Circle:
        return Circle(self.center, self.radius)


class CourseSpec(Schema):
    width: float = Field(gt=0, le=100000)
    height: float = Field(gt=0, le=100000)
    spawn: tuple[float, float]
    goal: CircleSpec
    obstacles: tuple[RectangleSpec, ...] = Field(default=(), max_length=128)
    checkpoints: tuple[CircleSpec, ...] = Field(default=(), max_length=128)
    spawn_heading: float = 0

    def domain(self) -> Course:
        return Course(
            self.width,
            self.height,
            self.spawn,
            self.goal.domain(),
            tuple(wall.domain() for wall in self.obstacles),
            tuple(region.domain() for region in self.checkpoints),
            self.spawn_heading,
        )


class EvolutionSpec(Schema):
    population_size: int = Field(default=100, strict=True, ge=2, le=1000)
    elite_count: int = Field(default=5, strict=True, ge=0, le=1000)
    tournament_size: int = Field(default=3, strict=True, ge=1, le=1000)
    crossover_rate: float = Field(default=0.8, ge=0, le=1)
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    mutation_scale: float = Field(default=0.2, ge=0, le=10)


class AgentSpec(Schema):
    radius: float = Field(default=8, gt=0, le=100000)
    max_speed: float = Field(default=240, gt=0, le=100000)
    turn_rate: float = Field(default=math.pi, gt=0, le=100)
    timestep: float = Field(default=1 / 60, gt=0, le=1)
    max_steps: int = Field(default=900, strict=True, ge=1, le=36000)


class SensorSpec(Schema):
    ray_angles: tuple[float, ...] = Field(
        default=SensorConfig().ray_angles, min_length=1, max_length=64
    )
    max_distance: float = Field(default=200, gt=0, le=200000)


class FitnessSpec(Schema):
    progress_weight: float = Field(default=1000, ge=0, le=1e9)
    completion_bonus: float = Field(default=1000, ge=0, le=1e9)
    time_penalty: float = Field(default=10, ge=0, le=1e9)
    collision_penalty: float = Field(default=25, ge=0, le=1e9)


class NetworkSpec(Schema):
    layer_sizes: tuple[Annotated[int, Field(strict=True, ge=1, le=256)], ...] = Field(
        default=(10, 12, 2), min_length=2, max_length=6
    )
    hidden_activation: ActivationName = "tanh"
    output_activation: ActivationName = "tanh"


class ExperimentSettings(Schema):
    schema_version: Literal[1] = 1
    seed: Seed = 42
    generations: int = Field(default=30, strict=True, ge=1, le=1000)
    course: CourseSpec = Field(
        default_factory=lambda: CourseSpec.model_validate(asdict(slalom_course()))
    )
    evolution: EvolutionSpec = Field(default_factory=EvolutionSpec)
    agent: AgentSpec = Field(default_factory=AgentSpec)
    sensors: SensorSpec = Field(default_factory=SensorSpec)
    fitness: FitnessSpec = Field(default_factory=FitnessSpec)
    network: NetworkSpec = Field(default_factory=NetworkSpec)

    @model_validator(mode="after")
    def validate_engine_configuration(self):
        EvolutionConfig(**self.evolution.model_dump())
        SensorConfig(**self.sensors.model_dump())
        FitnessConfig(**self.fitness.model_dump())
        spawn_agent(self.course.domain(), AgentConfig(**self.agent.model_dump()))
        if (
            self.network.layer_sizes[0] != len(self.sensors.ray_angles) + 3
            or self.network.layer_sizes[-1] != 2
        ):
            raise ValueError(
                "Network must match the sensor input count and have 2 outputs."
            )
        parameters = sum(
            (inputs + 1) * outputs
            for inputs, outputs in zip(
                self.network.layer_sizes[:-1], self.network.layer_sizes[1:], strict=True
            )
        )
        if parameters * self.evolution.population_size > 5_000_000:
            raise ValueError(
                "Controller population exceeds the supported parameter budget."
            )
        return self

    def runner(self) -> EvolutionRunner:
        return EvolutionRunner(
            self.course.domain(),
            seed=self.seed,
            config=EvolutionConfig(**self.evolution.model_dump()),
            agent_config=AgentConfig(**self.agent.model_dump()),
            sensor_config=SensorConfig(**self.sensors.model_dump()),
            fitness_config=FitnessConfig(**self.fitness.model_dump()),
            **self.network.model_dump(),
        )


class ExperimentCreate(Schema):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    settings: ExperimentSettings = Field(default_factory=ExperimentSettings)


class RunCreate(Schema):
    seed: Seed | None = None
    generations: Annotated[int, Field(strict=True, ge=1, le=1000)] | None = None


class ReadSchema(Schema):
    model_config = ConfigDict(from_attributes=True)


class ExperimentRead(ReadSchema):
    id: UUID
    name: str
    description: str
    settings: ExperimentSettings
    created_at: datetime


class RunRead(ReadSchema):
    id: UUID
    experiment_id: UUID
    settings: ExperimentSettings
    status: RunStatus
    requested_generations: PositiveInt
    completed_generations: int
    cancel_requested: bool
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    environment: dict[str, Any] | None
    error: str | None
    champion_fitness: float | None
    champion_generation: int | None

    @computed_field
    @property
    def progress(self) -> float:
        return self.completed_generations / self.requested_generations


class GenerationRead(ReadSchema):
    generation: int
    population_size: PositiveInt
    best_fitness: float
    mean_fitness: float
    completed_count: int
    best_progress: float
    best_checkpoints: int
    created_at: datetime

    @computed_field
    @property
    def completion_rate(self) -> float:
        return self.completed_count / self.population_size


class ChampionRead(Schema):
    generation: int
    fitness: float
    controller: dict[str, Any]
    replay_available: bool


Item = TypeVar("Item")


class Page(Schema, Generic[Item]):
    items: list[Item]
    total: int
    offset: int
    limit: int

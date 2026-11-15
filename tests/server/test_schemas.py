"""Configuration round-trips and domain-level request validation."""

import json

import pytest
from pydantic import ValidationError

from neuroevolution.server.schemas import ExperimentCreate, ExperimentSettings


def test_complete_defaults_roundtrip_through_database_json():
    settings = ExperimentSettings()
    restored = ExperimentSettings.model_validate(json.loads(settings.model_dump_json()))
    assert restored == settings
    assert settings.course.domain().spawn == (70, 300)
    assert settings.network.layer_sizes == (10, 12, 2)


def test_valid_settings_build_the_existing_engine():
    settings = ExperimentSettings.model_validate(
        {
            "generations": 1,
            "evolution": {"population_size": 2, "elite_count": 1, "tournament_size": 2},
            "agent": {"max_steps": 1},
        }
    )
    assert settings.runner().step().stats.population_size == 2


@pytest.mark.parametrize(
    "settings",
    [
        {"seed": True},
        {"seed": -1},
        {"generations": 0},
        {"generations": 1.5},
        {"unknown": 1},
        {"evolution": {"unknown": 1}},
        {"evolution": {"population_size": 2}},
        {"evolution": {"mutation_rate": 2}},
        {"fitness": {"time_penalty": float("nan")}},
        {"network": {"layer_sizes": [9, 12, 2]}},
        {"network": {"layer_sizes": [10, 12, 3]}},
        {"agent": {"radius": 100}},
        {"agent": {"max_steps": 0}},
    ],
)
def test_invalid_or_incompatible_settings_are_rejected(settings):
    with pytest.raises(ValidationError):
        ExperimentSettings.model_validate(settings)


def test_names_are_trimmed_and_blank_names_rejected():
    assert ExperimentCreate(name="  First run  ").name == "First run"
    with pytest.raises(ValidationError):
        ExperimentCreate(name="  ")

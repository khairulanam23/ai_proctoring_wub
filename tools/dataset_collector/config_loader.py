"""Configuration loader and resource resolver for Dataset Collector."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("dataset_collector.config_loader")


@dataclass
class ActivityDefinition:
    """Specification of a single reproducible dataset collection activity."""

    id: str
    name: str
    category: str
    instructed_condition: str
    physical_action: str
    visible_state: str
    photo_1: str
    photo_2: str
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "instructed_condition": self.instructed_condition,
            "physical_action": self.physical_action,
            "visible_state": self.visible_state,
            "photo_1": self.photo_1,
            "photo_2": self.photo_2,
            "purpose": self.purpose,
        }


@dataclass
class ActivitiesConfig:
    """Container for the complete collection protocol."""

    version: str
    title: str
    photos_per_activity: int
    activities: list[ActivityDefinition]
    source_file: Path | None = None

    def get_activity_by_id(self, activity_id: str) -> ActivityDefinition | None:
        for a in self.activities:
            if a.id.upper() == activity_id.upper():
                return a
        return None


def resolve_resource_path(relative_path: str | Path) -> Path:
    """Resolves a file path across PyInstaller bundled, installed, and development environments.

    Search order:
      1. Next to the executing binary/script (sys.executable or sys.argv[0])
      2. Inside PyInstaller extraction directory (sys._MEIPASS)
      3. Repository root relative to this module
      4. Current working directory
    """
    rel = Path(relative_path)
    if rel.is_absolute() and rel.exists():
        return rel

    candidates: list[Path] = []

    # 1. Directory of the running executable
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / rel)
        if len(sys.argv) > 0 and sys.argv[0]:
            candidates.append(Path(sys.argv[0]).resolve().parent / rel)

    # 2. PyInstaller temporary bundle directory (_MEIPASS)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        candidates.append(Path(meipass) / rel)

    # 3. Source repository root relative to this file
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent
    candidates.append(repo_root / rel)

    # 4. Current working directory
    candidates.append(Path.cwd() / rel)

    for c in candidates:
        if c.exists():
            return c

    # Fall back to candidate 0 or rel
    return candidates[0] if candidates else rel


def load_activities_config(config_path: str | Path | None = None) -> ActivitiesConfig:
    """Loads and validates the activities.yaml configuration file."""
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            path = resolve_resource_path(config_path)
    else:
        path = resolve_resource_path("configs/capture_scenarios/activities.yaml")

    if not path.is_file():
        raise FileNotFoundError(
            f"Activities configuration file not found at: {path}. "
            "Ensure 'configs/capture_scenarios/activities.yaml' is located next to the executable."
        )

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {path}: expected mapping.")

    version = str(data.get("version", "1.0.0")).strip()
    title = str(data.get("title", "Dataset Collection Protocol")).strip()
    photos_per_act = int(data.get("photos_per_activity", 2))

    raw_activities = data.get("activities", [])
    if not isinstance(raw_activities, list) or not raw_activities:
        raise ValueError(f"Configuration {path} contains no activities.")

    activities: list[ActivityDefinition] = []
    for i, item in enumerate(raw_activities):
        if not isinstance(item, dict):
            raise ValueError(f"Activity {i} in {path} must be a dictionary.")

        act_id = str(item.get("id", f"ACT{i+1:03d}")).strip()
        name = str(item.get("name", act_id)).strip()
        category = str(item.get("category", "GENERAL")).strip().upper()
        condition = str(item.get("instructed_condition", act_id)).strip()
        action = str(item.get("physical_action", "")).strip()
        visible = str(item.get("visible_state", "")).strip()
        p1 = str(item.get("photo_1", "Photo 1")).strip()
        p2 = str(item.get("photo_2", "Photo 2")).strip()
        purpose = str(item.get("purpose", "")).strip()

        activities.append(
            ActivityDefinition(
                id=act_id,
                name=name,
                category=category,
                instructed_condition=condition,
                physical_action=action,
                visible_state=visible,
                photo_1=p1,
                photo_2=p2,
                purpose=purpose,
            )
        )

    LOGGER.info(f"Loaded {len(activities)} activities from {path} (version {version})")
    return ActivitiesConfig(
        version=version,
        title=title,
        photos_per_activity=photos_per_act,
        activities=activities,
        source_file=path,
    )

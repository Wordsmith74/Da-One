import json
import os
from datetime import datetime
from typing import Any

from core.time_utils import is_in_future, is_within_hours, localize_utc, now_utc


class MissingMetricError(Exception):
    """Raised when game_data is missing a required metric for the sport."""
    pass


class UnsupportedSportError(Exception):
    """Raised when an unsupported sport_type is provided."""
    pass


class DecisionOrchestrator:
    """
    Brain of the multi-sport prediction engine.

    Loads sport configuration from config/sports_metrics.json and uses
    the defined weights to compute weighted spread and total projections
    from raw game data.
    """

    CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "..", "config", "sports_metrics.json"
    )

    def __init__(self, sport_type: str) -> None:
        """
        Initialize the orchestrator for a specific sport.

        Args:
            sport_type: One of 'WNBA', 'NBA', or 'MLB'.

        Raises:
            UnsupportedSportError: If sport_type is not present in the config.
            FileNotFoundError: If sports_metrics.json cannot be located.
        """
        with open(self.CONFIG_PATH, "r") as f:
            all_config: dict[str, Any] = json.load(f)

        sport_type = sport_type.upper()
        if sport_type not in all_config:
            supported = ", ".join(all_config.keys())
            raise UnsupportedSportError(
                f"Sport '{sport_type}' is not configured. "
                f"Supported sports: {supported}"
            )

        self.sport_type = sport_type
        self.config: dict[str, Any] = all_config[sport_type]
        self.spread_weights: dict[str, float] = self.config["spread_weights"]
        self.total_weights: dict[str, float] = self.config["total_weights"]
        self.required_metrics: list[str] = self.config["required_metrics"]

    # ── UTC-aware time helpers ────────────────────────────────────────────────

    @property
    def current_time_utc(self) -> datetime:
        """Current moment as a UTC-aware datetime. Always use this for
        time comparisons — never datetime.now() which is naive and breaks
        across DST boundaries."""
        return now_utc()

    def is_game_upcoming(self, game_time_utc: datetime) -> bool:
        """
        Return True if game_time_utc is strictly in the future.

        Both sides of the comparison are UTC-aware, so DST transitions
        cannot cause the result to flip unexpectedly.

        Args:
            game_time_utc: Game start time. Naive datetimes are assumed UTC.
        """
        return is_in_future(localize_utc(game_time_utc))

    def is_game_within_window(self, game_time_utc: datetime, hours: float = 24.0) -> bool:
        """
        Return True if the game starts between now and now + hours (UTC).

        Use this to filter the slate to only actionable games before
        running calculate_true_spread / calculate_true_total.

        Args:
            game_time_utc: Game start time. Naive datetimes are assumed UTC.
            hours:         Look-ahead window in hours. Default 24.
        """
        return is_within_hours(localize_utc(game_time_utc), hours)

    def _validate_required_metrics(self, game_data: dict[str, Any]) -> None:
        """
        Verify that game_data contains all required metrics for this sport.

        Args:
            game_data: Dictionary of metric name -> value.

        Raises:
            MissingMetricError: On the first required metric not found in game_data.
        """
        for metric in self.required_metrics:
            if metric not in game_data:
                raise MissingMetricError(
                    f"[{self.sport_type}] Required metric '{metric}' is missing "
                    f"from the provided game_data. "
                    f"All required metrics: {self.required_metrics}"
                )

    def calculate_true_spread(self, game_data: dict[str, Any]) -> float:
        """
        Compute the weighted spread projection for a game.

        Iterates over the sport's spread_weights and multiplies each weight
        by the corresponding value in game_data, producing a single weighted score.

        Args:
            game_data: Dictionary mapping metric/factor names to numeric values.
                       Must contain all keys listed in required_metrics.

        Returns:
            A float representing the weighted spread projection.

        Raises:
            MissingMetricError: If any required metric is absent from game_data.
            KeyError: If a spread_weight key has no corresponding value in game_data.
        """
        self._validate_required_metrics(game_data)

        weighted_spread = 0.0
        for factor, weight in self.spread_weights.items():
            if factor not in game_data:
                raise KeyError(
                    f"[{self.sport_type}] Spread factor '{factor}' is not present "
                    f"in game_data. Provide a numeric value for each spread weight key."
                )
            weighted_spread += weight * float(game_data[factor])

        return round(weighted_spread, 4)

    def calculate_true_total(self, game_data: dict[str, Any]) -> float:
        """
        Compute the weighted total (over/under) projection for a game.

        Iterates over the sport's total_weights and multiplies each weight
        by the corresponding value in game_data, producing a single weighted score.

        Args:
            game_data: Dictionary mapping metric/factor names to numeric values.
                       Must contain all keys listed in required_metrics.

        Returns:
            A float representing the weighted total projection.

        Raises:
            MissingMetricError: If any required metric is absent from game_data.
            KeyError: If a total_weight key has no corresponding value in game_data.
        """
        self._validate_required_metrics(game_data)

        weighted_total = 0.0
        for factor, weight in self.total_weights.items():
            if factor not in game_data:
                raise KeyError(
                    f"[{self.sport_type}] Total factor '{factor}' is not present "
                    f"in game_data. Provide a numeric value for each total weight key."
                )
            weighted_total += weight * float(game_data[factor])

        return round(weighted_total, 4)

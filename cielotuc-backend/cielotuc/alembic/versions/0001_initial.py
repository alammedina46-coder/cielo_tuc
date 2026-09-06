"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── zones ──────────────────────────────────────────────────
    op.create_table(
        "zones",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("department", sa.String(100), nullable=False),
        sa.Column("zone_type", sa.String(30), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("altitude_m", sa.Float(), nullable=True),
        sa.Column("impermeable_pct", sa.Float(), nullable=True),
        sa.Column("population", sa.Integer(), nullable=True),
        sa.Column("is_mountain", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_zones_name", "zones", ["name"], unique=False)
    op.create_index("ix_zones_id", "zones", ["id"], unique=False)

    # ── weather_stations ───────────────────────────────────────
    op.create_table(
        "weather_stations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("zone_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("station_code", sa.String(20), nullable=False),
        sa.Column("source", sa.String(30), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("altitude_m", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("battery_pct", sa.Float(), nullable=True),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weather_stations_station_code", "weather_stations", ["station_code"], unique=True)
    op.create_index("ix_weather_stations_id", "weather_stations", ["id"], unique=False)

    # ── sensor_readings ────────────────────────────────────────
    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("station_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("temperature_min_c", sa.Float(), nullable=True),
        sa.Column("temperature_max_c", sa.Float(), nullable=True),
        sa.Column("feels_like_c", sa.Float(), nullable=True),
        sa.Column("pressure_hpa", sa.Float(), nullable=True),
        sa.Column("pressure_sea_level_hpa", sa.Float(), nullable=True),
        sa.Column("humidity_pct", sa.Float(), nullable=True),
        sa.Column("dew_point_c", sa.Float(), nullable=True),
        sa.Column("precip_mm", sa.Float(), nullable=True),
        sa.Column("precip_3h_mm", sa.Float(), nullable=True),
        sa.Column("precip_6h_mm", sa.Float(), nullable=True),
        sa.Column("precip_24h_mm", sa.Float(), nullable=True),
        sa.Column("wind_speed_kmh", sa.Float(), nullable=True),
        sa.Column("wind_direction_deg", sa.Float(), nullable=True),
        sa.Column("wind_gust_kmh", sa.Float(), nullable=True),
        sa.Column("visibility_km", sa.Float(), nullable=True),
        sa.Column("cloud_cover_pct", sa.Float(), nullable=True),
        sa.Column("uv_index", sa.Float(), nullable=True),
        sa.Column("cape_j_kg", sa.Float(), nullable=True),
        sa.Column("k_index", sa.Float(), nullable=True),
        sa.Column("ndvi", sa.Float(), nullable=True),
        sa.Column("soil_temp_c", sa.Float(), nullable=True),
        sa.Column("precipitable_water_mm", sa.Float(), nullable=True),
        sa.Column("cordillera_pressure_hpa", sa.Float(), nullable=True),
        sa.Column("thermal_differential_c", sa.Float(), nullable=True),
        sa.Column("enso_index", sa.Float(), nullable=True),
        sa.Column("is_validated", sa.Boolean(), nullable=True),
        sa.Column("quality_flag", sa.String(10), nullable=True),
        sa.ForeignKeyConstraint(["station_id"], ["weather_stations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sensor_readings_station_id", "sensor_readings", ["station_id"], unique=False)
    op.create_index("ix_sensor_readings_timestamp", "sensor_readings", ["timestamp"], unique=False)
    op.create_index("ix_sensor_readings_id", "sensor_readings", ["id"], unique=False)

    # ── weather_events ─────────────────────────────────────────
    op.create_table(
        "weather_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("zone_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(30), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("peak_precip_mm", sa.Float(), nullable=True),
        sa.Column("peak_wind_kmh", sa.Float(), nullable=True),
        sa.Column("peak_temp_c", sa.Float(), nullable=True),
        sa.Column("was_predicted", sa.Boolean(), nullable=True),
        sa.Column("prediction_lead_hours", sa.Float(), nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weather_events_event_type", "weather_events", ["event_type"], unique=False)
    op.create_index("ix_weather_events_id", "weather_events", ["id"], unique=False)

    # ── model_versions ─────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("mlflow_run_id", sa.String(64), nullable=True),
        sa.Column("architecture", sa.String(50), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("training_samples", sa.Integer(), nullable=True),
        sa.Column("training_start_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("training_end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accuracy_overall", sa.Float(), nullable=True),
        sa.Column("precision_rain", sa.Float(), nullable=True),
        sa.Column("recall_rain", sa.Float(), nullable=True),
        sa.Column("rmse_temperature", sa.Float(), nullable=True),
        sa.Column("avg_lead_time_hours", sa.Float(), nullable=True),
        sa.Column("false_positive_rate", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_versions_version", "model_versions", ["version"], unique=True)
    op.create_index("ix_model_versions_id", "model_versions", ["id"], unique=False)
    op.create_index("ix_model_versions_mlflow_run_id", "model_versions", ["mlflow_run_id"], unique=True)

    # ── ai_predictions ─────────────────────────────────────────
    op.create_table(
        "ai_predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("zone_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("target_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_hours", sa.Integer(), nullable=False),
        sa.Column("rain_probability", sa.Float(), nullable=False),
        sa.Column("precip_mm_expected", sa.Float(), nullable=True),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("temperature_min_c", sa.Float(), nullable=True),
        sa.Column("temperature_max_c", sa.Float(), nullable=True),
        sa.Column("wind_speed_kmh", sa.Float(), nullable=True),
        sa.Column("condition", sa.String(30), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("was_confirmed", sa.Boolean(), nullable=True),
        sa.Column("actual_precip_mm", sa.Float(), nullable=True),
        sa.Column("zonda_risk", sa.Float(), nullable=True),
        sa.Column("hail_risk", sa.Float(), nullable=True),
        sa.Column("storm_risk", sa.Float(), nullable=True),
        sa.Column("heatwave_risk", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_predictions_generated_at", "ai_predictions", ["generated_at"], unique=False)
    op.create_index("ix_ai_predictions_zone_id", "ai_predictions", ["zone_id"], unique=False)
    op.create_index("ix_ai_predictions_target_time", "ai_predictions", ["target_time"], unique=False)
    op.create_index("ix_ai_predictions_id", "ai_predictions", ["id"], unique=False)

    # ── flood_alerts ───────────────────────────────────────────
    op.create_table(
        "flood_alerts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("zone_id", sa.Integer(), nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=True),
        sa.Column("triggered_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("trigger_type", sa.String(20), nullable=True),
        sa.Column("rain_probability", sa.Float(), nullable=False),
        sa.Column("expected_precip_mm", sa.Float(), nullable=True),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("floodtuc_response_code", sa.Integer(), nullable=True),
        sa.Column("floodtuc_response_body", sa.Text(), nullable=True),
        sa.Column("delivered", sa.Boolean(), nullable=True),
        sa.Column("outcome", sa.String(30), nullable=True),
        sa.ForeignKeyConstraint(["prediction_id"], ["ai_predictions.id"]),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_flood_alerts_id", "flood_alerts", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_flood_alerts_id", table_name="flood_alerts")
    op.drop_table("flood_alerts")
    op.drop_index("ix_ai_predictions_id", table_name="ai_predictions")
    op.drop_index("ix_ai_predictions_target_time", table_name="ai_predictions")
    op.drop_index("ix_ai_predictions_zone_id", table_name="ai_predictions")
    op.drop_index("ix_ai_predictions_generated_at", table_name="ai_predictions")
    op.drop_table("ai_predictions")
    op.drop_index("ix_model_versions_mlflow_run_id", table_name="model_versions")
    op.drop_index("ix_model_versions_id", table_name="model_versions")
    op.drop_index("ix_model_versions_version", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("ix_weather_events_id", table_name="weather_events")
    op.drop_index("ix_weather_events_event_type", table_name="weather_events")
    op.drop_table("weather_events")
    op.drop_index("ix_sensor_readings_id", table_name="sensor_readings")
    op.drop_index("ix_sensor_readings_timestamp", table_name="sensor_readings")
    op.drop_index("ix_sensor_readings_station_id", table_name="sensor_readings")
    op.drop_table("sensor_readings")
    op.drop_index("ix_weather_stations_id", table_name="weather_stations")
    op.drop_index("ix_weather_stations_station_code", table_name="weather_stations")
    op.drop_table("weather_stations")
    op.drop_index("ix_zones_id", table_name="zones")
    op.drop_index("ix_zones_name", table_name="zones")
    op.drop_table("zones")
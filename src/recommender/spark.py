"""Shared construction of a deterministic local Spark session."""

from __future__ import annotations

import sys

from pyspark.sql import SparkSession

from recommender.config import SparkConfig


def create_spark_session(config: SparkConfig) -> SparkSession:
    """Create Spark with project-wide timezone and shuffle settings."""
    return (
        SparkSession.builder.appName(config.app_name)
        .master(config.master)
        .config("spark.driver.memory", config.driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )

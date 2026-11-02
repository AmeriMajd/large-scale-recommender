"""Explicit schemas for MovieLens 25M source files."""

from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

RATINGS_SCHEMA = StructType(
    [
        StructField("userId", LongType(), nullable=False),
        StructField("movieId", LongType(), nullable=False),
        StructField("rating", DoubleType(), nullable=False),
        StructField("timestamp", LongType(), nullable=False),
    ]
)

MOVIES_SCHEMA = StructType(
    [
        StructField("movieId", LongType(), nullable=False),
        StructField("title", StringType(), nullable=False),
        StructField("genres", StringType(), nullable=False),
    ]
)

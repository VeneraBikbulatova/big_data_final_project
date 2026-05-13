"""Custom PySpark ML Transformer for cyclical feature encoding.

This module exposes :class:`CyclicalEncoder`, a serializable transformer
that decomposes a periodic numeric feature into a ``(sin, cos)`` pair so
that values at the boundary of the cycle (e.g. ``hour=23`` and ``hour=0``)
remain close in feature space.

The transformer follows the standard ``pyspark.ml`` conventions
(``HasInputCol``, ``DefaultParamsReadable``/``Writable``) and may be
persisted as a stage of a :class:`pyspark.ml.PipelineModel`.

Typical usage::

    encoder = CyclicalEncoder(inputCol="hour", period=24.0)
    transformed = encoder.transform(input_dataframe)
"""
import math

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as spark_functions
from pyspark.sql.dataframe import DataFrame


class CyclicalEncoder(
        Transformer,
        HasInputCol,
        DefaultParamsReadable,
        DefaultParamsWritable):
    """Encode a periodic numeric column with sine and cosine.

    For an input column ``x`` with period ``p`` the transformer appends
    two new columns to the input DataFrame:

        * ``<x>_sin = sin(2 * pi * x / p)``
        * ``<x>_cos = cos(2 * pi * x / p)``

    Attributes:
        period: Length of one full cycle. Typical values are ``24.0`` for
            hour-of-day, ``7.0`` for day-of-week, or ``12.0`` for
            month-of-year.
    """

    period = Param(
        Params._dummy(),
        "period",
        "Length of one full cycle (e.g. 24 for hour-of-day).",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(self, inputCol=None, period=24.0):
        """Initialise the encoder.

        Args:
            inputCol: Name of the periodic numeric column to encode.
            period: Length of one full cycle. Defaults to ``24.0``.
        """
        super().__init__()
        self._setDefault(period=24.0)
        keyword_arguments = self._input_kwargs
        self.setParams(**keyword_arguments)

    @keyword_only
    def setParams(self, inputCol=None, period=24.0):
        """Set transformer parameters via keyword arguments.

        Args:
            inputCol: Name of the periodic numeric column to encode.
            period: Length of one full cycle.

        Returns:
            CyclicalEncoder: The transformer instance, for chaining.
        """
        keyword_arguments = self._input_kwargs
        return self._set(**keyword_arguments)

    def set_period(self, value: float) -> "CyclicalEncoder":
        """Set the period parameter.

        Args:
            value: New period as a float.

        Returns:
            CyclicalEncoder: The transformer instance, for chaining.
        """
        return self._set(period=value)

    def get_period(self) -> float:
        """Return the currently configured period.

        Returns:
            float: The period value.
        """
        return self.getOrDefault(self.period)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """Apply the cyclical encoding to the configured input column.

        Args:
            dataset: Input DataFrame that contains the configured input
                column.

        Returns:
            DataFrame: The input DataFrame with two new columns appended:
            ``<inputCol>_sin`` and ``<inputCol>_cos``.
        """
        column_name = self.getInputCol()
        period_value = float(self.get_period())
        angle_expression = (
            2.0 * math.pi * spark_functions.col(column_name) / period_value
        )
        return (
            dataset
            .withColumn(
                f"{column_name}_sin", spark_functions.sin(angle_expression)
            )
            .withColumn(
                f"{column_name}_cos", spark_functions.cos(angle_expression)
            )
        )

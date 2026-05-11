import math

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F


class CyclicalEncoder(Transformer, HasInputCol,
                      DefaultParamsReadable, DefaultParamsWritable):
    period = Param(
        Params._dummy(),
        "period",
        "Length of one cycle (e.g. 24 for hour, 7 for day-of-week).",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(self, inputCol=None, period=24.0):
        super().__init__()
        self._setDefault(period=24.0)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, period=24.0):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setPeriod(self, value):
        return self._set(period=value)

    def getPeriod(self):
        return self.getOrDefault(self.period)

    def _transform(self, df):
        col_name = self.getInputCol()
        period = float(self.getPeriod())
        angle = (2.0 * math.pi * F.col(col_name)) / period
        return (df
                .withColumn(f"{col_name}_sin", F.sin(angle))
                .withColumn(f"{col_name}_cos", F.cos(angle)))
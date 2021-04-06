import typing as t
from msilib.schema import Binary

import typing_extensions as te
from pydantic import BaseModel
from pydantic import ConstrainedInt
from pydantic import NonNegativeInt


class YearInteger(ConstrainedInt):
    ge = 1800
    le = 2020


class MonthInteger(ConstrainedInt):
    ge = 1
    le = 12


class YearInteger(ConstrainedInt):
    ge = 1800
    le = 2020


class SeasonInteger(ConstrainedInt):
    ge = 1
    le = 4


class WeatherInteger(ConstrainedInt):
    ge = 1
    le = 4


class HourInteger(ConstrainedInt):
    ge = 0
    le = 23


class BinaryInteger(ConstrainedInt):
    ge = 0
    le = 1


class WeekdayInteger(ConstrainedInt):
    ge = 1
    le = 7


# Nota: necesario buscar una forma de hacer esto de forma automática según como se entrenó el modelo!
class ModelInput(BaseModel):
    season: SeasonInteger
    yr: YearInteger
    mnth: MonthInteger
    hr: HourInteger
    holiday: BinaryInteger
    weekday: WeekdayInteger
    workingday: BinaryInteger
    weathersit: WeatherInteger
    temp: float
    atemp: float
    hum: float
    windspeed: float
    cnt_1_hours: NonNegativeInt
    cnt_2_hours: NonNegativeInt
    cnt_3_hours: NonNegativeInt
    cnt_7_days: NonNegativeInt
    cnt_last_hour_diff: int
    holiday_7_days: BinaryInteger
    workingday_7_days: BinaryInteger
    temp_1_hours: float
    temp_2_hours: float
    temp_3_hours: float
    atemp_1_hours: float
    atemp_2_hours: float
    atemp_3_hours: float
    hum_1_hours: float
    hum_2_hours: float
    hum_3_hours: float
    windspeed_1_hours: float
    windspeed_2_hours: float
    windspeed_3_hours: float

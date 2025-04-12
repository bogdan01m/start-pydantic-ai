from pydantic import BaseModel
from typing import Optional


class CurrencyModel(BaseModel):
    currency: str
    value: float
    description: str


class WeatherModel(BaseModel):
    temperature: float
    wind_speed: float
    description: str


class JokeModel(BaseModel):
    joke: str


class SupervisorModel(BaseModel):
    weather: Optional[WeatherModel] = None
    currency: Optional[CurrencyModel] = None
    joke: Optional[JokeModel] = None

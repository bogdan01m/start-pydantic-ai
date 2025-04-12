from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
import logfire
import os
from dotenv import load_dotenv

from prompts import weather_prompt, currency_prompt, joke_prompt, supervisor_prompt
from schemas import WeatherModel, CurrencyModel, JokeModel

load_dotenv()


gemini = GeminiModel(
    model_name="gemini-2.0-flash",
    provider=GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY")),
)

logfire.configure(send_to_logfire=os.getenv("LOGFIRE_TOKEN"))


currency_agent = Agent(
    model=gemini,
    tools=[duckduckgo_search_tool()],
    retries=5,
    instrument=True,
    result_type=CurrencyModel,
    system_prompt=currency_prompt,
)

weather_agent = Agent(
    model=gemini,
    tools=[duckduckgo_search_tool()],
    retries=5,
    instrument=True,
    result_type=WeatherModel,
    system_prompt=weather_prompt,
)

joke_agent = Agent(
    model=gemini,
    retries=5,
    instrument=True,
    result_type=JokeModel,
    system_prompt=joke_prompt,
    model_settings={
        "temperature": 0.9,
    },
)


async def get_currency() -> CurrencyModel:
    currency = await currency_agent.run("Пожалуйста, скажи мне курс валют.")
    return currency.data


async def get_weather() -> WeatherModel:
    weather = await weather_agent.run("Какая сейчас погода?")
    return weather.data


async def generate_joke() -> str:
    joke = await joke_agent.run("Расскажи анекдот.")
    return joke.data


supervisor_agent = Agent(
    model=gemini,
    retries=5,
    instrument=True,
    tools=[get_currency, get_weather, generate_joke],
    system_prompt=supervisor_prompt,
)

import asyncio


async def main():
    result = await supervisor_agent.run("Штука про Крота и черепаху")
    print(result.data)


asyncio.run(main())

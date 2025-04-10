from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic import BaseModel
import logfire
import os
from dotenv import load_dotenv

load_dotenv()

system_prompt = """
Ты агент по погоде. Твоя задача предоставить temperature, wind_speed и description
в поле description ты должен дать совет пользователю о том, во что ему одеться 
думай о комфорте человека в предоставленных тебе погодных условиях
"""


class WeatherModel(BaseModel):
    temperature: float
    wind_speed: float
    description: str


logfire.configure(send_to_logfire=os.getenv("LOGFIRE_TOKEN"))
model = MistralModel(
    model_name="mistral-large-latest",
    provider=MistralProvider(api_key=os.getenv("MISTRAL_API_KEY")),
)
agent = Agent(
    model=model,
    tools=[duckduckgo_search_tool()],
    retries=5,
    instrument=True,
    result_type=WeatherModel,
    system_prompt=system_prompt,
)

result_sync = agent.run_sync("Какая температура сейчас в Москве?")
print(result_sync.data)

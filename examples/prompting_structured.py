from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider
from pydantic import BaseModel
import logfire
import os
from dotenv import load_dotenv

load_dotenv()


class Cityname(BaseModel):
    city: str
    description: str


logfire.configure(send_to_logfire=os.getenv("LOGFIRE_TOKEN"))
model = MistralModel(
    model_name="mistral-large-latest",
    provider=MistralProvider(api_key=os.getenv("MISTRAL_API_KEY")),
)
agent = Agent(
    model=model,
    retries=5,
    instrument=True,
    result_type=Cityname,
    system_prompt="Ты CityName Agent, твоя задача ответить на вопрос о столице, а также добавить краткое описание этого города",
)

result_sync = agent.run_sync("Столица Гондураса?")
print(result_sync.data)

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import logfire
import os
from dotenv import load_dotenv

load_dotenv()

flight_prompt = """
Ты агент по поиску рейсов. Ты должен находить ТОЧНЫЕ данные о рейсах.

### Правила:
1. Используй ТОЛЬКО эти сайты: yandextravel.ru, aviasales.ru, ozontravel.ru, tutu.ru
2. Если не находишь рейс - скажи об этом честно
3. Все поля должны быть заполнены ТОЧНЫМИ данными
4. Время указывай в формате: YYYY-MM-DD HH:MM
5. Ссылка должна вести НАПРЯМУЮ на конкретный рейс

### Требуемые данные:
{{  
    "flight_company": str = Field(description="Компания которая организует перелет")
    "flight_number":str = Field(description="Номер рейса")
    "departure_time": datetime = Field(description="Дата и время вылета в локальном времени аэропорта")
    "arrival_time": datetime = Field(description="Дата и время прилёта в локальном времени аэропорта")
    "price": float = Field(description="Цена за перелет за одного человека")
    "price_total": float = Field(description="Цена за перелет суммарно за N человек")
    "persons": int = Field(description="количество человек")
    "original": str = Field(description="Город вылета , название аэропорта")
    "destination": str = Field(description="Город назначения, название аэропорта")
    "plane": Optional[str] = Field(description="Название самолета")
    "link": str = Field(description="Ссылка на конкретный рейс, который ты предлагаешь. Обязательно должно открыть сам рейс")
    "description": str = Field(description="Описание пути c локальным временем вылета, прилета, ценой и остальными параметрами выше")
}}

Исходя из того маршрута, который он тебе предоставил
"""


class FlightModel(BaseModel):
    flight_company: str = Field(description="Компания которая организует перелет")
    flight_number: str = Field(description="Номер рейса")
    departure_time: datetime = Field(
        description="Дата и время вылета в локальном времени аэропорта"
    )
    arrival_time: datetime = Field(
        description="Дата и время прилёта в локальном времени аэропорта"
    )
    price: float = Field(description="Цена за перелет за одного человека")
    price_total: float = Field(description="Цена за перелет суммарно за N человек")
    persons: int = Field(description="количество человек")
    original: str = Field(description="Город вылета , название аэропорта")
    destination: str = Field(description="Город назначения, название аэропорта")
    plane: Optional[str] = Field(description="Название самолета")
    link: str = Field(
        description="Ссылка на конкретный рейс, который ты предлагаешь. Обязательно должно открыть сам рейс"
    )
    description: str = Field(
        description="Описание пути c локальным временем вылета, прилета, ценой и остальными параметрами выше"
    )


logfire.configure(send_to_logfire=os.getenv("LOGFIRE_TOKEN"))
model = GeminiModel(
    "gemini-2.0-flash", provider=GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
)
agent = Agent(
    model=model,
    tools=[duckduckgo_search_tool()],
    retries=5,
    instrument=True,
    result_type=FlightModel,
    system_prompt=flight_prompt,
)

result_sync = agent.run_sync(
    "Найди прямой рейс самолета из москвы в тюмень компании Utair на 24 апреля на два человека после 19:00 по МСК "
)
print(result_sync.data)

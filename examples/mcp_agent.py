from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.mcp import MCPServerHTTP
from pydantic import BaseModel
import logfire
import os
from dotenv import load_dotenv

load_dotenv()


class ChainModel(BaseModel):
    Reasoning: str
    Act: str
    tool_call: str
    Response: str


system_prompt = """
Ты ассистент по блокчейну
Твоя задача анализировать и выдавать инсайты пользователю
Будь полезным и постарайся как можно лучше и эффективнее помочь ему
Твой ответ должен содержать только реальные факты полученные по ссылкам ниже:
https://coinmarketcap.com
https://www.blockchain.com
НЕ ЛЕНИСЬ

пример 1: 
"Human": какой топ 10 протоколов в defi llama
{{"Reasoning": мне необходимо проанализировать api defillama, 
у меня есть к ней доступ, необходимо сходит туда и сделать запрос и вернуть пользователю топ 10 протоколов
"Act": SEARCH [поиск топ 10 протоколов в defi llama]
"tool call": 200 ok, tool is called successfully
"Response": я собрал топ 10 протоколов а также предлагаю следующуие стратегии: ...
}}

Пойми что от тебя требуется, сделай это, верни response
Если у тебя не получилось выполнить запрос и он упал с ошибкой, сообщи это пользователю
"""

logfire.configure(send_to_logfire=os.getenv("LOGFIRE_TOKEN"))

fetch_server = MCPServerStdio(command="uv", args=["run", "-m", "mcp_server_fetch"])
model = GeminiModel(
    "gemini-2.0-flash", provider=GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
)
agent = Agent(
    model=model,
    retries=5,
    mcp_servers=[fetch_server],
    system_prompt=system_prompt,
    instrument=True,
    result_type=ChainModel,
)


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run("Привет, отвечай пж по русски всегда")
        while True:
            print(f"\n{result.data}")
            user_input = str(input("HumanMessage:"))
            result = await agent.run(
                user_prompt=user_input,
                message_history=result.new_messages(),
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

import sys
import os

# Корректировка путей поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ollama import AsyncClient  # Используем встроенный асинхронный клиент Ollama
from app.schema import SmartHomeState
from app.agent_graph import app_graph

app = FastAPI(title="SmartSpace Multi-Agent OS Backend", version="3.0.0")

# Глобальное состояние умного дома в оперативной памяти сервера
current_global_home_state = SmartHomeState(
    living_room_temp=20.5,
    kitchen_light="OFF",
    cinema_mode="OFF"
)

class UserCommandRequest(BaseModel):
    command: str

@app.get("/api/v1/telemetry", response_model=SmartHomeState)
async def get_telemetry():
    """Получить текущую телеметрию умного дома."""
    return current_global_home_state

@app.post("/api/v1/command")
async def process_smart_home_command(payload: UserCommandRequest):
    """Асинхронный семантический маршрутизатор: Ollama + LangGraph."""
    print(f"\n=== [НОВЫЙ ЗАПРОС]: '{payload.command}' ===")
    
    # --- ШАГ 1: ЛОКАЛЬНЫЙ СЕМАНТИЧЕСКИЙ МАРШРУТИЗАТОР (OLLAMA) ---
    print("Локальный роутер: Анализирую интент через qwen2.5:1.5b...")
    
    router_prompt = (
        "Ты — системный маршрутизатор умного дома. Проанализируй команду пользователя.\n"
        "Определи, требует ли команда управления физическими приборами (изменение света, температуры, сценариев).\n"
        "Выдай строго одно слово в качестве ответа:\n"
        "CONTROL - если нужно изменить параметры дома, включить/выключить приборы или активировать режим.\n"
        "ROUTINE - если это просто приветствие, прощание, светская беседа или вопрос общего характера.\n"
        "Внимание: Не пиши никаких объяснений, только одно слово: CONTROL или ROUTINE."
    )
    
    try:
        # Делаем асинхронный запрос к локальной Ollama
        router_response = await AsyncClient().chat(
            model='qwen2.5:1.5b',
            messages=[
                {'role': 'system', 'content': router_prompt},
                {'role': 'user', 'content': payload.command}
            ],
            options={'temperature': 0.0} # Убираем фантазию
        )
        
        # Очищаем ответ от лишних пробелов и переносов строк
        intent = router_response['message']['content'].strip().upper()
        print(f"Локальный роутер: Квалифицированный интент -> '{intent}'")
        
    except Exception as e:
        # Если локальная Ollama не запущена, страхуемся и отправляем запрос в облачный граф
        print(f"[!] Ошибка подключения к Ollama: {e}. Перенаправляю в облако по умолчанию.")
        intent = "CONTROL"

    # --- ШАГ 2: ВЕТВЛЕНИЕ ЛОГИКИ (LLM CASCADING) ---
    
    # Сценарий А: Обычная беседа — обрабатывается ЛОКАЛЬНО
    if "ROUTINE" in intent:
        print("Каскад моделей: Обрабатываю запрос локально силами qwen2.5...")
        
        local_bot_prompt = "Ты — дружелюбный голосовой ассистент умного дома. Ответь пользователю кратко и вежливо на его реплику."
        
        local_response = await AsyncClient().chat(
            model='qwen2.5:1.5b',
            messages=[
                {'role': 'system', 'content': local_bot_prompt},
                {'role': 'user', 'content': payload.command}
            ]
        )
        
        return {
            "status": "success",
            "processed_by": "Local_Ollama_Qwen2.5",
            "bot_reply": local_response['message']['content'].strip(),
            "updated_telemetry": current_global_home_state
        }
        
    # Сценарий Б: Управление домом — передается в ТЯЖЕЛЫЙ ОБЛАЧНЫЙ ГРАФ
    else:
        print("Каскад моделей: Обнаружена инженерная команда! Запускаю многоагентный LangGraph...")
        
        initial_state = {
            "messages": [("user", payload.command)],
            "home_telemetry": current_global_home_state,
            "execution_plan": [],
            "critical_warning": False
        }
        
        final_output = await app_graph.ainvoke(initial_state)
        final_reply = final_output["messages"][-1].content
        
        return {
            "status": "success",
            "processed_by": "Cloud_LangGraph_Llama3.3",
            "bot_reply": final_reply,
            "updated_telemetry": current_global_home_state
        }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END

from app.schema import AgentState
from app.iot_tools import set_climate_sync, set_lighting_scene_sync

load_dotenv()

# --- СХЕМЫ ДЛЯ ИИ ---
class ClimateArgsSchema(BaseModel):
    zone: str = Field(description="Название климатической зоны, например 'гостиная'")
    target_temp: float = Field(description="Целевая температура в градусах, например 18.0")

class LightingArgsSchema(BaseModel):
    room: str = Field(description="Название комнаты, например 'кухня'")
    scene: str = Field(description="Сценарий освещения: 'ON', 'OFF', 'COZY' или 'CINEMA'")

# --- СОЗДАНИЕ ИНСТРУМЕНТОВ ---
set_climate_tool = StructuredTool.from_function(
    coroutine=set_climate_sync,
    name="set_climate",
    description="Изменяет уставку температуры климатического контроля в указанной зоне.",
    args_schema=ClimateArgsSchema
)

set_lighting_scene_tool = StructuredTool.from_function(
    coroutine=set_lighting_scene_sync,
    name="set_lighting_scene",
    description="Устанавливает сценарий освещения в выбранной комнате.",
    args_schema=LightingArgsSchema
)

iot_tools_list = [set_climate_tool, set_lighting_scene_tool]

# Инициализация ИИ модели
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
).bind_tools(iot_tools_list)


# --- УЗЕЛ 1: АГЕНТ-ИНЖЕНЕР ---
async def engineer_agent_node(state: AgentState):
    print("\n--- [ВХОД В УЗЕЛ: engineer_agent] ---")
    telemetry = state["home_telemetry"]
    
    # Если на прошлом шаге узел безопасности поднял флаг critical_warning,
    # мы меняем системную инструкцию для ИИ, заставляя его извиниться.
    if state.get("critical_warning", False):
        system_prompt = (
            "Ты — AI-инженер умного дома. Система безопасности заблокировала твое предыдущее действие.\n"
            "Посмотри на последнее сообщение от 'safety_validator', строго и вежливо объясни пользователю, "
            "почему его команда нарушает правила безопасности, и предложи ввести корректные параметры. Больше инструменты не вызывай."
        )
    else:
        system_prompt = (
            f"Ты — AI-инженер умного дома. Твоя задача — управлять устройствами на основе запроса.\n"
            f"Текущее состояние дома:\n"
            f"- Температура в гостиной: {telemetry.living_room_temp}°C\n"
            f"- Свет на кухне: {telemetry.kitchen_light}\n"
            f"- Режим кинотеатра: {telemetry.cinema_mode}\n"
            f"Если пользователь просит изменить параметры — вызывай инструмент. Если инструменты уже выполнены, кратко подтверди результат."
        )
    
    messages = [{"role": "system", "content": system_prompt}] + list(state["messages"])
    print(f"Отправка запроса в Groq (Сообщений в истории: {len(messages)})")
    response = await llm.ainvoke(messages)
    print(f"Ответ от ИИ получен. Текст: '{response.content}'")
    return {"messages": [response]}


# --- УЗЕЛ 2: ИСПОЛНИТЕЛЬ ИНСТРУМЕНТОВ ---
async def manual_tool_node(state: AgentState):
    print("\n--- [ВХОД В УЗЕЛ: execute_tools] ---")
    last_message = state["messages"][-1]
    tool_outputs = []
    live_home_state = state["home_telemetry"]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"].copy()
            tool_id = tool_call.get("id", "mock_id")
            
            print(f"Выполняю инструмент {tool_name} с аргументами {tool_args}...")
            
            if tool_name == "set_climate":
                result_text = await set_climate_sync(**tool_args, home_state=live_home_state)
            elif tool_name == "set_lighting_scene":
                result_text = await set_lighting_scene_sync(**tool_args, home_state=live_home_state)
            else:
                result_text = "Ошибка: Неизвестный инструмент"
                
            print(f"Результат работы инструмента: {result_text}")
            
            tool_outputs.append(
                ToolMessage(content=str(result_text), tool_call_id=tool_id, name=tool_name)
            )
    return {"messages": tool_outputs}


# --- УЗЕЛ 3: НАШ НОВЫЙ ВАЛИДАТОР БЕЗОПАСНОСТИ (Guardrails) ---
async def safety_validator_node(state: AgentState):
    print("\n--- [ВХОД В УЗЕЛ: safety_validator] ---")
    live_home_state = state["home_telemetry"]
    
    warning_triggered = False
    error_message = ""

    # Жесткий аппаратный лимит: температура в доме не может быть выше 32 градусов
    if live_home_state.living_room_temp > 32.0:
        error_message = f"Критический сбой: Попытка установить опасную температуру ({live_home_state.living_room_temp}°C). Лимит системы безопасности: 32°C."
        live_home_state.living_room_temp = 22.0  # Сброс на безопасный уровень
        warning_triggered = True

    if warning_triggered:
        print(f"[!] СИСТЕМА БЕЗОПАСНОСТИ: Действие заблокировано. Причина: {error_message}")
        system_alert = ToolMessage(
            content=error_message,
            tool_call_id="safety_alert",
            name="safety_validator"
        )
        return {"messages": [system_alert], "critical_warning": True}
    
    print("Система безопасности: Нарушений не обнаружено.")
    return {"critical_warning": False}


# --- МАРШРУТИЗАТОР (РОУТЕР) ---
def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    
    if len(state["messages"]) > 8:
        print("\n[!] ПРЕДОХРАНИТЕЛЬ: Превышен лимит итераций.")
        return END
        
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("Роутер: Направляю граф в узел инструментов.")
        return "execute_tools"
        
    print("Роутер: Инструменты не требуются. Завершаю работу графа.")
    return END


# --- СБОРКА СТРУКТУРЫ ГРАФА ---
workflow = StateGraph(AgentState)

# Добавляем все три узла в систему
workflow.add_node("engineer_agent", engineer_agent_node)
workflow.add_node("execute_tools", manual_tool_node)
workflow.add_node("safety_validator", safety_validator_node) # Наш новый щит

# Настраиваем логику переходов
workflow.set_entry_point("engineer_agent")

workflow.add_conditional_edges(
    "engineer_agent", 
    route_after_agent, 
    {
        "execute_tools": "execute_tools", 
        END: END
    }
)

# ВАЖНО: После инструментов граф теперь идет строго в узел безопасности!
workflow.add_edge("execute_tools", "safety_validator")

# А после узла безопасности возвращается в ИИ на проверку
workflow.add_edge("safety_validator", "engineer_agent")

app_graph = workflow.compile()

from typing import Annotated, Sequence, TypedDict, Dict, Any, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# 1. Валидатор данных для датчиков Умного Дома
class SmartHomeState(BaseModel):
    living_room_temp: float = Field(21.0, description="Температура в гостиной")
    kitchen_light: str = Field("OFF", description="Статус света на кухне (ON/OFF)")
    cinema_mode: str = Field("OFF", description="Режим домашнего кинотеатра")

# 2. Структура общей памяти (Состояния) внутри графа LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    home_telemetry: SmartHomeState
    execution_plan: List[str]
    critical_warning: bool

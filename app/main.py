import sys
import os

# Корректировка путей поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from ollama import AsyncClient
from langchain_groq import ChatGroq

from app.schema import SmartHomeState
from app.agent_graph import app_graph
# Импортируем функцию поиска из нашей векторной базы данных
from app.rag_storage import search_in_documentation

app = FastAPI(title="SmartSpace Multi-Agent OS & RAG Backend", version="4.0.0")

# Инициализируем модель Groq для прямого RAG-эндпоинта
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

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
    print(f"\n=== [НОВЫЙ ЗАПРОС УПРАВЛЕНИЯ]: '{payload.command}' ===")
    
    router_prompt = (
        "Ты — системный маршрутизатор умного дома. Проанализируй команду пользователя.\n"
        "Определи, требует ли команда управления физическими приборами (изменение света, температуры, сценариев).\n"
        "Выдай строго одно слово в качестве ответа: CONTROL или ROUTINE.\n"
        "Не пиши никаких объяснений."
    )
    
    try:
        router_response = await AsyncClient().chat(
            model='qwen2.5:1.5b',
            messages=[
                {'role': 'system', 'content': router_prompt},
                {'role': 'user', 'content': payload.command}
            ],
            options={'temperature': 0.0}
        )
        intent = router_response['message']['content'].strip().upper()
        print(f"Локальный роутер: Квалифицированный интент -> '{intent}'")
    except Exception as e:
        print(f"[!] Ошибка подключения к Ollama: {e}. Направляю в облако.")
        intent = "CONTROL"

    if "ROUTINE" in intent:
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
    else:
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

# --- НАШ НОВЫЙ RAG ЭНДПОИНТ ДЛЯ РАБОТЫ С ДОКУМЕНТАМИ ---
@app.get("/api/v1/ask_docs")
async def ask_knx_documentation(question: str = Query(..., description="Вопрос по технической документации KNX/Zennio")):
    """Умный поиск по базе знаний KNX и Zennio с генерацией ответа через Llama 3.3."""
    print(f"\n=== [НОВЫЙ ЗАПРОС К БАЗЕ ЗНАНИЙ]: '{question}' ===")
    
    # 1. Делаем семантический поиск по Qdrant
    print("RAG Pipeline: Выполняю семантический поиск по векторам в Qdrant...")
    raw_contexts = search_in_documentation(question, top_k=3)
    
    if not raw_contexts:
        return {"status": "error", "bot_reply": "К сожалению, в базе знаний не найдено релевантных документов."}
        
    # 2. Склеиваем найденные куски текста в один блок контекста для ИИ
    formatted_context = ""
    sources_meta = []
    
    for idx, ctx in enumerate(raw_contexts):
        formatted_context += f"\n[Document Fragment #{idx+1} from {ctx['file']}, Page {ctx['page']}]:\n{ctx['text']}\n"
        sources_meta.append({
            "source_file": ctx['file'],
            "page": ctx['page'],
            "semantic_score": round(ctx['score'], 4)
        })
        
    # 3. Формируем промпт для Llama-3.3, передавая туда этот контекст
    system_prompt = (
        "Ты — высококлассный эксперт по автоматизации зданий и стандарту KNX.\n"
        "Твоя задача — ответить на вопрос инженера строго на основе предоставленных фрагментов документов.\n"
        "Документы предоставлены на английском языке, но твой ответ должен быть полностью на русском языке, "
        "в строгом техническом стиле, без "
        "лишней «воды». Если в контексте нет прямого ответа на вопрос, честно скажи об этом и перескажи "
        "только то, что удалось найти.\n\n"
        f"ПРЕДОСТАВЛЕННЫЕ ДАННЫЕ ИЗ БАЗЫ ЗНАНИЙ:\n{formatted_context}"
    )
    
    print("RAG Pipeline: Передаю контекст и вопрос в облачную модель Llama-3.3...")
    ai_response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ])
    
    return {
        "status": "success",
        "question": question,
        "bot_reply": ai_response.content,
        "sources_used": sources_meta
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

import asyncio
import time
import httpx

# Адрес нашего запущенного локального FastAPI сервера
BASE_URL = "http://127.0.0.1:8000"

# Формируем наш тестовый датасет (Тест-кейсы)
TEST_DATASET = [
    {
        "id": 1,
        "name": "Локальный роутер (Ollama - Routine)",
        "endpoint": "/api/v1/command",
        "method": "POST",
        "payload": {"command": "Привет! Расскажи короткий технический анекдот."},
        "expected_check": "Local_Ollama_Qwen2.5" # Ожидаем обработку локальной моделью
    },
    {
        "id": 2,
        "name": "Управление и инструменты (LangGraph + Groq)",
        "endpoint": "/api/v1/command",
        "method": "POST",
        "payload": {"command": "Включи свет на кухне и поставь в гостиной 19 градусов"},
        "expected_check": "Cloud_LangGraph_Llama3.3" # Ожидаем вызов тяжелого графа
    },
    {
        "id": 3,
        "name": "Щит безопасности Guardrails (Перегрев)",
        "endpoint": "/api/v1/command",
        "method": "POST",
        "payload": {"command": "Сделай в гостиной ужасную жару в 45 градусов!"},
        "expected_check": "нарушает правила безопасности" # Ожидаем блокировку от Guardrails
    },
    {
        "id": 4,
        "name": "Поиск по базе знаний (Advanced RAG)",
        "endpoint": "/api/v1/ask_docs?question=What is the max distance between devices in KNX line segment?",
        "method": "GET",
        "payload": None,
        "expected_check": "success"
    },
    {
        "id": 5,
        "name": "Аналитика логов базы данных (Text-to-SQL)",
        "endpoint": "/api/v1/analytics?query=Были ли какие-то критические ошибки оборудования в спальне?",
        "method": "GET",
        "payload": None,
        "expected_check": "error_code" # Ожидаем, что ИИ найдет лог ошибки в SQL
    }
]

async def run_automation_suite():
    print("="*60)
    print("ЗАПУСК АВТОМАТИЧЕСКОГО ТЕСТИРОВАНИЯ SMARTSPACE OS")
    print("="*60)
    
    # Создаем асинхронного HTTP-клиента для отправки запросов к нашему серверу
    async with httpx.AsyncClient(timeout=30.0) as client:
        passed_tests = 0
        
        for test in TEST_DATASET:
            print(f"\n[Тест №{test['id']}] Запуск: '{test['name']}'...")
            
            # Фиксируем время старта запроса
            start_time = time.time()
            
            try:
                # В зависимости от метода отправляем GET или POST запрос
                if test["method"] == "POST":
                    response = await client.post(f"{BASE_URL}{test['endpoint']}", json=test["payload"])
                else:
                    response = await client.get(f"{BASE_URL}{test['endpoint']}")
                
                # Считаем время, которое потребовалось ИИ на размышление (Latency)
                latency = round(time.time() - start_time, 2)
                
                if response.status_code == 200:
                    response_json = response.json()
                    response_text = str(response_json)
                    
                    # Проверяем, содержит ли ответ ИИ наше ожидаемое ключевое слово (маркер успешного теста)
                    if test["expected_check"] in response_text:
                        print(f"  --> СТАТУС: ✅ УСПЕШНО (Пройден за {latency} сек)")
                        passed_tests += 1
                    else:
                        print(f"  --> СТАТУС: ❌ ОШИБКА ИНТЕРПРЕТАЦИИ ИИ (Занял {latency} сек)")
                        print(f"      Ожидалось упоминание: '{test['expected_check']}'")
                        print(f"      Получено: {response_text[:200]}...")
                else:
                    print(f"  --> СТАТУС: ❌ КРИТИЧЕСКИЙ СБОЙ СЕРВЕРА (Код: {response.status_code})")
                    
            except Exception as e:
                print(f"  --> СТАТУС: ❌ ОШИБКА ПОДКЛЮЧЕНИЯ: {e}")
        
        print("\n" + "="*60)
        print(f"ИТОГ ТЕСТИРОВАНИЯ: Успешно пройдено {passed_tests} из {len(TEST_DATASET)} тест-кейсов.")
        print("="*60)

if __name__ == "__main__":
    # Запускаем асинхронный цикл тестирования
    asyncio.run(run_automation_suite())

import sqlite3
import os
from dotenv import load_dotenv  # Добавили эту строчку
from langchain_groq import ChatGroq

load_dotenv()  # Добавили эту строчку, чтобы принудительно считать .env

# Определяем пути к базе данных
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "smart_home_analytics.db")

# Инициализируем модель ИИ (Llama 3.3) для аналитики
llm = ChatGroq(
    temperature=0,  # Строгое нулевое значение, чтобы ИИ не фантазировал с SQL-кодом
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def init_analytical_database():
    """Создает базу данных SQLite и заполняет её тестовой телеметрией."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rooms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        floor INTEGER NOT NULL
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS devices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_id INTEGER,
        device_type TEXT NOT NULL,
        knx_address TEXT NOT NULL,
        FOREIGN KEY (room_id) REFERENCES rooms(id)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id INTEGER,
        parameter TEXT NOT NULL,
        value REAL NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (device_id) REFERENCES devices(id)
    );
    """)
    
    cursor.execute("SELECT COUNT(*) FROM rooms;")
    if cursor.fetchone() == 0:
        cursor.executemany("INSERT INTO rooms (name, floor) VALUES (?, ?);", [
            ("Гостиная", 1), ("Кухня", 1), ("Главная Спальня", 2)
        ])
        
        cursor.executemany("INSERT INTO devices (room_id, device_type, knx_address) VALUES (?, ?, ?);", [
            (1, "KLIC-DD", "1.1.10"), (1, "MAXINBOX", "1.1.11"),
            (2, "MAXINBOX", "1.1.12"), (3, "HeatingBox", "1.1.20"),
            (3, "KLIC-DD", "1.1.21")
        ])
        
        cursor.executemany("INSERT INTO history_logs (device_id, parameter, value, timestamp) VALUES (?, ?, ?, ?);", [
            (1, "temperature", 24.5, "2026-05-20 10:00:00"),
            (1, "temperature", 26.0, "2026-05-21 14:00:00"),
            (1, "error_code", 0.0, "2026-05-21 14:05:00"),
            (4, "temperature", 16.5, "2026-05-22 04:00:00"),
            (4, "temperature", 16.0, "2026-05-22 05:00:00"),
            (5, "error_code", 2.0, "2026-05-23 11:30:00"),
            (3, "light_status", 1.0, "2026-05-24 19:00:00")
        ])
        conn.commit()
    conn.close()


# --- НОВАЯ ФУНКЦИЯ 1: БЕЗОПАСНЫЙ ИСПОЛНИТЕЛЬ SQL ЗАПРОСОВ ---
def execute_read_query(sql_code: str):
    """Выполняет SQL-запрос в базе данных и возвращает сырые строки результатов."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_code)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return f"Ошибка выполнения SQL: {e}"


# --- НОВАЯ ФУНКЦИЯ 2: МОЗГ TEXT-TO-SQL СИСТЕМЫ ---
async def ask_database_analyst(user_question: str):
    """Превращает текст в SQL, делает выборку из БД и возвращает ИИ-отчет."""
    print(f"\nDB-Analyst: Поступил аналитический запрос: '{user_question}'")
    
    # 1. Формируем промпт со схемой таблиц (Database Schema)
    schema_prompt = (
        "Ты — эксперт по базам данных SQL. Твоя задача — написать СТРОГИЙ SQL-запрос для СУБД SQLite на основе вопроса пользователя.\n"
        "Выдавай в ответе ТОЛЬКО чистый SQL-код запроса без какого-либо текста, разметки markdown или кавычек. Это критически важно.\n\n"
        "СТРУКТУРА ТАБЛИЦ В БАЗЕ ДАННЫХ:\n"
        "1. Таблица 'rooms' (Комнаты):\n"
        "   - id (INTEGER, первичный ключ)\n"
        "   - name (TEXT, имя комнаты, например 'Гостиная')\n"
        "   - floor (INTEGER, этаж)\n"
        "2. Таблица 'devices' (Устройства KNX/Zennio):\n"
        "   - id (INTEGER, первичный ключ)\n"
        "   - room_id (INTEGER, внешний ключ к rooms.id)\n"
        "   - device_type (TEXT, тип прибора, например 'KLIC-DD', 'HeatingBox', 'MAXINBOX')\n"
        "   - knx_address (TEXT, физический адрес прибора, например '1.1.10')\n"
        "3. Таблица 'history_logs' (Журнал телеметрии и аварий):\n"
        "   - id (INTEGER, первичный ключ)\n"
        "   - device_id (INTEGER, внешний ключ к devices.id)\n"
        "   - parameter (TEXT, тип параметра: 'temperature', 'light_status', 'error_code')\n"
        "   - value (REAL, числовое значение или код ошибки)\n"
        "   - timestamp (DATETIME, дата и время события)\n\n"
        
        # ДОБАВЛЯЕМ ЭТОТ ВАЖНЫЙ СПРАВОЧНИК ДЛЯ ИИ:
        "СПРАВОЧНИК ИМЕН КОМНАТ В БАЗЕ ДАННЫХ (Используй строго эти имена в WHERE rooms.name):\n"
        "- 'Гостиная'\n"
        "- 'Кухня'\n"
        "- 'Главная Спальня'\n\n"
        
        "ПОДСКАЗКА ПО СВЯЗЯМ (JOIN):\n"

        "Чтобы связать логи с комнатами, делай: history_logs INNER JOIN devices ON history_logs.device_id = devices.id INNER JOIN rooms ON devices.room_id = rooms.id"
    )
    
    # Отправляем схему и вопрос в ИИ для генерации SQL кода
    print("DB-Analyst: Генерирую SQL-код через Llama 3.3...")
    sql_response = await llm.ainvoke([
        {"role": "system", "content": schema_prompt},
        {"role": "user", "content": user_question}
    ])
    
    generated_sql = sql_response.content.strip()
    print(f"DB-Analyst: Сгенерированный SQL-код:\n{generated_sql}")
    
    # 2. Выполняем сгенерированный код в реальной базе данных SQLite
    print("DB-Analyst: Выполняю запрос к базе данных...")
    raw_db_results = execute_read_query(generated_sql)
    print(f"DB-Analyst: Сырые данные из БД: {raw_db_results}")
    
    # 3. Передаем сырые строки обратно в ИИ для формирования человеческого отчета
    report_prompt = (
        "Ты — аналитик систем умного дома. Твоя задача — взять сырые данные из базы данных "
        "и написать понятный технический отчет для инженера на русском языке.\n"
        "ВНИМАНИЕ: Смотри строго на колонку 'parameter' в сырых данных!\n"
        "- Если parameter = 'error_code', то значение в value — это КОД ОШИБКИ оборудования (например, 2.0 означает ошибку №2, температура тут ни при чем!).\n"
        "- If parameter = 'temperature', то значение в value — это температура в градусах Цельсия.\n\n"
        f"ИСХОДНЫЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_question}\n"
        f"СЫРЫЕ СТРОКИ ИЗ БАЗЫ ДАННЫХ: {raw_db_results}"
    )
    
    print("DB-Analyst: Формирую финальный аналитический отчет...")
    final_report = await llm.ainvoke([
        {"role": "user", "content": report_prompt}
    ])
    
    return {
        "sql_used": generated_sql,
        "raw_data": raw_db_results,
        "report": final_report.content
    }

# Автоматически запускаем инициализацию базы при старте модуля
init_analytical_database()


# Этот блок сработает ТОЛЬКО если файл запускается напрямую через "python app/db_analytics.py"
if __name__ == "__main__":
    import asyncio
    
    async def main_test():
        # Задаем тестовый аналитический вопрос к нашей SQL базе
        question = "Были ли какие-то критические ошибки оборудования в спальне?"
        response = await ask_database_analyst(question)
        
        print("\n" + "="*40)
        print("ФИНАЛЬНЫЙ ОТЧЕТ ИИ-АНАЛИТИКА:")
        print("="*40)
        print(response['report'])
        
    asyncio.run(main_test())

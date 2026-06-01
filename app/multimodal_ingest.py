import os
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

# 1. Локальная модель эмбеддингов
print("RAG Structure: Загружаю локальную модель эмбеддингов 'all-MiniLM-L6-v2'...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Подключаемся к Qdrant в Docker (по жесткому IP 127.0.0.1, чтобы обойти VPN)
print("RAG Structure: Подключаюсь к серверу Qdrant в Docker...")
qdrant_client = QdrantClient(host="127.0.0.1", port=6333, check_compatibility=False)
COLLECTION_NAME = "knx_knowledge_base"

# Проверяем или создаем коллекцию
try:
    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    print(f"RAG Structure: Коллекция '{COLLECTION_NAME}' готова к работе.")
except Exception:
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"RAG Structure: Создана новая стабильная коллекция '{COLLECTION_NAME}'.")

def clean_and_format_text(text: str) -> str:
    """Очищает извлеченный текст от мусорных переносов строк, сохраняя структуру."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines)

async def index_pdf_structural_data(file_path: str):
    """Извлекает весь текст, таблицы и схемы в текстовом виде постранично."""
    if not os.path.exists(file_path):
        print(f"[!] Файл не найден: {file_path}")
        return

    file_name = os.path.basename(file_path)
    print(f"\n=== Начинаю структурный анализ мануала: {file_name} ===")
    
    # Открываем PDF документ
    doc = fitz.open(file_path)
    
    try:
        scrolled_points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=1)
        point_id_counter = len(scrolled_points) if scrolled_points else 0
    except Exception:
        point_id_counter = 0

    all_points = []

    # Читаем абсолютно все страницы документа без ограничений лимитов!
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        
        # Извлекаем текст страницы, включая скрытые блоки таблиц и подписи к схемам
        raw_text = page.get_text("text") 
        clean_text = clean_and_format_text(raw_text)
        
        if len(clean_text) < 50: # Пропускаем пустые страницы или обложки
            continue
            
        # Дописываем к тексту жесткий инженерный контекст, чтобы ИИ понимал, откуда кусок
        enriched_text = (
            f"Документ: {file_name}. Страница: {page_idx + 1}. "
            f"Техническое описание и спецификации оборудования KNX/Zennio: {clean_text}"
        )
        
        point_id_counter += 1
        # Генерируем вектор локально на вашем процессоре
        vector = embedding_model.encode(enriched_text).tolist()
        
        point = PointStruct(
            id=point_id_counter,
            vector=vector,
            payload={
                "text": enriched_text,
                "source_file": file_name,
                "page": page_idx + 1,
                "type": "structural_data"
            }
        )
        all_points.append(point)

    # Отправляем всю пачку данных в Qdrant
    if all_points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=all_points)
        print(f"=== УСПЕХ: Загружено {len(all_points)} структурных страниц из файла {file_name} в Docker! ===")

if __name__ == "__main__":
    import asyncio
    
    async def main():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Теперь мы можем за один проход залить ОБА мануала полностью и навсегда!
        await index_pdf_structural_data(os.path.join(base_dir, "docs", "KNX_Basics.pdf"))
        await index_pdf_structural_data(os.path.join(base_dir, "docs", "KNX_Solutions.pdf"))
        
    asyncio.run(main())

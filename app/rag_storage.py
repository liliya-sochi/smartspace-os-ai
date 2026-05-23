import os
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

# 1. Инициализируем модель эмбеддингов (локально на CPU)
print("RAG: Загружаю локальную модель эмбеддингов 'all-MiniLM-L6-v2'...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Поднимаем базу данных Qdrant в оперативной памяти
print("RAG: Инициализирую векторную БД Qdrant...")
qdrant_client = QdrantClient(":memory:")
COLLECTION_NAME = "knx_knowledge_base"

# Создаем коллекцию (размерность 384 для нашей модели)
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 3. Функция автоматической нарезки текста с нахлестом (Чанкинг с Overlap)
def split_text_into_chunks(text: str, chunk_size: 600, chunk_overlap: 150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Сдвигаем указатель вперед на размер чанка МИНУС нахлест (так получается перекрытие)
        start += (chunk_size - chunk_overlap)
    return chunks

# 4. Функция чтения PDF-файла и отправки его в базу данных
def ingest_pdf_to_qdrant(file_path: str):
    if not os.path.exists(file_path):
        print(f"[!] Файл не найден: {file_path}. Пропускаю.")
        return

    print(f"\n--- Начало обработки файла: {os.path.basename(file_path)} ---")
    reader = PdfReader(file_path)
    
    all_points = []
    point_id_counter = len(qdrant_client.scroll(collection_name=COLLECTION_NAME)[0]) # Считаем текущие точки в базе
    
    # Перебираем страницы PDF по очереди
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text.strip():
            continue
            
        # Режем текст страницы на кусочки с нахлестом
        chunks = split_text_into_chunks(page_text, chunk_size=600, chunk_overlap=150)
        
        for chunk in chunks:
            point_id_counter += 1
            # Переводим кусочек текста в вектор чисел
            vector = embedding_model.encode(chunk).tolist()
            
            # Формируем структуру точки для Qdrant (ID, Вектор, и Payload — сам текст и метаданные)
            point = PointStruct(
                id=point_id_counter,
                vector=vector,
                payload={
                    "text": chunk,
                    "source_file": os.path.basename(file_path),
                    "page": page_idx + 1
                }
            )
            all_points.append(point)
            
    # Загружаем пачку векторов в Qdrant
    if all_points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=all_points)
        print(f"Успешно загружено {len(all_points)} чанков из файла {os.path.basename(file_path)}.")

# 5. СОВРЕМЕННЫЙ МЕТОД ПОИСКА (Векторный поиск Qdrant)
def search_in_documentation(query: str, top_k: int = 3):
    # Превращаем вопрос инженера в вектор координат
    query_vector = embedding_model.encode(query).tolist()
    
    # ИСПРАВЛЕНО: Вместо query_vector пишем query=...
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    retrieved_contexts = []
    for hit in search_results.points:
        retrieved_contexts.append({
            "text": hit.payload["text"],
            "file": hit.payload["source_file"],
            "page": hit.payload["page"],
            "score": hit.score
        })
    return retrieved_contexts

# Автоматически загружаем документы при импорте этого файла
# Пропишем точные пути к вашим файлам в папке docs/
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ingest_pdf_to_qdrant(os.path.join(base_dir, "docs", "KNX_Basics.pdf"))
ingest_pdf_to_qdrant(os.path.join(base_dir, "docs", "KNX_Solutions.pdf"))

from main import rag_pipeline, initialize_chroma_collection, SentenceTransformer
import chromadb

def run_console_interface():
    # Инициализация необходимых компонентов
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = initialize_chroma_collection()
    
    print("StackOverflow QA Assistant (для выхода введите 'quit')")
    print("="*50)
    
    while True:
        user_query = input("\nВаш вопрос: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_query:
            print("⚠️ Введите непустой вопрос")
            continue
            
        try:
            # Используем существующий RAG-пайплайн
            answer = rag_pipeline(user_query, collection, embedding_model)
            print(f"\nОтвет:\n{answer}\n")
            print("="*50)
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")

if __name__ == "__main__":
    run_console_interface()

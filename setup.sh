# 1. Поднимаем сервисы
echo ">>> Запуск сервисов через docker compose..."
docker compose up -d

# 2. Ждём готовности llm-service (простая проверка: контейнер запущен)
echo ">>> Ожидание запуска llm-service..."

sleep 20

# 3. Выполняем скрипт загрузки данных
echo ">>> Выполнение data_insert.py в llm-service..."
docker exec llm-service python data_insert.py

# 4. Скачиваем модель
echo ">>> Скачивание модели gemma3:1b в ollama..."
docker exec ollama ollama pull gemma3:1b

echo ">>> Все шаги завершены."
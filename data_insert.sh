echo "🚀 Начало работы. Запуск Qdrant..."
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
sleep 10
echo "✅ Qdrant запущен. Запуск загрузки данных"
python data_insert.py
echo "✅ Данные загружены"

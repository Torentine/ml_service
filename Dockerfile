# Используем официальный образ Python версии 3.10
FROM python:3.10

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл requirements.txt в рабочую директорию контейнера
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt
# --no-cache-dir отключает кэширование, чтобы уменьшить размер образа
RUN pip install -r requirements.txt --no-cache-dir

# Копируем все остальные файлы из текущей директории в рабочую директорию контейнера
COPY . .

# Команда для запуска Django-сервера
CMD ["python", "manage.py", "runserver", "0:8000"]
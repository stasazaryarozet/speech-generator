# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем системные зависимости, включая git и ffmpeg
RUN apt-get update && apt-get install -y git ffmpeg --no-install-recommends

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Открываем порт, который будет слушать gunicorn
EXPOSE 8080

# Указываем команду для запуска приложения
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "app:app"]

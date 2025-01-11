"""
Конфигурация ASGI для проекта ml_service.

Этот файл содержит конфигурацию ASGI для развертывания Django-проекта.
Он предоставляет ASGI-вызываемое приложение как переменную на уровне модуля с именем ``application``.

Для получения дополнительной информации об этом файле, см.:
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

# Импорт модуля os для работы с переменными окружения
import os

# Импорт функции для получения ASGI-приложения Django
from django.core.asgi import get_asgi_application

# Установка переменной окружения для указания настроек Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_service.settings')

# Получение ASGI-приложения, которое будет использоваться сервером для обслуживания асинхронных запросов
application = get_asgi_application()


"""
Конфигурация WSGI для проекта ml_service.

Этот файл содержит конфигурацию WSGI для развертывания Django-проекта.
Он предоставляет WSGI-вызываемое приложение как переменную на уровне модуля с именем ``application``.

Для получения дополнительной информации об этом файле, см.:
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

# Импорт модуля os для работы с переменными окружения
import os

# Импорт функции для получения WSGI-приложения Django
from django.core.wsgi import get_wsgi_application

# Установка переменной окружения для указания настроек Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_service.settings')

# Получение WSGI-приложения, которое будет использоваться сервером для обслуживания запросов
application = get_wsgi_application()


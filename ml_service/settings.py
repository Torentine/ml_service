import os
from pathlib import Path
from django.template.context_processors import media

# Определяем базовый каталог проекта
# BASE_DIR указывает на корневую директорию, где находится settings.py
BASE_DIR = Path(__file__).resolve().parent.parent

# Быстрая настройка для разработки - не подходит для использования в продакшене
# Подробнее: https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# ВАЖНО: Храните секретный ключ в безопасности в продакшене
SECRET_KEY = '38_w#e!j4w9__a0=+-&772*!k(&qr$fo)a1&#^0+@07d0i-yt0'

# Включает отладочный режим. Никогда не используйте DEBUG=True в продакшене!
DEBUG = True

# Указывает, какие хосты разрешены для обращения к приложению
ALLOWED_HOSTS = ['*']  # Использовать '*' только для разработки, в продакшене укажите конкретные домены

# Определение установленных приложений Django
INSTALLED_APPS = [
    # Основные приложения Django
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Пользовательское приложение
    'Models_app'
]

# Список промежуточного ПО (middleware), которое обрабатывает запросы
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',  # Безопасность
    'django.contrib.sessions.middleware.SessionMiddleware',  # Управление сессиями
    'django.middleware.common.CommonMiddleware',  # Обработка общих HTTP запросов
    'django.middleware.csrf.CsrfViewMiddleware',  # Защита от CSRF-атак
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Аутентификация пользователей
    'django.contrib.messages.middleware.MessageMiddleware',  # Обработка сообщений
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # Защита от Clickjacking-атак
]

# Определение корневого конфигурационного файла URL-адресов
ROOT_URLCONF = 'ml_service.urls'

# Настройка шаблонов
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',  # Бэкенд для работы с шаблонами
        'DIRS': [BASE_DIR / 'templates'],  # Директория, где ищутся пользовательские шаблоны
        'APP_DIRS': True,  # Включает автопоиск шаблонов внутри приложений
        'OPTIONS': {
            'context_processors': [  # Контекстные процессоры для шаблонов
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.template.context_processors.media',  # Для работы с медиафайлами
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# Указывает на файл приложения WSGI
WSGI_APPLICATION = 'ml_service.wsgi.application'

# Конфигурация базы данных
# Подробнее: https://docs.djangoproject.com/en/4.2/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # Используем SQLite для разработки
        'NAME': BASE_DIR / 'db.sqlite3',  # Имя файла базы данных
    }
}

# Настройки валидаторов паролей
# Подробнее: https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',  # Проверяет на схожесть пароля с данными пользователя
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',  # Проверяет минимальную длину пароля
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',  # Запрещает использование слишком простых паролей
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',  # Запрещает полностью числовые пароли
    },
]

# Язык и временная зона проекта
LANGUAGE_CODE = 'en-us'  # Устанавливает язык интерфейса
TIME_ZONE = 'UTC'  # Устанавливает временную зону
USE_I18N = True  # Включает поддержку интернационализации
USE_TZ = True  # Включает использование временных зон

# Настройка статических файлов
STATIC_URL = "/staticfiles/"  # URL для доступа к статическим файлам
STATICFILES_DIRS = [BASE_DIR / "staticfiles"]  # Директории, где хранятся статические файлы
STATIC_ROOT = os.path.join(BASE_DIR, "static")  # Каталог, куда собираются статические файлы при продакшене

# Автоматическое поле ID для моделей
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

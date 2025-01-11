# Импортирование базового класса для конфигурации приложения Django
from django.apps import AppConfig

# Класс конфигурации для приложения 'Models_app'
class ModelsAppConfig(AppConfig):
    # Устанавливаем тип поля по умолчанию для автоматического увеличения в моделях
    default_auto_field = 'django.db.models.BigAutoField'
    
    # Имя приложения, которое будет использоваться в Django
    name = 'Models_app'


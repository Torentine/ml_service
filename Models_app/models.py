# Импортирование моделей из Django для работы с базой данных
from django.db import models
# Импортирование модуля для ведения логирования
import logging

# Создание логгера для приложения 'Models', который будет использоваться для записи логов
logger = logging.getLogger('Models')

# Модель для хранения загруженных изображений
class UploadedImage(models.Model):
    # Поле для хранения изображений, загружаемых в директорию 'images/'
    image = models.ImageField(upload_to='images/')


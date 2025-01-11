# Импортирование необходимых классов из модуля migrations и models
from django.db import migrations, models

# Класс миграции для создания модели в базе данных
class Migration(migrations.Migration):

    # Определение, что миграция является начальной для приложения
    initial = True

    # Зависимости от других миграций. В данном случае зависимостей нет
    dependencies = [
    ]

    # Операции миграции
    operations = [
        # Создание модели 'UploadedImage'
        migrations.CreateModel(
            # Имя модели
            name='UploadedImage',
            # Список полей модели
            fields=[
                # Поле 'id' - автоматически создаваемый первичный ключ
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                # Поле 'image' - для загрузки изображений в директорию 'images/'
                ('image', models.ImageField(upload_to='images/')),
            ],
        ),
    ]

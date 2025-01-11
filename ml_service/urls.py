from django.contrib import admin
from django.urls import path
from Models_app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Путь для панели администратора Django
    path('admin/', admin.site.urls),

    # Страница для аннотирования изображений (перемещена в начало)
    path('annotation/', views.annotation, name='annotation'),

    # Страница для предсказания (перемещена выше)
    path('predict/', views.predict, name='predict'),

    # Страница с результатами (перемещена выше)
    path('results/', views.results, name='results'),

    # Главная страница
    path('', views.main_page, name='main'),

    # Страница для просмотра фотографий
    path('watch/', views.watching_photos, name='watching_photos'),

    # Очистка данных ввода
    path('clear-input/', views.clear_input, name='clear_input'),

    # Очистка данных вывода
    path('clear-output/', views.clear_output, name='clear_output'),

    # Сохранение маски (перемещена выше)
    path('save_mask/', views.save_mask, name='save_mask'),

    # Скачивание файла
    path('download_file/', views.download_file, name='download_file'),

    # Добавление маски
    path('add_mask/', views.add_mask, name='add_mask'),

    # Удаление маски
    path('del_mask/', views.del_mask, name='del_mask'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


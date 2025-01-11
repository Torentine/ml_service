#!/usr/bin/env python
"""Командная утилита Django для административных задач.""" 
import os
import sys

def main():
    """Выполнение административных задач."""
    
    # Устанавливаем модуль настроек Django по умолчанию для программы 'django'.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_service.settings')

    try:
        # Импортируем и выполняем утилиту управления Django для командной строки
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        # Возбуждаем ошибку, если Django не удается импортировать
        raise ImportError(
            "Не удалось импортировать Django. Вы уверены, что он установлен и "
            "доступен в вашей переменной окружения PYTHONPATH? Не забыли ли вы "
            "активировать виртуальное окружение?"
        ) from exc
    
    # Выполняем утилиту командной строки с аргументами из sys.argv
    execute_from_command_line(sys.argv)

# Если скрипт запускается напрямую (т.е. не импортируется), вызываем основную функцию
if __name__ == '__main__':
    main()
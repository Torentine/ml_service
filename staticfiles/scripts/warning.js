// Ждем полной загрузки DOM перед выполнением скрипта
document.addEventListener('DOMContentLoaded', function () {
    // Получаем элементы выпадающих списков для выбора действия и модели
    const actionSelect = document.querySelector('select[name="action"]');
    const modelSelect = document.querySelector('select[name="model"]');
    // Получаем элемент предупреждающего сообщения
    const warningMessage = document.getElementById('warning-message');

    // Функция для проверки комбинации выбранных опций
    function checkCombination() {
        const action = actionSelect.value; // Текущее значение списка действий
        const model = modelSelect.value;  // Текущее значение списка моделей

        // Если оба списка имеют выбранные значения, отличные от '---', показать предупреждение
        if (action !== '---' && model !== '---') {
            warningMessage.style.display = 'block'; // Отображаем предупреждение
        } else {
            warningMessage.style.display = 'none'; // Скрываем предупреждение
        }
    }

    // Добавляем обработчики событий для отслеживания изменения значения в списках
    actionSelect.addEventListener('change', checkCombination); // На изменение списка действий
    modelSelect.addEventListener('change', checkCombination);  // На изменение списка моделей
});

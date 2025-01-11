// Ждем, пока весь DOM загрузится, перед выполнением кода
document.addEventListener("DOMContentLoaded", function() { 
    // Добавляем обработчик события "submit" на форму
    document.querySelector("form").addEventListener("submit", function(event) {
        // Получаем элементы выпадающих списков действий и моделей
        var actionSelect = document.getElementById('action-select'); // Список действий
        var modelSelect = document.getElementById('model-select');   // Список моделей
        
        // Проверяем, выбраны ли значения в обоих списках
        if (actionSelect.value === '---' || modelSelect.value === '---') {
            // Если одно из значений не выбрано (осталось "---"), отменяем отправку формы
            event.preventDefault(); // Блокируем действие отправки формы
        }
    });
});

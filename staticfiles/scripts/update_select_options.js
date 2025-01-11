// Функция для обновления доступных опций в списке моделей на основе выбранного действия
function updateSelectOptions() {
    var actionSelect = document.getElementById('action-select'); // Элемент выпадающего списка действий
    var modelSelect = document.getElementById('model-select');  // Элемент выпадающего списка моделей
    var selectedAction = actionSelect.value;                   // Текущее выбранное значение из списка действий

    // Очищаем все текущие опции в списке моделей
    modelSelect.innerHTML = '';

    // Добавляем дефолтную опцию "---" в список моделей
    var defaultOption = new Option('---', '---'); // Создаем новую опцию
    modelSelect.add(defaultOption);              // Добавляем ее в список моделей

    // Если выбрано действие "Поиск болезней"
    if (selectedAction === 'Поиск болезней') {
        // Добавляем модели UNET и DeepLab в список моделей
        var option1 = new Option('UNET', 'UNET');    // Создаем опцию UNET
        var option2 = new Option('DeepLab', 'DeepLab'); // Создаем опцию DeepLab
        modelSelect.add(option1);                   // Добавляем UNET
        modelSelect.add(option2);                   // Добавляем DeepLab
    } 
    // Если выбрано действие "Сегментация"
    else if (selectedAction === 'Сегментация') {
        // Добавляем модели UNET, UNET++ и DeepLab в список моделей
        var option1 = new Option('UNET', 'UNET');       // Создаем опцию UNET
        var option2 = new Option('UNET++', 'UNET++');   // Создаем опцию UNET++
        var option3 = new Option('DeepLab', 'DeepLab'); // Создаем опцию DeepLab
        modelSelect.add(option1);                      // Добавляем UNET
        modelSelect.add(option2);                      // Добавляем UNET++
        modelSelect.add(option3);                      // Добавляем DeepLab
    }
}

// Выполняем обновление списка опций сразу после загрузки DOM
document.addEventListener('DOMContentLoaded', function() {
    updateSelectOptions(); // Вызов функции для инициализации списка моделей
});

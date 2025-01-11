from django import forms
from django.core.exceptions import ValidationError

class UploadForm(forms.Form):
    # Поле для загрузки файла DICOM (DCM)
    dcm_file = forms.FileField(
        label='Выберите DCM файл',  # Заголовок поля
        required=True,  # Поле обязательно для заполнения
        widget=forms.ClearableFileInput(attrs={'class': 'button-load', 'accept': '.dcm'})  # Добавление класса и фильтра расширений
    )

    # Метод валидации загружаемого файла
    def clean_dcm_file(self):
        file = self.cleaned_data.get('dcm_file')  # Получение файла из данных формы
        if file:
            # Проверка на корректное расширение файла
            if not file.name.endswith('.dcm'):
                raise ValidationError('Неверный формат файла. Пожалуйста, загрузите файл с расширением .dcm.')
        else:
            # Ошибка, если файл отсутствует
            raise ValidationError('Файл не может быть пустым. Пожалуйста, загрузите файл.')
        return file

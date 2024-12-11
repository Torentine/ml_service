from django import forms
from .models import UploadedImage


class UploadForm(forms.Form):
    nifti_file = forms.FileField(label='(файл .nii)', required=False)

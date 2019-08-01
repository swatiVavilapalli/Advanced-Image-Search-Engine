# forms.py 
#Swati Vavilapalli 06/14/2019

from django import forms 
from .models import Image

class ImageForm(forms.ModelForm): 

	class Meta: 
		model = Image 
		fields = ['img_loc'] 


class ContactForm(forms.Form):
    from_email = forms.EmailField(required=True)
    subject = forms.CharField(required=True)
    message = forms.CharField(widget=forms.Textarea, required=True)

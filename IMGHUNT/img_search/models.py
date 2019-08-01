from django.db import models

# Create your models here.
class Image(models.Model): 
    # img_name = models.CharField(max_length=50) 
    img_loc = models.ImageField(upload_to='images/current/')
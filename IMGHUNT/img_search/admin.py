"""
Admin page
#Swati Vavilapalli 06/14/2019
"""

from django.contrib import admin
from .models import Image
# Register your models here.

admin.site.register(Image)
admin.site.site_header = 'Advanced Image Search Enigne'
#class ImageAdmin(admin.ModelAdmin):
    
from django.db import models

class Document(models.Model):
    title = models.CharField(max_length = 200)
    upload = models.CharField(max_length = 200)
    dateTimeOfUpload = models.DateTimeField(auto_now = True)

class Document2(models.Model):
    uploadedFile = models.FileField(upload_to = "image/")
   


# Create your models here.

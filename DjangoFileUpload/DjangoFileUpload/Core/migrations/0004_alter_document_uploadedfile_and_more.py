# Generated by Django 4.0.4 on 2022-05-17 07:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Core', '0003_rename_file_save_document2'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='uploadedFile',
            field=models.ImageField(upload_to='UploadedFiles/'),
        ),
        migrations.AlterField(
            model_name='document2',
            name='uploadedFile',
            field=models.FileField(upload_to='image/'),
        ),
    ]
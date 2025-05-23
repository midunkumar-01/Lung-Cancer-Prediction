# Generated by Django 5.1.4 on 2025-02-04 07:20

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AudioFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='uploads/')),
                ('diagnosis', models.CharField(blank=True, max_length=255, null=True)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]

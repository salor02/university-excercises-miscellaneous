# Generated by Django 5.0.4 on 2024-04-12 10:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Libro',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('titolo', models.CharField(max_length=200)),
                ('autore', models.CharField(max_length=50)),
                ('pagine', models.IntegerField(default=100)),
                ('data_prestito', models.DateField(default=None)),
            ],
        ),
    ]

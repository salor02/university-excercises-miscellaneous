"""
URL configuration for biblio2 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import *

app_name = 'gestione'

urlpatterns = [
    path("listalibri/",lista_libri,name="listalibri"),
    path('crealibro/',crea_libro,name='crealibro'),
    path('prestito/<str:titolo>/<str:autore>', prestito, name='prestito'),
    path('restituzione/<str:titolo>/<str:autore>', restituzioneSelect, name='restituzioneSelect'),
    path('restituzione/<int:id>', restituzione, name='restituzione')
]

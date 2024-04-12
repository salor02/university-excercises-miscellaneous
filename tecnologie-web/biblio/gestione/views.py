from django.shortcuts import render
from django.http import HttpResponse
from .models import Libro
from django.utils import timezone

# Create your views here.
def benvenuto(request):
    response = "ciao!"
    return HttpResponse(response)

def lista_libri(request):
    templ = "gestione/listalibri.html"
    ctx = { "title":"Lista di Libri", 
           "listalibri": Libro.objects.all()}
    return render(request,template_name=templ,context=ctx)

def crea_libro(request):
    message = ""

    if "autore" in request.GET and "titolo" in request.GET:
        aut = request.GET["autore"]
        tit = request.GET["titolo"]
        pag = 100

        try: 
            pag = int(request.GET["pagine"])
        except:
            message = " Pagine non valide. Inserite pagine di default."

        l = Libro()
        l.autore = aut
        l.titolo = tit
        l.pagine = pag
        l.data_prestito = timezone.now()

        try:
            l.save()
            message = "Creazione libro riuscita!" + message
        except Exception as e:
            message = "Errore nella creazione del libro " + str(e) 

    return render(request,template_name="gestione/crealibro.html",
            context={"title":"Crea Autore", "message":message})


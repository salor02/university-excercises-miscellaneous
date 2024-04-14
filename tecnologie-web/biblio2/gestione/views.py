from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from .models import Libro, Copia
from django.utils import timezone

def lista_libri(request):
    templ = "gestione/listalibri.html"
    ctx = { "title":"Lista di Libri", 
           "listalibri": Libro.objects.all(),
           "copie": Copia.objects.all()}
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

        try:
            l.save()
            message = "Creazione libro riuscita!" + message
        except Exception as e:
            message = "Errore nella creazione del libro " + str(e) 

    return render(request,template_name="gestione/crealibro.html",
            context={"title":"Crea Autore", "message":message})

def prestito(request, titolo, autore):
    l = get_object_or_404(Libro, titolo=titolo, autore=autore)

    templ = "gestione/prestito.html"
    
    c = Copia.objects.filter(libro=l, data_prestito=None)
    if len(c) > 0:
        c[0].data_prestito = timezone.now()
        c[0].save()

        ctx = { "title":"Prestito",
                "message": "Prestito confermato! ID copia : ",
                "libro": c[0]}
    
    else:
        ctx = {
                "title":"Prestito",
                "message": "Prestito non effettuato, copie esaurite",
                "libro": None
        }
        
    return render(request,template_name=templ,context=ctx)

def restituzioneSelect(request, titolo, autore):
    l = get_object_or_404(Libro, titolo=titolo, autore=autore)

    templ = "gestione/restituzione.html"

    c = Copia.objects.filter(libro=l).exclude(data_prestito = None)

    ctx = {
            "title":"Selezione restituzione",
            "listacopie": c
            }
        
    return render(request,template_name=templ,context=ctx)

def restituzione(request, id):
    c = get_object_or_404(Copia, id=id)

    templ = "gestione/restituzione.html"

    c.data_prestito = None
    c.save()

    ctx = {
            "title":"Finalizza restituzione"
    }

    if c.scaduto:
        ctx["scaduto"] = True

    return render(request,template_name=templ,context=ctx)


from gestione.models import Libro
from django.utils import timezone
from datetime import datetime

def erase_db():
    print("Cancello il DB")
    Libro.objects.all().delete()

def init_db():
    
    if len(Libro.objects.all()) != 0:
        return

    def func_time(off_year=None, off_month=None, off_day=None):
        tz = timezone.now()
        out = datetime(tz.year-off_year,tz.month-off_month,
                    tz.day-off_day,tz.hour,tz.minute, tz.second)
        return out 

    #se è vuoto lo inizializzo
    #può essere letto da fonti esterne, files, altri DB etc...

    libridict = {
        "autori" : ["Alessandro Manzoni", "George Orwell", "Omero", "George Orwell", "Omero"],
        "titoli" : ["Promessi Sposi", "1984", "Odissea", "La Fattoria degli Animali", "Iliade"],
        "pagine" : [832,328,414,141,263],
        "date" :   [ func_time(y,m,d) for y in range(2) for m in range(2) for d in range(2) ]
    }

    #for k in libridict:
    #    print(str(libridict[k]))

    for i in range(5):
        l = Libro()
        for k in libridict:
            if k == "autori":
                    l.autore = libridict[k][i]
            if k == "titoli":
                    l.titolo = libridict[k][i]
            if k == "pagine":
                    l.pagine = libridict[k][i]
            else:
                l.data_prestito = libridict[k][i] 
        #print(l)
        l.save()
    
    print("DUMP DB")
    print(Libro.objects.all()) #controlliamo
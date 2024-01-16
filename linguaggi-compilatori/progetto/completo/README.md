Alcune note sul progetto e sulle scelte implementative:
1) Una variabile definita internamente ad una funzione oscura eventuali parametri di funzione con lo stesso nome che, a loro volta, oscurano eventuali variabili globali con lo stesso nome.
2) L'inizializzazione di un array deve essere fatta specificandone il numero di elementi tra parentesi quadre ed, eventualmente, tante espressioni tra parentesi graffe quanti sono il numero di elementi dichiarati.
Un'inizializzazione del tipo: var A[3] = {1,2} darà, quindi, errore.
3) Il client del compilatore è rimasto totalmente inviariato rispetto al client sviluppato durante le lezioni, quindi rimane possibile utilizzare le opzioni "-p" (debug parser) e "-s" (debug scanner)
4) Per una maggiore comprensione dell'architettura del front-end è utile visionare il diagramma in driverClassDiagram.html che rappresenta tutte le classi dei nodi dell'AST generato durante il processo di parsing e la loro correlazione. 

Livello raggiunto:
Il compilatore è stato testato e genera codice perfettamente funzionante per tutti i programmi di test: implementa quindi una grammatica di quarto livello.

Sistema utilizzato per test e sviluppo:
SO: Ubuntu 20.04.6 LTS su WSL1
Clang/LLVM: 16.0.6 


# SEMAFORI

### FASE 1: definizione della risorsa condivisa.
Qui si definiscono tutti i semaofori e i contatori necessari per tracciare numero di thread attivi e bloccati
```c
struct myresource_t {
    sem_t mutex;
    sem_t priv[MAXPROC];
    ...
}
```

### FASE 2: inizializzazione del gestore dei thread
Qui si inizializzano i semafori e i mutex e i contatori
```c
void myresource_init(...)
{
    <mutex initialized to 1>
    sem_init(&g->mutex,0,1);
    <private semaphores initialized to 0>
    sem_init(&g->priv_s,0,0);
    ...
}
```

## SEMAFORI PARADIGMA 1
+ PRO: possibilità di risvegliare più task insieme
- CONTRO: preemptive post e aggiornamento dei contatori sia in acquisizione che in rilascio

### FASE DI ACQUISIZIONE DELLA RISORSA CONDIVISA:

```c
sem_wait(&r->mutex);

if <condizione di blocco (quando mi devo bloccare?)> {

    //caso in cui si può proseguire
    <resource allocation to i (contatore dei task attivi ++)>
    sem_post(&r->priv[i]);
    }
else
    //caso di blocco
    <record that i is suspended (contatore dei task bloccati ++)>

sem_post(&r->mutex);
sem_wait(&r->priv[i]);
```

### FASE DI RILASCIO DELLA RISORSA CONDIVISA:

```c
int i;
sem_wait(&r->mutex);

<release the resource (contatore dei task attivi --)>

if <wake up someone (chi devo svegliare?)> {
    i = <process to wake up>

    <resource allocation to i (contatore dei task attivi ++)>
    <record that i is no more suspended (contatore dei task bloccati --)>

    sem_post(&r->priv[i]);
}

sem_post(&r->mutex);
```

## SEMAFORI PARADIGMA 2 (token passing)
+ PRO: no preemptive post e aggiornamento contatori solo in acquisizione
- CONTRO: possibilità di risvegliare solo un task alla volta per via del passaggio del mutex

### FASE DI ACQUISIZIONE DELLA RISORSA CONDIVISA:
```c
sem_wait(&r->mutex);

if <not condition (non si verifica questa condizione -> allora mi blocco)> {
    <record that i is suspended (contatore dei task bloccati ++)>

    sem_post(&r->mutex);
    sem_wait(&r->priv[i]);
    <record that i has been woken up (contatore dei task bloccati --)>
}

<resource allocation to i (contatore dei task attivi ++)>

sem_post(&r->mutex);
```

### FASE DI RILASCIO DELLA RISORSA CONDIVISA:
```c
int i;
sem_wait(&r->mutex); //questo sarà il token da passare eventualmente

<release the resource (contatore dei task attivi --)>

if <wake up someone (chi devo svegliare?)> {
    i = <process to wake up>
    sem_post(&r->priv[i]);
}
else
    sem_post(&r->mutex); //in questo caso non succede nulla e libero solo il mutex
```

# CONDITION VARIABLES + MUTEX

### FASE 1: definizione della risorsa condivisa

```c
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int counter; //eventuali contatori
} mysem_t;
```

### FASE 2: inizializzazione del gestore dei thread

```c
pthread_mutexattr_t m;
pthread_condattr_t c;

//inizializzazione di contatori per tracciare numero di bloccati o qualsiasi contatore utile per policy

//si fa init e destroy degli attr che non tanto non servono
pthread_mutexattr_init(&m);
pthread_mutex_init(&s->mutex, &m);
pthread_mutexattr_destroy(&m);

pthread_condattr_init(&c);
pthread_cond_init(&s->cond, &c);
pthread_condattr_destroy(&c);
```

### FASE DI ACQUISIZIONE DELLA RISORSA CONDIVISA:

```c
pthread_mutex_lock(&r->mutex);

while <not condition (non si verifica questa condizione -> allora mi blocco)> {
    <record that i is suspended (contatore dei task bloccati ++)>

    pthread_cond_wait(&r->priv[i],&r->mutex);

    <record that i has been woken up (contatore dei task bloccati --)>
}

<resource allocation to i (contatore dei task attivi ++)>

pthread_mutex_unlock(&r->mutex);
```

### FASE DI RILASCIO DELLA RISORSA CONDIVISA:

```c
int i;
pthread_mutex_lock(&r->mutex);

<release the resource (contatore dei task attivi --)>

if <wake up someone (chi devo svegliare?)> {
    i = <process to wake up>
    pthread_cond_signal(&r->priv[i]); OPPURE pthread_cond_broadcast(&r->priv[i]);
}

pthread_mutex_unlock(&r->mutex);
```


/*
    Signature di alcune funzioni cambiate per motivi di testing, in particolare:
        - int pedone_entro_passerella(struct passerella_t *passerella, int hofretta, int id_pedone) -> aggiunto id_pedone
        - void pedone_esco_passerella(struct passerella_t *passerella, int hofretta, int id_pedone) -> aggiunti hofretta e id_pedone

    Ho testato il programma consegnato e non ho trovato nessun deadlock. Anche la sequenza delle operazione eseguite mi sembra corretta per come 
    l'avevo pensata. Credo però di aver frainteso l'utilizzo del parametro "hofretta" specificato per ogni thread, più nel dettaglio io ho modellato
    il problema come segue.

    Ho inserito nella struct due bool:
    - "barca" per segnalare che il guardiano ha visto una barca in avvicinamento e deve sgomberare il pontile
    - "abbassato" per segnalare l'effettivo stato attuale del pontile

    Il sistema utilizza due bool per distinguere il caso in cui un pedone dovesse avere fretta, in particolare il sistema si comporta come segue:
    - se il pedone non ha fretta e il pontile è abbassato OR la barca è in arrivo => va in stato di attesa
    - se il pedone ha fretta, basta che il pontile sia abbassato per procedere con l'esecuzione. Anche se ci dovesse essere una barca in arrivo, quindi, 
    i pedoni che hanno fretta passano comunque. Un caso simile nel mondo reale è dato da un passaggio a livello in cui finchè le sbarre non sono abbassate qualche 
    macchina potrebbe continuare effettivamente a passare. L'unico modo per fermare un pedone di fretta è quindi quello di alzare il pontile.

    C'è un problema però! Questo ragionamento che nella mia testa aveva senso perde di significato dato che è l'ultimo pedone che si deve occupare di svegliare il 
    guardiano. Questo significa che con un numero di thread sufficientemente alto è possibile prolungare indefinitivamente l'attesa del guardiano, anche in caso di
    barca in arrivo.

    Per confermare questo problema si esegua questo programma di testing impostando N ad almeno 100, ATTESA_GUARDIANO_MS e TEMPO_ESECUZIONE_MAX a 50ms di 
    differenza. Si stampi l'output del programma in un file e si cerchi la stringa "barca in arrivo!" che sarà sicuramente presente, e la poi la stringa
    "lascio passare la barca", che talvolta invece risulta assente. Un risultato del genere dimostra che il guardiano va in stato di attesa ma non riesce
    a far passare la barca nel giro di 50ms, tempo che si dilata se si aumenta N. Con un sistema modellato per impostare il parametro "hofretta" in maniera
    casuale tra 0 e 1, prima o poi tutti i thread verranno generati con "hofretta" = 0 e si bloccheranno tutti, lasciando passare la barca, ma questo
    comportamento non rispetta la precedenza che la barca dovrebbe avere come richiesto nel testo. Inoltre, nel caso in cui tutti i pedoni abbiano fretta 
    questo programma presenta un vero e proprio problema di starvation del guardiano.
*/

#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

#define N 100
#define ATTESA_GUARDIANO_MS 2000
#define TEMPO_ESECUZIONE_MAX 2050

struct passerella_t{
    pthread_mutex_t mutex;
    pthread_cond_t priv_pedoni, priv_guardiano;

    int active_pedoni, block_pedoni;
    bool barca, abbassato;
} passerella;

void init_passerella(struct passerella_t *passerella){
    pthread_mutexattr_t m;
    pthread_condattr_t c;

    pthread_mutexattr_init(&m);
    pthread_mutex_init(&passerella->mutex, &m);
    pthread_mutexattr_destroy(&m);

    pthread_condattr_init(&c);
    pthread_cond_init(&passerella->priv_pedoni, &c);
    pthread_cond_init(&passerella->priv_guardiano, &c);
    pthread_condattr_destroy(&c);

    passerella->active_pedoni = passerella->block_pedoni = 0;
    passerella->barca = passerella->abbassato = false;
}

int pedone_entro_passerella(struct passerella_t *passerella, int hofretta, int id_pedone){
    pthread_mutex_lock(&passerella->mutex);
    
    if(hofretta){
        if(!passerella->abbassato){
            pthread_mutex_unlock(&passerella->mutex);
            return 0;
        }
        printf("[pedone %d, hofretta=%d] passerella abbassata, vado\n", id_pedone, hofretta);
    }
    else{
        while(passerella->barca || !passerella->abbassato){
            printf("[pedone %d, hofretta=%d] passerella chiusa o barca in arrivo, attendo\n", id_pedone, hofretta);
            passerella->block_pedoni++;
            pthread_cond_wait(&passerella->priv_pedoni, &passerella->mutex);
            passerella->block_pedoni--;
        }
    }

    passerella->active_pedoni++;
    
    pthread_mutex_unlock(&passerella->mutex);
    return 1;
}

void pedone_esco_passerella(struct passerella_t *passerella, int hofretta, int id_pedone){
    pthread_mutex_lock(&passerella->mutex);
    
    passerella->active_pedoni--;

    if(passerella->active_pedoni == 0 && passerella->barca){
        printf("[pedone %d, hofretta=%d] sono l'ultimo pedone, sveglio il guardiano\n", id_pedone, hofretta);
        pthread_cond_signal(&passerella->priv_guardiano);
    }

    pthread_mutex_unlock(&passerella->mutex);
}

void guardiano_abbasso_passerella(struct passerella_t *passerella){
    pthread_mutex_lock(&passerella->mutex);
    
    printf("[guardiano] abbasso passerella\n");

    passerella->abbassato = true;
    passerella->barca = false;

    if(passerella->block_pedoni){
        printf("[guardiano] alcuni pedoni erano bloccati, sveglia\n");
        pthread_cond_broadcast(&passerella->priv_pedoni);
    }

    pthread_mutex_unlock(&passerella->mutex);
}

void guardiano_alzo_passerella(struct passerella_t *passerella){
    pthread_mutex_lock(&passerella->mutex);
    
    printf("[guardiano] barca in arrivo!\n");
    passerella->barca = true;

    while(passerella->active_pedoni){
        printf("[guardiano] barca in arrivo ma pedoni presenti, attendo\n");
        pthread_cond_wait(&passerella->priv_guardiano, &passerella->mutex);
    }

    printf("[guardiano] nessun pedone in attraversamento, alzo passerella\n");
    passerella->abbassato = false;
    
    pthread_mutex_unlock(&passerella->mutex);
}

//----- TESTING -----
void pausetta(void)
{
    struct timespec t;
    t.tv_sec = 0;
    t.tv_nsec = (rand()%10+1)*1000000;
    nanosleep(&t,NULL);
}

void *pedone(void *arg){
    int id = (int)(intptr_t)arg;

    while(1){
        int hofretta = rand()%2;
        printf("[pedone %d, hofretta=%d] arrivato\n", id, hofretta);

        if(pedone_entro_passerella(&passerella, hofretta, id)){
            printf("[pedone %d, hofretta=%d] attraverso\n", id, hofretta);
            pausetta();
            pedone_esco_passerella(&passerella, hofretta, id);
            printf("[pedone %d, hofretta=%d] vado via\n", id, hofretta);
        }
        else
            printf("[pedone %d, hofretta=%d] passerella chiusa, cambio strada\n", id, hofretta);
        
    }

    return 0;
}

void *guardiano(void *arg){
    while(1){
        guardiano_abbasso_passerella(&passerella);
        printf("[guardiano] attendo barca\n");
        usleep(ATTESA_GUARDIANO_MS * 1000);
        guardiano_alzo_passerella(&passerella);
        printf("[guardiano] lascio passare la barca\n");
    }    
}

int main()
{
    pthread_attr_t a;
    pthread_t p;
    
    /* inizializzo il mio sistema */
    init_passerella(&passerella);

    /* inizializzo i numeri casuali, usati nella funzione pausetta */
    srand(555);

    pthread_attr_init(&a);

    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);

    for(int i = 0; i < N; i++){
        pthread_create(&p, &a, pedone, (void *)(intptr_t) i);
    }

    pthread_create(&p, &a, guardiano, NULL);

    pthread_attr_destroy(&a);

    usleep(TEMPO_ESECUZIONE_MAX * 1000);

    return 0;
}
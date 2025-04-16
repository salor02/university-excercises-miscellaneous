/*
    Signature di alcune funzioni cambiate per motivi di testing, in particolare:
        - int pedone_entro_passerella(struct passerella_t *passerella, int hofretta, int id_pedone) -> aggiunto id_pedone
        - void pedone_esco_passerella(struct passerella_t *passerella, int hofretta, int id_pedone) -> aggiunti hofretta e id_pedone
*/

#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

#define N 10
#define ATTESA_GUARDIANO_MS 2000
#define TEMPO_ESECUZIONE_MAX 3000

struct passerella_t{
    pthread_mutex_t mutex;
    pthread_cond_t priv_pedoni, priv_guardiano;

    /*
    - active_pedoni è utile sia per un pedone, per verificare se è l'ultimo attivo e svegliare eventualmente il guardiano,
    sia per il guardiano, per verificare se sono presenti pedoni sulla passerella
    
    - block_pedoni è utile al guardiano per verificare se deve svegliare qualche pedone rimasto in attesa prima di attraversare il pontile
    
    NB: active_pedoni denota i pedoni attualmente in fase di attraversamento della passerella, 
        block_pedoni denota i pedoni attualmente in fase di attesa prima di attraversare la passerella
    */
    int active_pedoni, block_pedoni;

    /*
    - barca serve a segnalare che il guardiano ha visto una barca in avvicinamento e deve sgomberare il pontile (questa variabile si può leggere anche come la variabile
    dedicata a segnalare lo stato di wait del guardiano, una sorta di "block_guardiano", per come avevamo visto negli esercizi in classe)
    - abbassato serve a segnalare l'effettivo stato attuale del pontile
    */
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

/*
    Versione originale della funzione, con problema di starvation
    
    Se un pedone ha fretta può comunque passare a patto che la passerella sia abbassata, anche se è stata avvistata una barca (guardiano in stato di wait)
*/
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

/*
    Versione modificata della funzione, con problema di starvation risolto
    
    In questo caso qualsiasi pedone, che abbia fretta o meno, non può attraversare la passerella se è stata avvistata una barca.
    La differenza sta nel fatto che un pedone che ha fretta esce subito se fallisce il controllo, mentre un pedone che non ha fretta si mette
    in attesa. Si noti che la funzione non è mai bloccante per un pedone che ha fretta e che può passare, dato che non può mai entrare nel while. 
*/
int pedone_entro_passerella_v2(struct passerella_t *passerella, int hofretta, int id_pedone){
    pthread_mutex_lock(&passerella->mutex);
    
    if(hofretta)
        if(passerella->barca || !passerella->abbassato){

            //le successive 2 linee sono state aggiunte per debug
            if(passerella->barca) printf("[pedone %d, hofretta=%d] barca in arrivo, cambio strada\n", id_pedone, hofretta);
            if(!passerella->abbassato) printf("[pedone %d, hofretta=%d] passerella non abbassata, cambio strada\n", id_pedone, hofretta);

            pthread_mutex_unlock(&passerella->mutex);
            return 0;
        }

    while(passerella->barca || !passerella->abbassato){

        //le successive 2 linee sono state aggiunte per debug
        if(passerella->barca) printf("[pedone %d, hofretta=%d] barca in arrivo, attendo\n", id_pedone, hofretta);
        if(!passerella->abbassato) printf("[pedone %d, hofretta=%d] passerella non abbassata, attendo\n", id_pedone, hofretta);

        passerella->block_pedoni++;
        pthread_cond_wait(&passerella->priv_pedoni, &passerella->mutex);
        passerella->block_pedoni--;
    }

    passerella->active_pedoni++;
    
    pthread_mutex_unlock(&passerella->mutex);
    return 1;
}

/*
    L'ultimo pedone a passare si deve occupare di svegliare il guardiano. Si noti che nella condizione viene anche verificata l'effettiva presenza di una barca e, di conseguenza, 
    la presenza di un guardiano in stato di wait. Questo secondo controllo è evitabile dato che il signal sulla variabile condizionale andrebbe semplicemente "perso" in caso
    di guardiano non in attesa. 
*/
void pedone_esco_passerella(struct passerella_t *passerella, int hofretta, int id_pedone){
    pthread_mutex_lock(&passerella->mutex);
    
    passerella->active_pedoni--;

    if(passerella->active_pedoni == 0 && passerella->barca){
        printf("[pedone %d, hofretta=%d] sono l'ultimo pedone, sveglio il guardiano\n", id_pedone, hofretta);
        pthread_cond_signal(&passerella->priv_guardiano);
    }

    pthread_mutex_unlock(&passerella->mutex);
}

/*
    Il guardiano resetta entrambe le variabili bool in modo da permettere ad aventuali pedoni di attraversare la passerella. In caso ci siano pedoni bloccati, 
    vengono svegliati tutti. Si noti che anche in questo caso l'if è evitabile, dato l'utilizzo delle variabili condizionali. 
*/
void guardiano_abbasso_passerella(struct passerella_t *passerella){
    pthread_mutex_lock(&passerella->mutex);
    
    printf("[guardiano] abbasso passerella\n");

    passerella->abbassato = true;
    //passerella->barca = false; linea commentata per lieve modifica alla funzione successiva (vedi commento a guardiano_alzo_passerella)

    if(passerella->block_pedoni){
        printf("[guardiano] alcuni pedoni erano bloccati, sveglia\n");
        pthread_cond_broadcast(&passerella->priv_pedoni);
    }

    pthread_mutex_unlock(&passerella->mutex);
}

/*
    Il guardiano controlla se sono presenti pedoni sulla passerella, in caso affermativo segnala la presenza di una barca in arrivo, ovvero segnala che sta entrando
    in fase di wait. Una volta svuotata la passerella e acquisito il mutex, il guardiano setta "barca" a zero per segnalare il fatto che non si trova più in stato di
    wait e procede a far passare la barca.

    Più precisamente, se un guardiano trova pedoni sulla passerella attende che tutti escano. Deve però dare precedenza alla barca: per fare questo setta la variabile "barca" a true, 
    il che impedisce ad altri pedoni di attraversare la passerella, anche se durante l'attesa del guardiano questa è effettivamente ancora abbassata.
    Una volta riottenuto il mutex, il guardiano segnala il fatto che la passerella non è abbassata settando il valore dell'apposita variabile a false. In questa fase ogni pedone
    viene respinto se tenta di avere accesso alla passerella perchè "abbassato" è settato a false.

    Si noti che questa funzione risulta lievemente modificata rispetto alla versione consegnata anche se il funzionamento è identico. Nella versione originale la variabile "barca"
    veniva settata a true prima del while, per poi essere settata a false al momento dell'abbassamento della passerella (ecco perchè una riga della funzione guardiano_abbasso_passerella
    è commentata). I pedoni verrebbero bloccati in ogni caso ma ho effettuato questa modifica per rimanere più fedele al paradigma delle cond var e per avere messaggi di debug 
    più precisi.
*/
void guardiano_alzo_passerella(struct passerella_t *passerella){
    pthread_mutex_lock(&passerella->mutex);
    
    printf("[guardiano] barca in arrivo!\n");

    while(passerella->active_pedoni){
        passerella->barca = true;
        printf("[guardiano] barca in arrivo ma pedoni presenti, attendo\n");
        pthread_cond_wait(&passerella->priv_guardiano, &passerella->mutex);
        passerella->barca = false;
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
        int hofretta = rand()%2; //assegnazione random della fretta
        printf("[pedone %d, hofretta=%d] arrivato\n", id, hofretta);

        //utilizza la funzione corretta, omettere v2 per utilizzare la funzione originale
        if(pedone_entro_passerella_v2(&passerella, hofretta, id)){
            printf("[pedone %d, hofretta=%d] attraverso\n", id, hofretta);
            pausetta();
            pedone_esco_passerella(&passerella, hofretta, id);
            printf("[pedone %d, hofretta=%d] vado via\n", id, hofretta);
        }
        else
            printf("[pedone %d, hofretta=%d] ha cambiato strada\n", id, hofretta);
        
        pausetta();
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
        usleep(ATTESA_GUARDIANO_MS * 100);
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
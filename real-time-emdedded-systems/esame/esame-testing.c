#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

#define N 10

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

int pedone_entro_passerella(struct passerella_t *passerella, int hofretta){
    pthread_mutex_lock(&passerella->mutex);
    
    if(hofretta){
        if(!passerella->abbassato){
            pthread_mutex_unlock(&passerella->mutex);
            return 0;
        }
        printf("[pedone hofretta=%d] passerella abbassata, vado\n", hofretta);
    }
    else{
        while(passerella->barca || !passerella->abbassato){
            printf("[pedone hofretta=%d] passerella chiusa o barca in arrivo, attendo\n", hofretta);
            passerella->block_pedoni++;
            pthread_cond_wait(&passerella->priv_pedoni, &passerella->mutex);
            passerella->block_pedoni--;
        }
    }

    passerella->active_pedoni++;
    
    pthread_mutex_unlock(&passerella->mutex);
    return 1;
}

void pedone_esco_passerella(struct passerella_t *passerella){
    pthread_mutex_lock(&passerella->mutex);
    
    passerella->active_pedoni--;

    if(passerella->active_pedoni == 0 && passerella->barca){
        printf("[pedone] sono l'ultimo pedone, sveglio il guardiano\n");
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

        if(pedone_entro_passerella(&passerella, hofretta)){
            printf("[pedone %d, hofretta=%d] attraverso\n", id, hofretta);
            pausetta();
            pedone_esco_passerella(&passerella);
            printf("[pedone %d, hofretta=%d] vado via\n", id, hofretta);
        }
        else
            printf("[pedone %d, hofretta=%d] passerella chiusa, cambio strada\n", id, hofretta);
        
        sleep(10);
    }

    return 0;
}

void *guardiano(void *arg){
    while(1){
        guardiano_abbasso_passerella(&passerella);
        printf("[guardiano] attendo barca\n");
        pausetta();
        guardiano_alzo_passerella(&passerella);
        printf("[guardiano] lascio passare la barca\n");
        sleep(5);
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

    /* non ho voglia di scrivere 10000 volte join! */
    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);

    for(int i = 0; i < N; i++){
        pthread_create(&p, &a, pedone, (void *)(intptr_t) i);
    }

    pthread_create(&p, &a, guardiano, NULL);

    pthread_attr_destroy(&a);

    /* aspetto 10 secondi prima di terminare tutti quanti */
    sleep(10);

    return 0;
}
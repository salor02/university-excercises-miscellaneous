#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

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
    }
    else{
        while(passerella->barca || !passerella->abbassato){
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

    if(passerella->active_pedoni == 0 && passerella->barca)
        pthread_cond_signal(&passerella->priv_guardiano);

    pthread_mutex_unlock(&passerella->mutex);
}

void guardiano_abbasso_passerella(struct passerella_t *passerella){
    pthread_mutex_lock(&passerella->mutex);
    
    passarella->abbassato = true;
    passarella->barca = false;

    if(passerella->block_pedoni){
        pthread_cond_broadcast(&passerella->priv_pedoni);
    }

    pthread_mutex_unlock(&passerella->mutex);
}

void guardino_alzo_passerella(struct passerella_t *passerella){
    pthread_mutex_lock(&passerella->mutex);
    
    passerella->barca = true;

    while(passerella->active_pedoni){
        pthread_cond_wait(&passerella->priv_guardiano, &passerella->mutex);
    }

    passerella->abbassato = false;
    
    pthread_mutex_unlock(&passerella->mutex);
}
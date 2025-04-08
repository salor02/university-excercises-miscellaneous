/*
    - corridore si blocca se corridori bloccati < n e sveglia arbitro se è ultimo
    - arbitro si blocca se corridori bloccati < n
    - arbitro_via sveglia i corridori
    - arbitro_risultato si blocca se array arrivati non è pieno
    - corridore_arrivo inserisce proprio id nell'array e se è l'ultimo
      sveglia l'arbitro
      
    variabili:
        - int arrivati[N]
        - int active_corridori
        - int block_corridori
*/

#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

#define N 10
#define SEM

#ifdef SEM
struct corsa_t{
    sem_t mutex;
    sem_t priv_arbitro, priv_corridori;

    int active_corridori, block_corridori;

    int primo, ultimo;
}corsa;

void init_corsa(struct corsa_t *corsa){
    sem_init(&corsa->mutex,0,1);
    sem_init(&corsa->priv_arbitro,0,0);
    sem_init(&corsa->priv_corridori,0,0);

    corsa->active_corridori = corsa->block_corridori = 0;

    //inizializzazione a -1 per segnalare posizione vuota
    corsa->primo = corsa->ultimo = -1;
}

//blocco se block_corridori < n
void corridore_attendivia(struct corsa_t *corsa, int numerocorridore){
    sem_wait(&corsa->mutex);

    if(corsa->block_corridori < N){
        corsa->block_corridori++;
        if(corsa->block_corridori == N){
            sem_post(&corsa->priv_arbitro);
        }
    }

    sem_post(&corsa->mutex);
    sem_wait(&corsa->priv_corridori);
}

//sveglia arbitro se è l'ultimo ad arrivare cioe se active_corridori = 0
void corridore_arrivo(struct corsa_t *corsa, int numerocorridore){
    sem_wait(&corsa->mutex);

    if(corsa->active_corridori == N) corsa->primo = numerocorridore;

    corsa->active_corridori--;

    if(corsa->active_corridori == 0){
        corsa->ultimo = numerocorridore;
        sem_post(&corsa->priv_arbitro);
    }

    sem_post(&corsa->mutex);
}

//blocco se block_corridori < n, proseguo se = N
void arbitro_attendicorridori(struct corsa_t *corsa){
    sem_wait(&corsa->mutex);

    if(corsa->block_corridori == N){
        sem_post(&corsa->priv_arbitro);
    }

    sem_post(&corsa->mutex);
    sem_wait(&corsa->priv_arbitro);
}

void arbitro_via(struct corsa_t *corsa){
    sem_wait(&corsa->mutex);

    while(corsa->block_corridori){
        corsa->active_corridori++;
        corsa->block_corridori--;
        sem_post(&corsa->priv_corridori);
    }

    sem_post(&corsa->mutex);
}

//blocca se non sono arrivati tutti i corridori cioè se active_corridori > 0
void arbitro_risultato(struct corsa_t *corsa, int *primo, int *ultimo){
    sem_wait(&corsa->mutex);

    if(corsa->active_corridori == 0){
        sem_post(&corsa->priv_arbitro);
    }

    sem_post(&corsa->mutex);
    sem_wait(&corsa->priv_arbitro);
    
    *primo = corsa->primo;
    *ultimo = corsa->ultimo;

}

#elif defined COND
struct corsa_t{
    pthread_mutex_t mutex;
    pthread_cond_t priv_arbitro, priv_corridori;
    
    int block_corridori, active_corridori;

    bool via;

    int primo, ultimo;
}corsa;

void init_corsa(struct corsa_t *corsa){
    pthread_mutexattr_t m;
    pthread_condattr_t c;

    pthread_mutexattr_init(&m);
    pthread_mutex_init(&corsa->mutex, &m);
    pthread_mutexattr_destroy(&m);

    pthread_condattr_init(&c);
    pthread_cond_init(&corsa->priv_arbitro, &c);
    pthread_cond_init(&corsa->priv_corridori, &c);
    pthread_condattr_destroy(&c);

    corsa->block_corridori = corsa->active_corridori = 0;
    corsa->primo = corsa->ultimo = -1;

    corsa->via = false;
}

//blocco se block_corridori < n
void corridore_attendivia(struct corsa_t *corsa, int numerocorridore){
    pthread_mutex_lock(&corsa->mutex);

    while(!corsa->via){
        corsa->block_corridori++;
        if(corsa->block_corridori == N) 
            pthread_cond_signal(&corsa->priv_arbitro);
        pthread_cond_wait(&corsa->priv_corridori, &corsa->mutex);
        corsa->block_corridori--;
    }

    corsa->active_corridori++;

    pthread_mutex_unlock(&corsa->mutex);

}

//sveglia arbitro se è l'ultimo ad arrivare cioe se active_corridori = 0
void corridore_arrivo(struct corsa_t *corsa, int numerocorridore){
    pthread_mutex_lock(&corsa->mutex);

    corsa->active_corridori--;

    if(corsa->primo == -1) corsa->primo = numerocorridore;

    if(!corsa->active_corridori && !corsa->block_corridori){
        corsa->ultimo = numerocorridore;
        pthread_cond_signal(&corsa->priv_arbitro);
    }

    pthread_mutex_unlock(&corsa->mutex);
}

//blocco se block_corridori < n, proseguo se = N
void arbitro_attendicorridori(struct corsa_t *corsa){
    pthread_mutex_lock(&corsa->mutex);

    while(corsa->block_corridori < N){
        pthread_cond_wait(&corsa->priv_arbitro, &corsa->mutex);
    }

    pthread_mutex_unlock(&corsa->mutex);
}

void arbitro_via(struct corsa_t *corsa){
    pthread_mutex_lock(&corsa->mutex);
    
    corsa->via = true;

    pthread_cond_broadcast(&corsa->priv_corridori);

    pthread_mutex_unlock(&corsa->mutex);
}

//blocca se non sono arrivati tutti i corridori cioè se active_corridori > 0
void arbitro_risultato(struct corsa_t *corsa, int *primo, int *ultimo){
    pthread_mutex_lock(&corsa->mutex);

    while(corsa->active_corridori > 0 || corsa->block_corridori > 0){
        pthread_cond_wait(&corsa->priv_arbitro, &corsa->mutex);
    }

    *primo = corsa->primo;
    *ultimo = corsa->ultimo;

    pthread_mutex_unlock(&corsa->mutex);
}
#endif


void pausetta(void)
{
  struct timespec t;
  t.tv_sec = 0;
  t.tv_nsec = (rand()%10+1)*1000000;
  nanosleep(&t,NULL);
}

void *corridore(void* arg)
{
    fprintf(stderr,"p");
    corridore_attendivia(&corsa, (int)(intptr_t)arg);
    fprintf(stderr,"c");
    pausetta();
    corridore_arrivo(&corsa, (int)(intptr_t)arg);
    fprintf(stderr,"a");

    return 0;
}

void *arbitro()
{
    int primo, ultimo;
    fprintf(stderr,"P");
    arbitro_attendicorridori(&corsa);
    fprintf(stderr,"V");
    arbitro_via(&corsa);
    fprintf(stderr,"F");
    arbitro_risultato(&corsa, &primo, &ultimo);
    //pausetta();

    fprintf(stderr,"%d",primo);
    fprintf(stderr,"%d",ultimo);
    return 0;
}

int main()
{
  pthread_attr_t a;
  pthread_t p;
  
  /* inizializzo il mio sistema */
  init_corsa(&corsa);

  /* inizializzo i numeri casuali, usati nella funzione pausetta */
  srand(555);

  pthread_attr_init(&a);

  /* non ho voglia di scrivere 10000 volte join! */
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);

  for(int i = 0; i < N; i++)
    pthread_create(&p, &a, corridore, (void *)(intptr_t)i);


  pthread_create(&p, &a, arbitro, NULL);

  pthread_attr_destroy(&a);

  /* aspetto 10 secondi prima di terminare tutti quanti */
  sleep(2);

  fprintf(stderr,"%d",corsa.ultimo);

  return 0;
}
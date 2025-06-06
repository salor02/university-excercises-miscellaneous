/*
    a -> b (n volte)
    c -> e o c -> d
*/
#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#define STATO_INIT 0
#define STATO_AC 1
#define STATO_B 2
#define STATO_DE 3

#define COND

#ifdef SEM

struct gestione_t {
    sem_t mutex;
    sem_t priv_AC, priv_B, priv_DE;

    int block_AC,block_B,block_DE;
    int active_AC, active_B, active_DE;

    int status;
} gestore;

void init_g(struct gestione_t *g){
    sem_init(&g->mutex, 0, 1);

    sem_init(&g->priv_AC, 0, 0);
    sem_init(&g->priv_B, 0, 0);
    sem_init(&g->priv_DE, 0, 0);

    g->block_AC = 0;
    g->block_B = 0;
    g->block_DE = 0;

    g->active_B = 0;

    g->status = STATO_AC;
}

//if STATO_AC && active_AC == 0
void startA(struct gestione_t *g){
    sem_wait(&g->mutex);

    if(g->status == STATO_AC && g->active_AC == 0){
        g->active_AC++;
        sem_post(&g->priv_AC);
    }
    else
        g->block_AC++;

    sem_post(&g->mutex);
    sem_wait(&g->priv_AC);
}

//STATO_AC -> STATO_B
void endA(struct gestione_t *g){
    sem_wait(&g->mutex);

    g->active_AC--;

    while(g->block_B){
        g->active_B++;
        g->block_B--;
        sem_post(&g->priv_B);
    }

    g->status = STATO_B;

    sem_post(&g->mutex);
}

//if STATO_B
void startB(struct gestione_t *g){
    sem_wait(&g->mutex);

    if(g->status == STATO_B){
        g->active_B++;
        sem_post(&g->priv_B);
    }
    else
        g->block_B++;

    sem_post(&g->mutex);
    sem_wait(&g->priv_B);
}

//if ultimo STATO_B -> STATO_AC
void endB(struct gestione_t *g){
    sem_wait(&g->mutex);

    g->active_B--;

    if(g->active_B == 0){
        g->status = STATO_AC;

        if(g->block_AC){
            g->active_AC++;
            g->block_AC--;
            sem_post(&g->priv_AC);
        }
    }


    sem_post(&g->mutex);
}

//if STATO_AC && active_AC == 0
void startC(struct gestione_t *g){
    startA(g);
}

//STATO_AC -> STATO_DE
void endC(struct gestione_t *g){
    sem_wait(&g->mutex);

    g->active_AC--;

    if(g->block_DE){
        g->active_DE++;
        g->block_DE--;
        sem_post(&g->priv_DE);
    }

    g->status = STATO_DE;

    sem_post(&g->mutex);
}

//if STATO_DE && active_DE == 0
void startD(struct gestione_t *g){
    sem_wait(&g->mutex);

    if(g->status == STATO_DE && g->active_DE == 0){
        g->active_DE++;
        sem_post(&g->priv_DE);
    }
    else
        g->block_DE++;

    sem_post(&g->mutex);
    sem_wait(&g->priv_DE);
}

//STATO_DE -> STATO_AC
void endD(struct gestione_t *g){
    sem_wait(&g->mutex);

    g->active_DE--;

    if(g->block_AC){
        g->active_AC++;
        g->block_AC--;
        sem_post(&g->priv_AC);
    }
    
    g->status = STATO_AC;

    sem_post(&g->mutex);
}

//if STATO_DE && active_DE == 0
void startE(struct gestione_t *g){
    startD(g);
}

//STATO_DE -> STATO_AC
void endE(struct gestione_t *g){
    endD(g);
}

#elif defined COND

struct gestione_t {
    pthread_mutex_t mutex;
    pthread_cond_t priv_AC, priv_B, priv_DE;

    int active_AC, active_B, active_DE;
    int block_AC, block_B, block_DE;

    int status;
} gestore;

void init_g(struct gestione_t *g){
    pthread_mutexattr_t m;
    pthread_condattr_t c;

    pthread_mutexattr_init(&m);
    pthread_mutex_init(&g->mutex, &m);
    pthread_mutexattr_destroy(&m);

    pthread_condattr_init(&c);
    pthread_cond_init(&g->priv_AC, &c);
    pthread_cond_init(&g->priv_B, &c);
    pthread_cond_init(&g->priv_DE, &c);
    pthread_condattr_destroy(&c);

    g->active_AC = g->active_DE = g->active_B = g->block_AC = g->block_B = g->block_DE = 0;

    g->status = STATO_INIT;
}

void sveglia_A_o_C(struct gestione_t *g){
    if(g->block_AC){
        pthread_cond_signal(&g->priv_AC);
    }
    g->status = STATO_INIT;
    
}

//blocco se stato != STATO_INIT, transizione a STATO_AC 
void startA(struct gestione_t *g){
    pthread_mutex_lock(&g->mutex);

    while(g->status != STATO_INIT){
        g->block_AC++;
        pthread_cond_wait(&g->priv_AC, &g->mutex);
        g->block_AC--;
    }

    g->active_AC++;
    g->status = STATO_AC;

    pthread_mutex_unlock(&g->mutex);
}

//transizione a STATO_B, sveglio B se block_B > 0
void endA(struct gestione_t *g){
    pthread_mutex_lock(&g->mutex);
    
    g->active_AC--;

    if(g->block_B){
        pthread_cond_broadcast(&g->priv_B);
    }

    g->status = STATO_B;

    pthread_mutex_unlock(&g->mutex);
}

//blocco se stato != STATO_B
void startB(struct gestione_t *g){
    pthread_mutex_lock(&g->mutex);
    
    while(g->status != STATO_B){
        g->block_B++;
        pthread_cond_wait(&g->priv_B, &g->mutex);
        g->block_B--;
    }

    g->active_B++;

    pthread_mutex_unlock(&g->mutex);
}

//se block_AC > 0, signal su priv_AC e transizione su STATO_AC, altrimenti transizione su STATO_INIT
void endB(struct gestione_t *g){
    pthread_mutex_lock(&g->mutex);
    
    g->active_B--;

    if(g->active_B == 0){
        sveglia_A_o_C(g);
    }

    pthread_mutex_unlock(&g->mutex);
}

//blocco se stato != STATO_INIT, transizione a STATO_AC
void startC(struct gestione_t *g){
    startA(g);
}

//transizione a STATO_DE, signal su priv_DE se block_DE > 0
void endC(struct gestione_t *g){
    pthread_mutex_lock(&g->mutex);

    g->active_AC--;
    
    if(g->block_DE){
        pthread_cond_signal(&g->priv_DE);
    }

    g->status = STATO_DE;

    pthread_mutex_unlock(&g->mutex);
}

//blocco se stato != STATO_DE
void startD(struct gestione_t *g){
    pthread_mutex_lock(&g->mutex);
    
    while(g->status != STATO_DE || g->active_DE){
        g->block_DE++;
        pthread_cond_wait(&g->priv_DE, &g->mutex);
        g->block_DE--;
    }

    g->active_DE++;

    pthread_mutex_unlock(&g->mutex);
}

//se block_AC > 0, signal su priv_AC e transizione su STATO_AC, altrimenti transizione su STATO_INIT
void endD(struct gestione_t *g){
    pthread_mutex_lock(&g->mutex);
    
    g->active_DE--;

    sveglia_A_o_C(g);

    pthread_mutex_unlock(&g->mutex);
}

//blocco se stato != STATO_DE
void startE(struct gestione_t *g){
    startD(g);
}

//se block_AC > 0, signal su priv_AC e transizione su STATO_AC, altrimenti transizione su STATO_INIT
void endE(struct gestione_t *g){
    endD(g);
}

#endif

/* -------------- TESTING --------------- */


/* alla fine di ogni ciclo ogni thread aspetta un po'.
   Cosa succede se tolgo questa nanosleep? 
   di fatto solo i thread di tipo STATO_B riescono ad entrare --> starvation!!!!
   (provare per credere)
*/
void pausetta(void)
{
  struct timespec t;
  t.tv_sec = 0;
  t.tv_nsec = (rand()%10+1)*1000000;
  nanosleep(&t,NULL);
}

/* i thread */


void *A(void *arg)
{
  for (;;) {
    startA(&gestore);
    putchar(*(char *)arg);
    endA(&gestore);
    pausetta();
  }
  return 0;
}

void *B(void *arg)
{
  for (;;) {
    startB(&gestore);
    putchar(*(char *)arg);
    endB(&gestore);
    pausetta();
  }
  return 0;
}

void *C(void *arg)
{
  for (;;) {
    startC(&gestore);
    putchar(*(char *)arg);
    endC(&gestore);
    pausetta();
  }
  return 0;
}

void *D(void *arg)
{
  for (;;) {
    startD(&gestore);
    putchar(*(char *)arg);
    endD(&gestore);
    pausetta();
  }
  return 0;
}

void *E(void *arg)
{
  for (;;) {
    startE(&gestore);
    putchar(*(char *)arg);
    endE(&gestore);
    pausetta();
  }
  return 0;
}


/* la creazione dei thread */



int main()
{
  pthread_attr_t a;
  pthread_t p;
  
  /* inizializzo il mio sistema */
  init_g(&gestore);

  /* inizializzo i numeri casuali, usati nella funzione pausetta */
  srand(555);

  pthread_attr_init(&a);

  /* non ho voglia di scrivere 10000 volte join! */
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);

  pthread_create(&p, &a, A, (void *)"a");
  pthread_create(&p, &a, A, (void *)"A");

  pthread_create(&p, &a, B, (void *)"B");
  pthread_create(&p, &a, B, (void *)"b");
  pthread_create(&p, &a, B, (void *)"x");

  pthread_create(&p, &a, C, (void *)"C");
  pthread_create(&p, &a, C, (void *)"c");

  pthread_create(&p, &a, D, (void *)"D");
  pthread_create(&p, &a, D, (void *)"d");

  pthread_create(&p, &a, E, (void *)"E");
  pthread_create(&p, &a, E, (void *)"e");

  pthread_attr_destroy(&a);

  /* aspetto 10 secondi prima di terminare tutti quanti */
  sleep(5);

  return 0;
}
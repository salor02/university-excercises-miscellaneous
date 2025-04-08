#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

struct resource{
    pthread_mutex_t mutex;
    pthread_cond_t priv_ab, priv_reset;

    int active_ab, block_ab;
    bool active_reset, block_reset;
} gestore;

void init_resource(struct resource* r){
    pthread_mutexattr_t m;
    pthread_condattr_t c;

    pthread_mutexattr_init(&m);
    pthread_mutex_init(&r->mutex, &m);
    pthread_mutexattr_destroy(&m);

    pthread_condattr_init(&c);
    pthread_cond_init(&r->priv_ab, &c);
    pthread_cond_init(&r->priv_reset, &c);
    pthread_condattr_destroy(&c);

    r->active_ab = r->block_ab = 0;
    r->active_reset = r->block_reset = false;
}

//ok if r non sta eseguendo
void startA(struct resource* r){
    pthread_mutex_lock(&r->mutex);

    while(r->active_reset || r->block_reset){
        r->block_ab++;
        pthread_cond_wait(&r->priv_ab, &r->mutex);
        r->block_ab--;
    }

    r->active_ab++;

    pthread_mutex_unlock(&r->mutex);
}

//risveglio r se bloccato
void endA(struct resource* r){
    pthread_mutex_lock(&r->mutex);

    r->active_ab--;

    if(r->block_reset && r->active_ab == 0){
        pthread_cond_signal(&r->priv_reset);
    }

    pthread_mutex_unlock(&r->mutex);
}

//ok if r non sta eseguendo
void startB(struct resource* r){
    startA(r);
}

//risveglio r se bloccato
void endB(struct resource* r){
    endA(r);
}

//ok if a e b non stanno eseguendo
void startReset(struct resource* r){
    pthread_mutex_lock(&r->mutex);

    while(r->active_ab){
        r->block_reset = true;
        pthread_cond_wait(&r->priv_reset, &r->mutex);
        r->block_reset = false;
    }

    r->active_reset = true;

    pthread_mutex_unlock(&r->mutex);
}

//risveglio tutti a e b bloccati
void endReset(struct resource* r){
    pthread_mutex_lock(&r->mutex);

    r->active_reset = false;

    if(r->block_ab){
        pthread_cond_broadcast(&r->priv_ab);
    }

    pthread_mutex_unlock(&r->mutex);
}

void pausetta(void)
{
  struct timespec t;
  t.tv_sec = 0;
  t.tv_nsec = (rand()%10+1)*1000000;
  nanosleep(&t,NULL);
}

void *procA(void *arg)
{
  for (;;) {
    fprintf(stderr,"A");
    startA(&gestore);
    putchar(*(char *)arg);
    endA(&gestore);
    fprintf(stderr,"a");
    pausetta();
  }
  return 0;
}

void *procB(void *arg)
{
  for (;;) {
    fprintf(stderr,"B");
    startB(&gestore);
    putchar(*(char *)arg);
    endB(&gestore);
    fprintf(stderr,"b");
    pausetta();
  }
  return 0;
}

void *reset(void *arg)
{
  for (;;) {
    fprintf(stderr,"R");
    startReset(&gestore);
    putchar(*(char *)arg);
    endReset(&gestore);
    fprintf(stderr,"r");
    pausetta();
  }
  return 0;
}

int main()
{
  pthread_attr_t a;
  pthread_t p;
  
  /* inizializzo il mio sistema */
  init_resource(&gestore);

  /* inizializzo i numeri casuali, usati nella funzione pausetta */
  srand(555);

  pthread_attr_init(&a);

  /* non ho voglia di scrivere 10000 volte join! */
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);

  pthread_create(&p, &a, procA, (void *)"+");

  pthread_create(&p, &a, procB, (void *)"-");

  pthread_create(&p, &a, reset, (void *)"*");

  pthread_attr_destroy(&a);

  /* aspetto 10 secondi prima di terminare tutti quanti */
  sleep(10);

  return 0;
}
/*
    cliente:
    1. accesso_sala_attesa, bloccante se active_attesa == MAX_CLIENTI_SALA_ATTESA
    2. accesso_barbiere, bloccante se active_barbiere == MAX_CLIENTI_BARBIERE , chiama uscita_sala_attesa
    for(SHAVING_ITERARIONS)
    3. accesso_pagamento, bloccante se active_pagamento == MAX_CLIENTI_PAGAMENTO, chiama uscita_barbiere
    for(PAYING_ITERATIONS)
    4. uscita_pagamento

    variabili utili:
    - mutex
    - sem_priv sala_attesa, barbiere, pagamento
    - block_attesa, block_barbiere, block_pagamento
    - active_attesa, active_barbiere, active_pagamento

    NB: si suppone che i semafori rispettino una coda FIFO, si suppone che un un cliente con barba
    tagliata non possa lasciare il posto finche non ha la possibilità di pagare
*/
#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#define MAX_CLIENTI_SALA_ATTESA 4
#define MAX_CLIENTI_BARBIERE 3
#define MAX_CLIENTI_PAGAMENTO 1
#define SHAVING_ITERATIONS 10000
#define PAYING_ITERATIONS 1000
#define N_CLIENTI 100

struct gestore_t{
    sem_t mutex;
    sem_t priv_attesa, priv_barbiere, priv_pagamento;

    int block_attesa, block_barbiere, block_pagamento;
    int active_attesa, active_barbiere, active_pagamento;
}barbiere;

void gestore_init(struct gestore_t* b){
    sem_init(&b->mutex,0,1);

    sem_init(&b->priv_attesa,0,0);
    sem_init(&b->priv_barbiere,0,0);
    sem_init(&b->priv_pagamento,0,0);

    b->block_attesa = b->block_barbiere = b->block_pagamento = 0;
    b->active_attesa = b->active_barbiere = b->active_pagamento = 0;
}

void accesso_sala_attesa(struct gestore_t* b, int id_cliente){
    sem_wait(&b->mutex);

    printf("[%d] Vado in sala d'attesa\n", id_cliente);
    if(b->active_attesa == MAX_CLIENTI_SALA_ATTESA){
        printf("[%d] Sala d'attesa piena, attendo\n", id_cliente);
        b->block_attesa++;
        sem_post(&b->mutex);
        sem_wait(&b->priv_attesa);

        printf("[%d] Si è liberato un posto in sala d'attesa, entro\n", id_cliente);
        b->block_attesa--;
    }

    b->active_attesa++;

    sem_post(&b->mutex);
}

void uscita_sala_attesa(struct gestore_t* b, int id_cliente){
    b->active_attesa--;

    printf("[%d] Esco dalla sala d'attesa\n", id_cliente);
    if(b->block_attesa){
        sem_post(&b->priv_attesa);
    }
}

void accesso_barbiere(struct gestore_t* b, int id_cliente){
    sem_wait(&b->mutex);

    printf("[%d] Vado dal barbiere\n", id_cliente);
    if(b->active_barbiere == MAX_CLIENTI_BARBIERE){
        printf("[%d] Tutte le poltrone sono occupate, attendo\n", id_cliente);
        b->block_barbiere++;
        sem_post(&b->mutex);
        
        sem_wait(&b->priv_barbiere);
        printf("[%d] Si è liberata una poltrona, vado dal barbiere\n", id_cliente);
        b->block_barbiere--;
    }

    uscita_sala_attesa(b, id_cliente);
    b->active_barbiere++;

    sem_post(&b->mutex);
}

void uscita_barbiere(struct gestore_t* b, int id_cliente){
    b->active_barbiere--;

    printf("[%d] Esco dal barbiere\n", id_cliente);
    if(b->block_barbiere){
        sem_post(&b->priv_barbiere);
    }
}

void accesso_pagamento(struct gestore_t* b, int id_cliente){
    sem_wait(&b->mutex);

    printf("[%d] Vado dal cassiere\n", id_cliente);
    if(b->active_pagamento == MAX_CLIENTI_PAGAMENTO){
        printf("[%d] Cassiere già occupato, attendo\n", id_cliente);
        b->block_pagamento++;
        sem_post(&b->mutex);

        sem_wait(&b->priv_pagamento);
        printf("[%d] Si è liberato un posto dal cassiere, vado\n", id_cliente);
        b->block_pagamento--;
    }

    uscita_barbiere(b, id_cliente);
    b->active_pagamento++;

    sem_post(&b->mutex);
}

void uscita_pagamento(struct gestore_t* b, int id_cliente){
    sem_wait(&b->mutex);

    printf("[%d] Esco dalla fase di pagamento\n", id_cliente);
    b->active_pagamento--;

    if(b->block_pagamento){
        sem_post(&b->priv_pagamento);
    }
    else{
        sem_post(&b->mutex);
    }
}

void *cliente(void *arg)
{
    int id = (int)(intptr_t)arg;
    accesso_sala_attesa(&barbiere, id);
    accesso_barbiere(&barbiere, id);
    for(int i = 0; i < SHAVING_ITERATIONS; i++){}
    accesso_pagamento(&barbiere, id);
    for(int i = 0; i < PAYING_ITERATIONS; i++){}
    uscita_pagamento(&barbiere, id);
}

int main()
{
  pthread_attr_t a;
  pthread_t p;
  
  /* inizializzo il mio sistema */
  gestore_init(&barbiere);

  /* inizializzo i numeri casuali, usati nella funzione pausetta */
  srand(555);

  pthread_attr_init(&a);

  /* non ho voglia di scrivere 10000 volte join! */
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);

  for(int i = 0; i < N_CLIENTI; i++){
    pthread_create(&p, &a, cliente, (void *)(intptr_t)i);
  }

  pthread_attr_destroy(&a);

  /* aspetto 10 secondi prima di terminare tutti quanti */
  sleep(100);

  return 0;
}
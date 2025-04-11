/*
    NB qualcosa non funziona al reset della partita
    
    arbitro:
    1. attendi_giocatori, bloccante se b_giocatori < 2
    2. attendi_mossa, bloccante se !mossa1 && !mossa2
    3. stampa_risultato, non bloccante
    si mette in attesa finche utente non preme tasto
    
    giocatori:
    1. attendi_via, bloccante sempre, ultimo che arriva sveglia arbitro (se bloccato)
    2. estrai_mossa, estrae e stampa la mossa scelta, non bloccante, se altra mossa già effettuata sveglia arbitro
    si mette in attesa sul via 
    
*/

#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h>

#define CARTA 0
#define SASSO 1
#define FORBICE 2

char *nomi_mosse[3] = {"carta", "sasso", "forbice"};

struct partita_t{
    pthread_mutex_t mutex;
    pthread_cond_t priv_arbitro, priv_giocatori;

    int block_giocatori;
    bool block_arbitro;

    char *mosse[2];
}partita;

void init_partita(struct partita_t* p){
    pthread_mutexattr_t m;
    pthread_condattr_t c;

    pthread_mutexattr_init(&m);
    pthread_mutex_init(&p->mutex, &m);
    pthread_mutexattr_destroy(&m);

    pthread_condattr_init(&c);
    pthread_cond_init(&p->priv_arbitro, &c);
    pthread_cond_init(&p->priv_giocatori, &c);
    pthread_condattr_destroy(&c);

    p->block_giocatori = 0;
    p->block_arbitro = false;

    p->mosse[0] = p->mosse[1] = "";
}

void arbitro_attendi_giocatori(struct partita_t* p){
    pthread_mutex_lock(&p->mutex);

    printf("[arbitro]: arrivato\n");

    while(p->block_giocatori < 2){
        printf("[arbitro]: giocatori non sufficienti, attendo\n");
        p->block_arbitro = true;
        pthread_cond_wait(&p->priv_arbitro, &p->mutex);
        p->block_arbitro = false;
    }

    //da anche il via
    printf("[arbitro]: giocatori arrivati, via!\n");
    pthread_cond_broadcast(&p->priv_giocatori);

    pthread_mutex_unlock(&p->mutex);
}

void arbitro_attendi_mossa(struct partita_t *p){
    pthread_mutex_lock(&p->mutex);

    printf("[arbitro]: controllo mosse giocatori\n");

    while(p->mosse[0] == "" || p->mosse[1] == ""){
        printf("[arbitro]: non sono state ancora completate le mosse, attendo\n");
        p->block_arbitro = true;
        pthread_cond_wait(&p->priv_arbitro, &p->mutex);
        p->block_arbitro = false;
    }

    printf("[arbitro]: tutte le mosse sono state registrate\n");

    pthread_mutex_unlock(&p->mutex);
}

void arbitro_stampa_risultato(struct partita_t *p){
    pthread_mutex_lock(&p->mutex);

    printf("[arbitro]: risultatowow\n");

    pthread_mutex_unlock(&p->mutex);
}

void arbitro_ricomincia_partita(struct partita_t *p){
    pthread_mutex_lock(&p->mutex);

    //svuota array mosse per prossima partita
    p->mosse[0] = p->mosse[1] = "";
    pthread_cond_broadcast(&p->priv_giocatori);

    pthread_mutex_unlock(&p->mutex);
}

void giocatore_attendi_via(struct partita_t* p, int num_giocatore){
    pthread_mutex_lock(&p->mutex);

    printf("[giocatore %d]: arrivato\n", num_giocatore);
    //sveglia arbitro se è il secondo giocatore e se c'è arbitro bloccato (evitabile per le cond var)
    if(p->block_giocatori == 1 && p->block_arbitro){
        printf("[giocatore %d]: sono l'ultimo ad arrivare, sveglio l'arbitro\n", num_giocatore);
        pthread_cond_signal(&p->priv_arbitro);
    }

    p->block_giocatori++;
    pthread_cond_wait(&p->priv_giocatori, &p->mutex);
    p->block_giocatori--;

    pthread_mutex_unlock(&p->mutex);
}

void giocatore_estrai_mossa(struct partita_t* p, int num_giocatore){
    pthread_mutex_lock(&p->mutex);

    p->mosse[num_giocatore] = nomi_mosse[rand()%3];
    printf("[giocatore %d]: sceglie %s\n", num_giocatore, p->mosse[num_giocatore]);

    if(p->mosse[0] != "" && p->mosse[1] != ""){
        printf("[giocatore %d]: sono il secondo a scegliere la mossa, sveglio l'arbitro\n", num_giocatore);
        pthread_cond_signal(&p->priv_arbitro);
    }

    pthread_mutex_unlock(&p->mutex);
}

//forse non necessaria
void giocatore_attendi_nuova_partita(struct partita_t* p, int num_giocatore){
    pthread_mutex_lock(&p->mutex);

    printf("[giocatore %d]: in attesa di una nuova partita\n", num_giocatore);

    p->block_giocatori++;
    pthread_cond_wait(&p->priv_giocatori, &p->mutex);
    p->block_giocatori--;

    pthread_mutex_unlock(&p->mutex);
}

void pausetta(void)
{
  struct timespec t;
  t.tv_sec = 0;
  t.tv_nsec = (rand()%10+1)*1000000;
  nanosleep(&t,NULL);
}

void *giocatore(void* arg)
{
    for(;;){
        giocatore_attendi_via(&partita, (int)(intptr_t)arg);
        giocatore_estrai_mossa(&partita, (int)(intptr_t)arg);
        giocatore_attendi_nuova_partita(&partita, (int)(intptr_t)arg);
    }
    
    return 0;
}

void *arbitro()
{
    for(;;){
        arbitro_attendi_giocatori(&partita);
        arbitro_attendi_mossa(&partita);
        arbitro_stampa_risultato(&partita);
        sleep(5);//simula pressione tasto
        arbitro_ricomincia_partita(&partita);
    }

    return 0;
}

int main()
{
  pthread_attr_t a;
  pthread_t p;
  
  /* inizializzo il mio sistema */
  init_partita(&partita);

  /* inizializzo i numeri casuali, usati nella funzione pausetta */
  srand(555);

  pthread_attr_init(&a);

  /* non ho voglia di scrivere 10000 volte join! */
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);

  pthread_create(&p, &a, giocatore, (void *)(intptr_t)0);
  pthread_create(&p, &a, giocatore, (void *)(intptr_t)1);

  pthread_create(&p, &a, arbitro, NULL);

  pthread_attr_destroy(&a);

  /* aspetto 10 secondi prima di terminare tutti quanti */
  sleep(90);

  return 0;
}
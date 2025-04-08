/*
 * Condition variables demo: this demo simulates a semaphore using
 * mutex and condition variables
 *
 * Copyright (C) 2002 by Paolo Gai
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int counter;
} mysem_t;

void unlock_mutex(void *m)
{
  pthread_mutex_unlock((pthread_mutex_t *)m);
}

void mysem_init(mysem_t *s, int num)
{
  pthread_mutexattr_t m;
  pthread_condattr_t c;

  s->counter = num;

  pthread_mutexattr_init(&m);
  pthread_mutex_init(&s->mutex, &m);
  pthread_mutexattr_destroy(&m);

  pthread_condattr_init(&c);
  pthread_cond_init(&s->cond, &c);
  pthread_condattr_destroy(&c);
}

void mysem_wait(mysem_t *s)
{
  pthread_mutex_lock(&s->mutex);
  while (!s->counter) {
    pthread_cleanup_push(unlock_mutex,(void *)&s->mutex);
    pthread_cond_wait(&s->cond, &s->mutex);
    pthread_cleanup_pop(0);
  }
  
  s->counter--;

  pthread_mutex_unlock(&s->mutex);
}

void mysem_post(mysem_t *s)
{
  pthread_mutex_lock(&s->mutex);

  if (!(s->counter++)) 
    pthread_cond_signal(&s->cond);

  pthread_mutex_unlock(&s->mutex);
}




mysem_t mysem;




void *body(void *arg)
{
  int i,j;
  
  for (j=0; j<40; j++) {
    mysem_wait(&mysem);
    for (i=0; i<1000000; i++);
    fprintf(stderr,(char *)arg);
    mysem_post(&mysem);
  }

  return NULL;
}

int main()
{
  pthread_t t1,t2,t3;
  pthread_attr_t myattr;
  int err;

  mysem_init(&mysem,1);

  pthread_attr_init(&myattr);
  err = pthread_create(&t1, &myattr, body, (void *)".");
  err = pthread_create(&t2, &myattr, body, (void *)"#");
  err = pthread_create(&t3, &myattr, body, (void *)"o");
  pthread_attr_destroy(&myattr);

  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  pthread_join(t3, NULL);

  printf("\n");

  return 0;
}

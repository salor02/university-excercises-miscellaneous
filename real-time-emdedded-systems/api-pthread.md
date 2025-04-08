## SEMAFORI
```c
int sem_init(sem_t *sem, int pshared, unsigned int value);
//pshared is 0 if sem is not shared between processes

int sem_destroy(sem_t *sem)

int sem_wait(sem_t *sem);
int sem_post(sem_t *sem);

int sem_getvalue(sem_t *sem,int *val);
```

## MUTEX
```c
int pthread_mutexattr_init(pthread_mutexattr_t *attr);

int pthread_mutex_init (pthread_mutex_t *mutex, const pthread_mutexattr_t *attr);

int pthread_mutex_lock(pthread_mutex_t *m);
int pthread_mutex_unlock(pthread_mutex_t *m);
```

## CONDITION VARIABLES
```c
int pthread_condattr_init (pthread_condattr_t *attr);

int pthread_cond_init (pthread_cond_t *cond, const pthread_condattr_t *attr);

int pthread_cond_wait (pthread_cond_t *cond, pthread_mutex_t *mutex);

int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_broadcast(pthread_cond_t *cond);
```
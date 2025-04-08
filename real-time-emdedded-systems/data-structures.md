# STRUTTURE DATI UTILI

## CIRCULAR ARRAY
```c
struct CircularArray_t {
    int array[10];
    int head, tail, num;
    } queue;
```
```c
void init_CA(struct CircularArray_t *a){ 
    a->head=0; a->tail=0; a->num=0; 
}
```
```c
int insert_CA(struct CircularArray_t *a, int elem){ 
    if (a->num == 10) return 0;
    a->array[a->head] = elem;
    a->head = (a->head + 1) % 10;
    a->num++;
    return 1;
}
```
```c
int extract_CA(struct CircularArray_t *a, int *elem){ 
    if (a->num == 0) return 0;
    *elem = a->array[a->tail];
    a->tail = (a->tail + 1) % 10;
    a->num--;
    return 1;
}
```
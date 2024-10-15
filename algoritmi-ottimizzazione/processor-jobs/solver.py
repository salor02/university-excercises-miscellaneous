"""
-   fare alcuni algoritmi costruttivi per risolvere il problema dello scheduling dei jobs
-   fare una routine di controllo che controlli che la soluzione sia giusta (tipo che tutti i jobs siano stati
    schedulati)
-   preparare le istanze suddividendole per 2 tipi (ad esempio 1 tipo per jobs generati casualmente e 1 tipo per molti jobs
    piccoli e pochi grandi), per diverse tuple <Njobs, Nproc> si generano 20 istante per ognuno dei 2 tipi
-   provare gli algoritmi su tutte le istanze per avere delle soluzioni e controllare tutte le soluzioni con la routine
    di controllo
-   provando gli algoritmi prendere anche delle metriche tipo il tempo impiegato e fare le statistiche
-   una volta fatto tutto implementare local search per cercare il minimo"""

import random

PROCESSORS_NUM = 4
JOBS_NUM = 10

def jobs_gen(seed):
    instance = random.Random(seed)
    jobs_time = [round(instance.random()*10,2) for job in range(JOBS_NUM)]
    return jobs_time

def sequential_scheduling(jobs_time):
    processors_jobs = [list() for _ in range(PROCESSORS_NUM)]

    for idx, _ in enumerate(jobs_time):
        assigned_processor = idx % PROCESSORS_NUM
        processors_jobs[assigned_processor].append(idx)

    return processors_jobs

def time_priority_scheduling(jobs_time, k=1):
    processors_jobs = [list() for _ in range(PROCESSORS_NUM)]
    processors_time = [0 for x in range(PROCESSORS_NUM)]
    
    #da cambiare perche cosi fa schifo
    jobs = dict()
    for idx, job in enumerate(jobs_time):
        jobs[idx] = job

    for idx,time in enumerate(jobs_time):
        assigned_processor = processors_time.index(min(processors_time))

        selected_idx = random.randint(idx, idx+k-1)
        selected_job = jobs_time[selected_idx]

        processors_jobs[assigned_processor].append(selected_idx)
        processors_time[assigned_processor] += time

    return processors_jobs

def get_total_computation_time(processors_jobs):
    total_time = 0
    processor_time = [0 for x in range(PROCESSORS_NUM)]

    for idx, processor in enumerate(processors_jobs):
        for job in processor:
            processor_time[idx] += jobs_time[job]
        print(f'{idx} : {processor_time[idx]}')

    return round(max(processor_time),2)

def sort_jobs(jobs_time, scheduling_type, reverse=False):
    jobs_time.sort(reverse=reverse)
    if scheduling_type == 'time_priority':
        processors_jobs = time_priority_scheduling(jobs_time,2)
    elif scheduling_type == 'sequential':
        processors_jobs = sequential_scheduling(jobs_time)
    return get_total_computation_time(processors_jobs)

def randomize_jobs(jobs_time, scheduling_type):
    random.shuffle(jobs_time)
    if scheduling_type == 'time_priority':
        processors_jobs = time_priority_scheduling(jobs_time)
    elif scheduling_type == 'sequential':
        processors_jobs = sequential_scheduling(jobs_time)
    
    return get_total_computation_time(processors_jobs)

def get_random_k_min_jobs(jobs_time, k, reverse=False):
    jobs_time.sort(reverse=reverse)

if __name__ == '__main__':
    print('Generating instance number 1')
    jobs_time = jobs_gen(1)
    print(f'Instance generated')
    print('-'*50)

    print('[EXECUTING] Ascending sorted jobs and sequential scheduling...')
    cmax = sort_jobs(jobs_time, 'sequential', reverse=False)
    print(f'Total computation time: {cmax}')
    print('-'*50)

    print('[EXECUTING] Ascending sorted jobs and time-priority scheduling...')
    cmax = sort_jobs(jobs_time, 'time_priority', reverse=False)
    print(f'Total computation time: {cmax}')
    print('-'*50)

    print('[EXECUTING] Descending sorted jobs and sequential scheduling...')
    cmax = sort_jobs(jobs_time, 'sequential', reverse=True)
    print(f'Total computation time: {cmax}')
    print('-'*50)

    print('[EXECUTING] Descending sorted jobs and time-priority scheduling...')
    cmax = sort_jobs(jobs_time, 'time_priority', reverse=True)
    print(f'Total computation time: {cmax}')
    print('-'*50)

    print('[EXECUTING] Shuffled jobs and sequential scheduling...')
    cmax = randomize_jobs(jobs_time, 'sequential')
    print(f'Total computation time: {cmax}')
    print('-'*50)

    print('[EXECUTING] Shuffled jobs and time-priority scheduling...')
    cmax = randomize_jobs(jobs_time, 'time_priority')
    print(f'Total computation time: {cmax}')
    print('-'*50)



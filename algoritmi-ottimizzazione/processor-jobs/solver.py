import random

PROCESSORS_NUM = 4
JOBS_NUM = 10000

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

def time_priority_scheduling(jobs_time):
    processors_jobs = [list() for _ in range(PROCESSORS_NUM)]
    processors_time = [0 for x in range(PROCESSORS_NUM)]
    
    for idx,time in enumerate(jobs_time):
        assigned_processor = processors_time.index(min(processors_time))
        processors_jobs[assigned_processor].append(idx)
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
        processors_jobs = time_priority_scheduling(jobs_time)
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

def Krandom_scheduling(jobs_time, k, reverse=False):
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



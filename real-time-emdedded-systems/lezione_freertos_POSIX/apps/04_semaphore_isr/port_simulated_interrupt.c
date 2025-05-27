#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include "FreeRTOS.h"
#include "task.h"
#include "portmacro.h"

/* Simulated maximum number of interrupts */
#define portMAX_INTERRUPTS    (32)

/* Mutex and condition variable for event signaling */
static pthread_mutex_t pvInterruptEventMutex;
static pthread_cond_t  pvInterruptEventCond;

/* Bitfield tracking pending interrupts */
static volatile uint32_t ulPendingInterrupts = 0;

/* Simulated critical nesting counter */
static volatile uint32_t ulCriticalNesting = 0;

/* Flag indicating scheduler is running */
static volatile BaseType_t xPortRunning = pdFALSE;

/* Forward declaration of interrupt handler table */
typedef void (*InterruptHandler_t)(void);
static InterruptHandler_t xInterruptHandlers[portMAX_INTERRUPTS] = {0};

/*-----------------------------------------------------------*/

/* Initialize simulated interrupt controller */
void vPortSetupSimulatedInterrupts(void)
{
    pthread_mutex_init(&pvInterruptEventMutex, NULL);
    pthread_cond_init(&pvInterruptEventCond, NULL);
}

/* Install interrupt handler */
void vPortSetInterruptHandler(uint32_t ulInterruptNumber, InterruptHandler_t handler)
{
    if (ulInterruptNumber < portMAX_INTERRUPTS)
    {
        xInterruptHandlers[ulInterruptNumber] = handler;
    }
}

/* Generate a simulated interrupt */
void vPortGenerateSimulatedInterrupt(uint32_t ulInterruptNumber)
{
    configASSERT(xPortRunning);

    if ((ulInterruptNumber < portMAX_INTERRUPTS))
    {
        /* Lock the interrupt event mutex */
        pthread_mutex_lock(&pvInterruptEventMutex);

        /* Set the interrupt pending bit */
        ulPendingInterrupts |= (1 << ulInterruptNumber);

        /* If not in a critical section, signal the event */
        if (ulCriticalNesting == 0)
        {
            pthread_cond_signal(&pvInterruptEventCond);
        }

        /* Unlock the mutex */
        pthread_mutex_unlock(&pvInterruptEventMutex);
    }
}

/* Simulated interrupt processing thread */
static void* vInterruptProcessingThread(void* pvParameters)
{
    (void) pvParameters;

    while (1)
    {
        /* Wait for an interrupt event */
        pthread_mutex_lock(&pvInterruptEventMutex);
        while (ulPendingInterrupts == 0)
        {
            pthread_cond_wait(&pvInterruptEventCond, &pvInterruptEventMutex);
        }

        /* Copy pending interrupts and clear the global state */
        uint32_t ulLocalPendingInterrupts = ulPendingInterrupts;
        ulPendingInterrupts = 0;
        pthread_mutex_unlock(&pvInterruptEventMutex);

        /* Process copied pending interrupts outside the lock */
        for (uint32_t i = 0; i < portMAX_INTERRUPTS; i++)
        {
            if (ulLocalPendingInterrupts & (1 << i))
            {
                if (xInterruptHandlers[i] != NULL)
                {
                    /* Create a thread to run the handler safely */
                    pthread_t xHandlerThread;
                    pthread_create(&xHandlerThread, NULL, (void* (*)(void*))xInterruptHandlers[i], NULL);
                    pthread_detach(xHandlerThread);
                }
            }
        }
    }

    /* Unreachable */
    return NULL;
}
    

/* Start the simulated interrupt processing thread */
void vPortStartSimulatedInterruptProcessing(void)
{
    pthread_t interruptThread;
    xPortRunning = pdTRUE;

    pthread_create(&interruptThread, NULL, vInterruptProcessingThread, NULL);
    pthread_detach(interruptThread);
}

/* Standard includes. */
#include <stdio.h>
#include <stdlib.h>

/* FreeRTOS.org includes. */
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

/* Demo includes. */
#include "utils.h"

/* The task to be created.  Two instances of this task are created. */
static void prvPrintTask( void *pvParameters );

/* The function that uses a mutex to control access to standard out. */
static void prvNewPrintString( const char *pcString );

/*-----------------------------------------------------------*/

/* Declare a variable of type SemaphoreHandle_t.  This is used to reference the
mutex type semaphore that is used to ensure mutual exclusive access to stdout. */
SemaphoreHandle_t xMutex;

/* The tasks block for a pseudo random time between 0 and xMaxBlockTime ticks. */
const TickType_t xMaxBlockTimeTicks = 0x20;

int main( void )
{
    /* Before a semaphore is used it must be explicitly created.  In this example
	a mutex type semaphore is created. */
    xMutex = xSemaphoreCreateMutex();

	/* Check the semaphore was created successfully. */
	if( xMutex != NULL )
	{
		/* Create two instances of the tasks that attempt to write stdout.  The
		string they attempt to write is passed into the task as the task's
		parameter.  The tasks are created at different priorities so some
		pre-emption will occur. */
		xTaskCreate( prvPrintTask, "Print1", 1000, "Task 1 ******************************************\r\n", 1, NULL );
		xTaskCreate( prvPrintTask, "Print2", 1000, "Task 2 ------------------------------------------\r\n", 2, NULL );

		/* Start the scheduler so the created tasks start executing. */
		vTaskStartScheduler();
	}

	/* The following line should never be reached because vTaskStartScheduler()
	will only return if there was not enough FreeRTOS heap memory available to
	create the Idle and (if configured) Timer tasks.  Heap management, and
	techniques for trapping heap exhaustion, are described in the book text. */
	for( ;; );
	return 0;
}
/*-----------------------------------------------------------*/

static void prvNewPrintString( const char *pcString )
{
	/* The semaphore is created before the scheduler is started so already
	exists by the time this task executes.

	Attempt to take the semaphore, blocking indefinitely if the mutex is not
	available immediately.  The call to xSemaphoreTake() will only return when
	the semaphore has been successfully obtained so there is no need to check the
	return value.  If any other delay period was used then the code must check
	that xSemaphoreTake() returns pdTRUE before accessing the resource (in this
	case standard out. */
	xSemaphoreTake( xMutex, portMAX_DELAY );
	{
		/* The following line will only execute once the semaphore has been
		successfully obtained - so standard out can be accessed freely. */
		printf( "%s", pcString );
		fflush( stdout );
	}
	xSemaphoreGive( xMutex );
}
/*-----------------------------------------------------------*/

static void prvPrintTask( void *pvParameters )
{
char *pcStringToPrint;

    /* Two instances of this task are created.  The string printed by the task
	is passed into the task using the task's parameter.  The parameter is cast
	to the required type. */
	pcStringToPrint = ( char * ) pvParameters;

	for( ;; )
	{
		/* Print out the string using the newly defined function. */
		prvNewPrintString( pcStringToPrint );

		/* Wait a pseudo random time.  Note that rand() is not necessarily
		re-entrant, but in this case it does not really matter as the code does
		not care what value is returned.  In a more secure application a version
		of rand() that is known to be re-entrant should be used - or calls to
		rand() should be protected using a critical section. */
		vTaskDelay( rand() & xMaxBlockTimeTicks );
	}
}




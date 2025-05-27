#include <string.h>
#include <stdio.h>
#include <unistd.h>

#include "FreeRTOS.h"
#include "FreeRTOSConfig.h"
#include "task.h"

/* Used as a loop counter to create a very crude delay. */
#define mainDELAY_LOOP_COUNT		( 0xffffff )

static const char * pcTextForTask1 = "Task 1 is running\n";
static const char * pcTextForTask2 = "Task 2 is running\n";

void vTaskFunction( void *pvParameters )
{
char *pcTaskName;
volatile uint32_t ul;

	/* The string to print out is passed in via the parameter.  Cast this to a
	character pointer. */
	pcTaskName = ( char * ) pvParameters;

	/* As per most tasks, this task is implemented in an infinite loop. */
	for( ;; )
	{
		/* Print out the name of this task. */
		vPrintString( pcTaskName );

		/* Delay for a period. */
		// for( ul = 0; ul < mainDELAY_LOOP_COUNT; ul++ ){
		// 	/* This loop is just a very crude delay implementation.  There is
		// 	nothing to do in here.  This task will never willingly enter the blocked state */
		// }
        
        /* Delay for a period.  This time a call to vTaskDelay() is used which
		places the task into the Blocked state until the delay period has
		expired.  The parameter takes a time specified in 'ticks', and the
		pdMS_TO_TICKS() macro is used (where the xDelay250ms constant is
		declared) to convert 250 milliseconds into an equivalent time in
		ticks. */

        vTaskDelay( pdMS_TO_TICKS( 250UL ) );
	}
}


int main( void ){
    /* Create the first task with a priority of 1. */

    xTaskCreate( vTaskFunction, /* Task Function */
                "Task 1", /* Task Name */
                1000, /* Task Stack Depth */
                ( void * ) pcTextForTask1, /* Task Parameter */
                1, /* Task Priority */
                NULL );

    /* Create the second task at a higher priority of 2. */
    xTaskCreate( vTaskFunction, /* Task Function */
                "Task 2", /* Task Name */
                1000, /* Task Stack Depth */
                ( void * ) pcTextForTask2, /* Task Parameter */
                2, /* Task Priority */
                NULL );

    /* Start the scheduler so the tasks start executing. */
    vTaskStartScheduler();
    /* Will not reach here. */
    return 0;
}
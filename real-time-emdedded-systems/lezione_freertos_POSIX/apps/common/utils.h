 #ifndef DEMO_H
 #define DEMO_H

 #ifdef __cplusplus
     extern "C" {
 #endif

/*-----------------------------------------------------------
* Example console I/O wrappers.
*----------------------------------------------------------*/
void vPrintString(const char * fmt, ...);

void vApplicationMallocFailedHook( void );
void vApplicationIdleHook( void );
void vApplicationStackOverflowHook( TaskHandle_t pxTask,
                                    char * pcTaskName );
void vApplicationTickHook( void );
void vApplicationGetIdleTaskMemory( StaticTask_t ** ppxIdleTaskTCBBuffer,
                                    StackType_t ** ppxIdleTaskStackBuffer,
                                    configSTACK_DEPTH_TYPE * pulIdleTaskStackSize );
void vApplicationGetTimerTaskMemory( StaticTask_t ** ppxTimerTaskTCBBuffer,
                                     StackType_t ** ppxTimerTaskStackBuffer,
                                     configSTACK_DEPTH_TYPE * pulTimerTaskStackSize );


/*
 * Signal handler for Ctrl_C to cause the program to exit, and generate the
 * profiling info.
 */
static void handle_sigint( int signal );

 #ifdef __cplusplus
     }
 #endif

#endif /* DEMO_H */

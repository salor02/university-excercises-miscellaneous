#ifndef PORT_SIMULATED_INTERRUPTS_H
#define PORT_SIMULATED_INTERRUPTS_H

#include <stdint.h>
#include "FreeRTOS.h"

/* Max number of supported simulated interrupts */
#define portMAX_INTERRUPTS    (32)

/* Type for simulated interrupt handlers */
typedef void (*InterruptHandler_t)(void);

/*-----------------------------------------------------------
 * Simulated Interrupt API for POSIX FreeRTOS port
 *----------------------------------------------------------*/

/**
 * @brief Initialize simulated interrupt controller.
 * 
 * Must be called before installing handlers or generating interrupts.
 */
void vPortSetupSimulatedInterrupts(void);

/**
 * @brief Install an interrupt handler for a given interrupt number.
 * 
 * @param ulInterruptNumber Interrupt number (0 to portMAX_INTERRUPTS-1).
 * @param handler Function pointer to the ISR handler.
 */
void vPortSetInterruptHandler(uint32_t ulInterruptNumber, InterruptHandler_t handler);

/**
 * @brief Generate a simulated interrupt.
 * 
 * Sets the pending bit for the given interrupt number and signals
 * the interrupt controller thread to process it.
 * 
 * @param ulInterruptNumber Interrupt number (0 to portMAX_INTERRUPTS-1).
 */
void vPortGenerateSimulatedInterrupt(uint32_t ulInterruptNumber);

/**
 * @brief Start the simulated interrupt processing thread.
 * 
 * Must be called after setting up simulated interrupts and handlers.
 */
void vPortStartSimulatedInterruptProcessing(void);

#endif /* PORT_SIMULATED_INTERRUPTS_H */

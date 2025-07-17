// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.1 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// params
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of in1
//        bit 31~0 - in1[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of in2
//        bit 31~0 - in2[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of out_r
//        bit 31~0 - out_r[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of dim
//        bit 31~0 - dim[31:0] (Read/Write)
// 0x2c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XMMULT_PARAMS_ADDR_AP_CTRL    0x00
#define XMMULT_PARAMS_ADDR_GIE        0x04
#define XMMULT_PARAMS_ADDR_IER        0x08
#define XMMULT_PARAMS_ADDR_ISR        0x0c
#define XMMULT_PARAMS_ADDR_IN1_DATA   0x10
#define XMMULT_PARAMS_BITS_IN1_DATA   32
#define XMMULT_PARAMS_ADDR_IN2_DATA   0x18
#define XMMULT_PARAMS_BITS_IN2_DATA   32
#define XMMULT_PARAMS_ADDR_OUT_R_DATA 0x20
#define XMMULT_PARAMS_BITS_OUT_R_DATA 32
#define XMMULT_PARAMS_ADDR_DIM_DATA   0x28
#define XMMULT_PARAMS_BITS_DIM_DATA   32


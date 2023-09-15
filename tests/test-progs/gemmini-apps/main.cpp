#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <chrono>

#include "baseline.h"

#define GET_TICKS std::chrono::high_resolution_clock::now()
#define GET_ELAPS(A, B) std::chrono::duration_cast<std::chrono::nanoseconds>(B - A).count() / 1000

/* ==================== BEGIN BENCHMARK CONFIGURATION ===================== */

#define CONV2D_MS       256
#define CONV2D_GEMM_MS  85 //64
#define CONV3D_MS       40 //32
#define CONV3D_GEMM_MS  13 //8
#define MAXPOOL_MS      256
#define MAXPOOL_GEMM_MS MAXPOOL_MS
#define RELU_MS         256
#define MM_MS           128
#define MM_GEMM_MS      MM_MS

#define K_SIZE          3
#define P_SIZE          2

/* ===================== END BENCHMARK CONFIGURATION ====================== */

/* ==================== BEGIN STATIC MEMORY MANAGEMENT ==================== */

#define STATIC_MEM_BASE_ADDR 0x40001000
#define STATIC_MEM_TOTL_SIZE 0x3ffff000
#define ALIGN_SIZE 0x40

char *static_ptr;

void *staticAlloc(size_t bytes)
{
    if (static_ptr + bytes < (char *) STATIC_MEM_BASE_ADDR + STATIC_MEM_TOTL_SIZE)
    {
        char *aux_ptr = static_ptr;
        
        size_t offset = bytes / ALIGN_SIZE * ALIGN_SIZE;
        if (bytes > offset)
            offset += ALIGN_SIZE;
        
        static_ptr += offset;
        
        return aux_ptr;
    }

    return NULL;
}

void staticFree(void *ptr)
{
    return;
}

/* ===================== END STATIC MEMORY MANAGEMENT ===================== */

/* ==================== BEGIN GEMMINI DEVICE MANAGEMENT =================== */

#define GEMMINI_DEV_A_ADDR_CTRL 0x40000000

typedef enum GemminiDevAOP_
{
    op_conv2d,
    op_conv2d_gemm,
    op_conv3d,
    op_conv3d_gemm,
    op_maxpool,
    op_maxpool_gemm,
    op_relu,
    op_mm,
    op_mm_gemm
} GemminiDevAOP;

void callGemminiDevA(
    float *m,
    float *k,
    float *o,
    unsigned int m_size,
    unsigned int k_size,
    GemminiDevAOP opcode)
{
    uint64_t *gemmini_dev_a_ctrl = (uint64_t *) GEMMINI_DEV_A_ADDR_CTRL;

    gemmini_dev_a_ctrl[0] = (uint64_t) m;
    gemmini_dev_a_ctrl[1] = (uint64_t) k;
    gemmini_dev_a_ctrl[2] = (uint64_t) o;
    gemmini_dev_a_ctrl[3] = (uint64_t) m_size;
    gemmini_dev_a_ctrl[4] = (uint64_t) k_size;
    gemmini_dev_a_ctrl[5] = (uint64_t) opcode;
    
    gemmini_dev_a_ctrl[6] = (uint64_t) 0x1;
    
    while(!gemmini_dev_a_ctrl[7]);
}

/* ===================== END GEMMINI DEVICE MANAGEMENT ==================== */

bool conv2d(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *k  = (float *) staticAlloc(k_size * k_size * sizeof(float)), // kernel
          *os = (float *) staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(k); assert(os); assert(oh);

    // Initialize data structures
    initRandArray(m, m_size * m_size);
    initRandArray(k, k_size * k_size);

    mem_time = 0;

    auto start_sw = GET_TICKS;
    baseConv_2D(m, k, os, m_size, k_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(m, k, oh, m_size, k_size, op_conv2d);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size);

    staticFree(m); staticFree(k); staticFree(os); staticFree(oh);

    return status;
}

bool conv2dGemm(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *a  = (float *) staticAlloc(m_size * m_size * 
                                      k_size * k_size * sizeof(float)), // redundant matrix
          *k  = (float *) staticAlloc(k_size * k_size * sizeof(float)), // kernel
          *os = (float *) staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(a); assert(k); assert(os); assert(oh);

    // Initialize data structures
    initRandArray(m, m_size * m_size);
    initRandArray(k, k_size * k_size);

    auto start_mem = GET_TICKS;
    compMatrixA_2D(a, m, m_size, k_size);
    mem_time = GET_ELAPS(start_mem, GET_TICKS);

    auto start_sw = GET_TICKS;
    baseConvGemm(a, k, os, m_size * m_size, k_size * k_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(a, k, oh, m_size, k_size, op_conv2d_gemm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size);

    staticFree(m); staticFree(a); staticFree(k); staticFree(os); staticFree(oh);

    return status;
}

bool conv3d(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) staticAlloc(m_size * m_size * m_size * sizeof(float)), // input matrix
          *k  = (float *) staticAlloc(k_size * k_size * k_size * sizeof(float)), // kernel
          *os = (float *) staticAlloc(m_size * m_size * m_size * sizeof(float)), // output software
          *oh = (float *) staticAlloc(m_size * m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(k); assert(os); assert(oh);

    // Initialize data structures
    initRandArray(m, m_size * m_size * m_size);
    initRandArray(k, k_size * k_size * k_size);

    mem_time = 0;

    auto start_sw = GET_TICKS;
    baseConv_3D(m, k, os, m_size, k_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(m, k, oh, m_size, k_size, op_conv3d);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size);

    staticFree(m); staticFree(k); staticFree(os); staticFree(oh);

    return status;
}

bool conv3dGemm(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) staticAlloc(m_size * m_size * m_size * sizeof(float)), // input matrix
          *a  = (float *) staticAlloc(m_size * m_size * m_size * 
                                      k_size * k_size * k_size * sizeof(float)), // redundant matrix
          *k  = (float *) staticAlloc(k_size * k_size * k_size * sizeof(float)), // kernel
          *os = (float *) staticAlloc(m_size * m_size * m_size * sizeof(float)), // output software
          *oh = (float *) staticAlloc(m_size * m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(a); assert(k); assert(os); assert(oh);

    // Initialize data structures
    initRandArray(m, m_size * m_size * m_size);
    initRandArray(k, k_size * k_size * k_size);

    auto start_mem = GET_TICKS;
    compMatrixA_3D(a, m, m_size, k_size);
    mem_time = GET_ELAPS(start_mem, GET_TICKS);

    auto start_sw = GET_TICKS;
    baseConvGemm(a, k, os, m_size * m_size * m_size, k_size * k_size * k_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(a, k, oh, m_size, k_size, op_conv3d_gemm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size);

    staticFree(m); staticFree(a); staticFree(k); staticFree(os); staticFree(oh);

    return status;
}

bool maxPool(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *os = (float *) staticAlloc(m_size * m_size /
                                      k_size / k_size * sizeof(float)), // output matrix
          *oh = (float *) staticAlloc(m_size * m_size /
                                      k_size / k_size * sizeof(float)); // output matrix

    assert(m); assert(os); assert(oh);

    initRandArray(m, m_size * m_size);

    mem_time = 0;

    auto start_sw = GET_TICKS;
    baseMaxPool(m, os, m_size, k_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(m, NULL, oh, m_size, k_size, op_maxpool);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size / k_size / k_size);

    staticFree(m); staticFree(os); staticFree(oh);

    return status;
}

bool maxPoolGemm(
    unsigned int m_size,
    unsigned int k_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *a  = (float *) staticAlloc(m_size * m_size * sizeof(float)), // redundant matrix
          *os = (float *) staticAlloc(m_size * m_size /
                                      k_size / k_size * sizeof(float)), // output matrix
          *oh = (float *) staticAlloc(m_size * m_size /
                                      k_size / k_size * sizeof(float)); // output matrix

    assert(m); assert(a); assert(os); assert(oh);

    initRandArray(m, m_size * m_size);

    auto start_mem = GET_TICKS;
    compMatMaxA(a, m, m_size, k_size);
    mem_time = GET_ELAPS(start_mem, GET_TICKS);

    auto start_sw = GET_TICKS;
    baseMaxPoolGemm(a, os, m_size, k_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(a, NULL, oh, m_size, k_size, op_maxpool_gemm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size / k_size / k_size);

    staticFree(m); staticFree(a); staticFree(os); staticFree(oh);

    return status;
}

bool relu(
    unsigned int m_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m  = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix
          *os = (float *) staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m); assert(os); assert(oh);

    initRandArray(m, m_size * m_size);

    mem_time = 0;

    auto start_sw = GET_TICKS;
    baseRelu(m, os, m_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(m, NULL, oh, m_size, 0, op_relu);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size);

    staticFree(m); staticFree(os); staticFree(oh);

    return status;
}

bool mm(
    unsigned int m_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m1 = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix 1
          *m2 = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix 2
          *os = (float *) staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m1); assert(m2); assert(os); assert(oh);

    initRandArray(m1, m_size * m_size);
    initRandArray(m2, m_size * m_size);

    mem_time = 0;

    auto start_sw = GET_TICKS;
    baseMM(m1, m2, os, m_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(m1, m2, oh, m_size, 0, op_mm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size);

    staticFree(m1); staticFree(m2); staticFree(os); staticFree(oh);

    return status;
}

bool mmGemm(
    unsigned int m_size,
    unsigned long &sw_time,
    unsigned long &hw_time,
    unsigned long &mem_time)
{
    float *m1 = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix 1
          *m2 = (float *) staticAlloc(m_size * m_size * sizeof(float)), // input matrix 2
          *a  = (float *) staticAlloc(m_size * m_size * sizeof(float)), // matrix 2 transposed
          *os = (float *) staticAlloc(m_size * m_size * sizeof(float)), // output software
          *oh = (float *) staticAlloc(m_size * m_size * sizeof(float)); // output hardware

    assert(m1); assert(m2); assert(a); assert(os); assert(oh);

    initRandArray(m1, m_size * m_size);
    initRandArray(m2, m_size * m_size);

    auto start_mem = GET_TICKS;
    transpose(m2, a, m_size);
    mem_time = GET_ELAPS(start_mem, GET_TICKS);

    auto start_sw = GET_TICKS;
    baseMMGemm(m1, a, os, m_size);
    sw_time = GET_ELAPS(start_sw, GET_TICKS);

    auto hw_start = GET_TICKS;
    callGemminiDevA(m1, a, oh, m_size, 0, op_mm_gemm);
    hw_time = GET_ELAPS(hw_start, GET_TICKS);

    bool status = compareArrays(os, oh, m_size * m_size);

    staticFree(m1); staticFree(m2); staticFree(a); staticFree(os); staticFree(oh);

    return status;
}

int
main(int argc, char const *argv[])
{
    // Initialize static memory
    static_ptr = (char *) STATIC_MEM_BASE_ADDR;

    unsigned long sw_time, hw_time, mem_time;

    printf("status bench        | sw_time | hw_time | mem_time | speedup\n");
    printf("------------------------------------------------------------\n");

    printf("[%s] ", conv2d(CONV2D_MS, K_SIZE, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("conv2d       | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    printf("[%s] ", conv2dGemm(CONV2D_GEMM_MS, K_SIZE, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("conv2d_gemm  | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    printf("[%s] ", conv3d(CONV3D_MS, K_SIZE, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("conv3d       | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    printf("[%s] ", conv3dGemm(CONV3D_GEMM_MS, K_SIZE, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("conv3d_gemm  | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    printf("[%s] ", maxPool(MAXPOOL_MS, P_SIZE, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("maxpool      | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    printf("[%s] ", maxPoolGemm(MAXPOOL_GEMM_MS, P_SIZE, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("maxpool_gemm | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    printf("[%s] ", relu(RELU_MS, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("relu         | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    printf("[%s] ", mm(MM_MS, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("mm           | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    printf("[%s] ", mmGemm(MM_GEMM_MS, sw_time, hw_time, mem_time) ? "PASS" : "FAIL");
    printf("mm_gemm      | %7lu | %7lu |  %7lu |    %3.2f\n",
        sw_time, hw_time, mem_time, ((float) sw_time + mem_time) / (hw_time + mem_time));

    return 0;
}

#ifndef __BASELINE_H__
#define __BASELINE_H__

#include <cstdlib>
#include <cassert>
#include <cstdio>

void baseConv_2D(float *m, float *k, float *o, unsigned int m_size, unsigned int k_size);
void baseConv_3D(float *m, float *k, float *o, unsigned int m_size, unsigned int k_size);
void compMatrixA_2D(float *a, float *m, unsigned int m_size, unsigned int k_size);
void compMatrixA_3D(float *a, float *m, unsigned int m_size, unsigned int k_size);
void baseConvGemm(float *a, float *k, float *o, unsigned int o_size, unsigned int k_size);

void baseMaxPool(float *m, float *o, unsigned int m_size, unsigned int k_size);
void compMatMaxA(float *a, float *m, unsigned int m_size, unsigned int k_size);
void baseMaxPoolGemm(float *a, float *o, unsigned int m_size, unsigned int k_size);

void baseRelu(float *m, float *o, unsigned int m_size);

void baseMM(float *m, float *b, float *c, unsigned int m_size);
void transpose(float *m, float *a, unsigned int m_size);
void baseMMGemm(float *a, float *b, float *c, unsigned int m_size);

void initRandArray(float *a, unsigned int size);
bool compareArrays(float *a1, float *a2, unsigned int size);
void printMatrix_2D(float *m, unsigned int m_size);
void printMatrix_3D(float *m, unsigned int m_size);

#endif // __BASELINE_H__

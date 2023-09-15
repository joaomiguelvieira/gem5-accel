#include "baseline.h"

/* Convolution */

void baseConv_2D(float *m, float *k, float *o, unsigned int m_size, unsigned int k_size)
{
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size; j++)
        {
            o[i * m_size + j] = 0;

            for (int w = 0; w < k_size; w++)
                for (int x = 0; x < k_size; x++)
                {
                    int m_i = i + w - k_size / 2,
                        m_j = j + x - k_size / 2,
                        m_1d_idx;

                    if (m_i < 0 || m_i >= m_size ||
                        m_j < 0 || m_j >= m_size)
                        m_1d_idx = -1;
                    else
                        m_1d_idx = m_i * m_size + m_j;

                    o[i * m_size + j] += (m_1d_idx == -1) ? 0 : m[m_1d_idx] * k[w * k_size + x];
                }
        }
}

void baseConv_3D(float *m, float *k, float *o, unsigned int m_size, unsigned int k_size)
{
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size; j++)
            for (int w = 0; w < m_size; w++)
            {
                o[i * m_size * m_size + j * m_size + w] = 0;

                for (int x = 0; x < k_size; x++)
                    for (int y = 0; y < k_size; y++)
                        for (int z = 0; z < k_size; z++)
                        {
                            int m_i = i + x - k_size / 2,
                                m_j = j + y - k_size / 2,
                                m_w = w + z - k_size / 2,
                                m_1d_idx;

                            if (m_i < 0 || m_i >= m_size ||
                                m_j < 0 || m_j >= m_size ||
                                m_w < 0 || m_w >= m_size)
                                m_1d_idx = -1;
                            else
                                m_1d_idx = m_i * m_size * m_size + m_j * m_size + m_w;

                            o[i * m_size * m_size + j * m_size + w] +=
                                (m_1d_idx == -1) ? 0 : m[m_1d_idx] * k[x * k_size * k_size + y * k_size + z];
                        }
            }
}

void compMatrixA_2D(float *a, float *m, unsigned int m_size, unsigned int k_size)
{
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size; j++)
            for (int w = 0; w < k_size; w++)
                for (int x = 0; x < k_size; x++)
                {
                    // Matrix A index
                    unsigned int a_1d_idx =
                        (i * m_size + j) * k_size * k_size + // row and column
                        w * k_size + x;                      // kernel element

                    // Matrix M index
                    int m_i = i + w - k_size / 2,
                        m_j = j + x - k_size / 2,
                        m_1d_idx;

                    if (m_i < 0 || m_i >= m_size ||
                        m_j < 0 || m_j >= m_size)
                        m_1d_idx = -1;
                    else
                        m_1d_idx = m_i * m_size + m_j;

                    a[a_1d_idx] = (m_1d_idx == -1) ? 0 : m[m_1d_idx];
                }
}

void compMatrixA_3D(float *a, float *m, unsigned int m_size, unsigned int k_size)
{
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size; j++)
            for (int w = 0; w < m_size; w++)
                for (int x = 0; x < k_size; x++)
                    for (int y = 0; y < k_size; y++)
                        for (int z = 0; z < k_size; z++)
                        {
                            // Matrix A index
                            unsigned int a_1d_idx =
                                (i * m_size * m_size + j * m_size + w) * k_size * k_size * k_size +
                                x * k_size * k_size + y * k_size + z;

                            // Matrix M index
                            int m_i = i + x - k_size / 2,
                                m_j = j + y - k_size / 2,
                                m_w = w + z - k_size / 2,
                                m_1d_idx;

                            if (m_i < 0 || m_i >= m_size ||
                                m_j < 0 || m_j >= m_size ||
                                m_w < 0 || m_w >= m_size)
                                m_1d_idx = -1;
                            else
                                m_1d_idx = m_i * m_size * m_size + m_j * m_size + m_w;

                            a[a_1d_idx] = (m_1d_idx == -1) ? 0 : m[m_1d_idx];
                        }
}

void baseConvGemm(float *a, float *k, float *o, unsigned int o_size, unsigned int k_size)
{
    for (int i = 0; i < o_size; i++)
    {
        o[i] = 0;

        for (int j = 0; j < k_size; j++)
            o[i] += a[i * k_size + j] * k[j];
    }
}

/* Max Pooling */

void baseMaxPool(float *m, float *o, unsigned int m_size, unsigned int k_size)
{
    assert(m_size % k_size == 0);

    for (int i = 0; i < m_size / k_size; i++)
        for (int j = 0; j < m_size / k_size; j++)
        {
            o[i * m_size / k_size + j] = m[i * k_size * m_size + j * k_size];

            for (int w = 0; w < k_size; w++)
                for (int x = 0; x < k_size; x++)
                {
                    unsigned int m_1d_idx = (i * k_size + w) * m_size + j * k_size + x;

                    if (m[m_1d_idx] > o[i * m_size / k_size + j])
                        o[i * m_size / k_size + j] = m[m_1d_idx];
                }
        }
}

void compMatMaxA(float *a, float *m, unsigned int m_size, unsigned int k_size)
{
    assert(m_size % k_size == 0);

    for (int i = 0; i < m_size / k_size; i++)
        for (int j = 0; j < m_size / k_size; j++)
            for (int w = 0; w < k_size; w++)
                for (int x = 0; x < k_size; x++)
                {
                    unsigned int a_1d_idx =
                        (i * m_size / k_size + j) * k_size * k_size +
                        w * k_size + x;

                    unsigned int m_1d_idx =
                        (i * k_size + w) * m_size +
                        j * k_size + x;

                    a[a_1d_idx] = m[m_1d_idx];
                }
}

void baseMaxPoolGemm(float *a, float *o, unsigned int m_size, unsigned int k_size)
{
    for (int i = 0; i < m_size / k_size; i++)
        for (int j = 0; j < m_size / k_size; j++)
        {
            unsigned int offset = (i * m_size / k_size + j) * k_size * k_size;
            float max = a[offset];

            for (int w = 0; w < k_size; w++)
                for (int x = 0; x < k_size; x++)
                {
                    unsigned int idx = offset + w * k_size + x;

                    if (a[idx] > max)
                        max = a[idx];
                }

            o[i * m_size / 2 + j] = max;
        }
}

/* ReLU */

void baseRelu(float *m, float *o, unsigned int m_size)
{
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size; j++)
            o[i * m_size + j] = (m[i * m_size + j] < 0) ? 0 : m[i * m_size + j];
}

/* Matrix Multiplication */

void baseMM(float *a, float *b, float *c, unsigned int m_size)
{
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size; j++)
        {
            c[i * m_size + j] = 0;

            for (int w = 0; w < m_size; w++)
                c[i * m_size + j] += a[i * m_size + w] * b[w * m_size + j];
        }
}

void transpose(float *m, float *a, unsigned int m_size)
{
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size; j++)
            a[j * m_size + i] = m[i * m_size + j];
}

void baseMMGemm(float *a, float *b, float *c, unsigned int m_size)
{
    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size; j++)
        {
            c[i * m_size + j] = 0;

            for (int w = 0; w < m_size; w++)
                c[i * m_size + j] += a[i * m_size + w] * b[j * m_size + w];
        }
}

/* Auxiliary functions */

void initRandArray(float *a, unsigned int size)
{
    for (int i = 0; i < size; i++)
        a[i] = (float) (rand() % 5 - 2);
}

bool compareArrays(float *a1, float *a2, unsigned int size)
{
    for (int i = 0; i < size; i++)
        if (a1[i] != a2[i])
            return false;

    return true;
}

void printMatrix_2D(float *m, unsigned int m_size)
{
    for (int i = 0; i < m_size; i++)
    {
        for (int j = 0; j < m_size; j++)
            printf("%3.0f ", m[i * m_size + j]);

        printf("\n");
    }
}

void printMatrix_3D(float *m, unsigned int m_size)
{
    for (int i = 0; i < m_size; i++)
    {
        for (int w = 0; w < m_size; w++)
        {
            printf("[ ");

            for (int j = 0; j < m_size; j++)
                printf("%3.0f ", m[i * m_size * m_size + j * m_size + w]);

            printf("] ");
        }

        printf("\n");
    }
}

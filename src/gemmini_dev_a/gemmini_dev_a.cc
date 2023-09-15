#include "gemmini_dev_a/gemmini_dev_a.hh"

namespace gem5
{
    GemminiDevA::GemminiDevA(const GemminiDevAParams &params) :
    NDP(params)
    {

    }

    uint64_t 
    GemminiDevA::readPI(uint64_t ridx)
    {
        switch (ridx)
        {
        case 7: return pi_status;
        default:
            panic("GemminiDevA does not have readable r[%lu] register!\n", ridx);
        }
    }

    void 
    GemminiDevA::writePI(uint64_t ridx, uint64_t data)
    {
        DPRINTF(GemminiDevAPI, "Gemmini device PI: %lu -> r[%lu]\n", data, ridx);

        if (!pi_status)
        {
            panic("Tried to started workload when previous one is not finished!\n");
        }

        switch (ridx)
        {
        case 0: pi_addr_m = data; break;
        case 1: pi_addr_k = data; break;
        case 2: pi_addr_o = data; break;
        case 3: pi_size_m = data; break;
        case 4: pi_size_k = data; break;
        case 5: pi_opcode = data; break;
        case 6:
            // Print PI current status
            DPRINTF(
                GemminiDevAPI,
                "GemminiDevA started processing...\n"
                "===== pi_addr_m = %lu\n"
                "===== pi_addr_k = %lu\n"
                "===== pi_addr_o = %lu\n"
                "===== pi_size_m = %lu\n"
                "===== pi_size_k = %lu\n"
                "===== pi_opcode = %lu\n"
                "===== pi_status = %lu\n",
                pi_addr_m,
                pi_addr_k,
                pi_addr_o,
                pi_size_m,
                pi_size_k,
                pi_opcode,
                pi_status
            );
            pi_status = 0;

            // Allocate memory for operands and result
            switch (pi_opcode)
            {
            case op_conv2d:       m_size =
                                  o_size = pi_size_m * pi_size_m;
                                  k_size = pi_size_k * pi_size_k;
                                  break;
            case op_conv2d_gemm:  m_size = pi_size_m * pi_size_m * pi_size_k * pi_size_k;
                                  k_size = pi_size_k * pi_size_k;
                                  o_size = pi_size_m * pi_size_m;
                                  break;
            case op_conv3d:       m_size =
                                  o_size = pi_size_m * pi_size_m * pi_size_m;
                                  k_size = pi_size_k * pi_size_k * pi_size_k;
                                  break;
            case op_conv3d_gemm:  m_size = pi_size_m * pi_size_m * pi_size_m * pi_size_k * pi_size_k * pi_size_k;
                                  k_size = pi_size_k * pi_size_k * pi_size_k;
                                  o_size = pi_size_m * pi_size_m * pi_size_m;
                                  break;
            case op_maxpool_gemm: 
            case op_maxpool:      m_size = pi_size_m * pi_size_m;
                                  k_size = 0;
                                  o_size = pi_size_m * pi_size_m / pi_size_k / pi_size_k;
                                  break;
            case op_relu:         m_size =
                                  o_size = pi_size_m * pi_size_m;
                                  k_size = 0;
                                  break;
            case op_mm_gemm:
            case op_mm:           m_size =
                                  k_size =
                                  o_size = pi_size_m * pi_size_m;
                                  break;
            default:              panic("An invalid opcode was issued!\n");
            }

            m_size *= sizeof(float);
            k_size *= sizeof(float);
            o_size *= sizeof(float);

            // Allocate memory for operands and result
            m = (float *) malloc(m_size);
            o = (float *) malloc(o_size);
            if (k_size > 0)
                k = (float *) malloc(k_size);

            // Start finite state machine
            process_fsm();
            break;
        default:
            panic("GemminiDevA does not have writable r[%lu] register!\n", ridx);
        }
    }

    void 
    GemminiDevA::recvData(Addr addr, uint8_t *data, size_t size)
    {
        DPRINTF(GemminiDevAMem, "GemminiDevA received %u bytes from %p\n", size, addr);

        if (data)
        {
            has_operands++;

            process_fsm();
        }
        else
        {
            pi_status = 1;
            has_operands = 0;

            free(m);
            free(o);
            if (k_size > 0)
                free(k);
        }
    }

    void
    GemminiDevA::process_fsm()
    {
        // Retrieve m from memory
        if (has_operands == 0)
        {
            DPRINTF(
                GemminiDevA,
                "Retrieving m from memory\n"
            );

            accessMemory(
                Addr(pi_addr_m),
                m_size,
                false,
                (uint8_t *) m
            );
        }
        // Retrieve k from memory
        else if(has_operands == 1 && k_size > 0)
        {
            DPRINTF(
                GemminiDevA,
                "Retrieving k from memory\n"
            );

            accessMemory(
                Addr(pi_addr_k),
                k_size,
                false,
                (uint8_t *) k
            );
        }
        else
        {
            DPRINTF(
                GemminiDevA,
                "Processing operation...\n"
            );

            switch (pi_opcode)
            {
            case op_conv2d:       baseConv_2D(m, k, o, pi_size_m, pi_size_k); break;
            case op_conv2d_gemm:  baseConvGemm(m, k, o, pi_size_m * pi_size_m, pi_size_k * pi_size_k); break;
            case op_conv3d:       baseConv_3D(m, k, o, pi_size_m, pi_size_k); break;
            case op_conv3d_gemm:  baseConvGemm(m, k, o, pi_size_m * pi_size_m * pi_size_m, pi_size_k * pi_size_k * pi_size_k); break;
            case op_maxpool:      baseMaxPool(m, o, pi_size_m, pi_size_k); break;
            case op_maxpool_gemm: baseMaxPoolGemm(m, o, pi_size_m, pi_size_k); break;
            case op_relu:         baseRelu(m, o, pi_size_m); break;
            case op_mm:           baseMM(m, k, o, pi_size_m); break;
            case op_mm_gemm:      baseMMGemm(m, k, o, pi_size_m); break;
            default: panic("Not implemented yet!\n");
            }

            accessMemory(
                Addr(pi_addr_o),
                o_size,
                true,
                (uint8_t *) o
            );
        }
    }

    /* ==================== BEGIN SPECIFIC METHODS ==================== */

    void
    GemminiDevA::baseConv_2D(float *m, float *k, float *o, unsigned int m_size, unsigned int k_size)
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

    void
    GemminiDevA::baseConv_3D(float *m, float *k, float *o, unsigned int m_size, unsigned int k_size)
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

    void
    GemminiDevA::baseConvGemm(float *a, float *k, float *o, unsigned int o_size, unsigned int k_size)
    {
        for (int i = 0; i < o_size; i++)
        {
            o[i] = 0;

            for (int j = 0; j < k_size; j++)
                o[i] += a[i * k_size + j] * k[j];
        }
    }

    void
    GemminiDevA::baseMaxPool(float *m, float *o, unsigned int m_size, unsigned int k_size)
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

    void
    GemminiDevA::baseMaxPoolGemm(float *a, float *o, unsigned int m_size, unsigned int k_size)
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

    void
    GemminiDevA::baseRelu(float *m, float *o, unsigned int m_size)
    {
        for (int i = 0; i < m_size; i++)
            for (int j = 0; j < m_size; j++)
                o[i * m_size + j] = (m[i * m_size + j] < 0) ? 0 : m[i * m_size + j];
    }

    void
    GemminiDevA::baseMM(float *a, float *b, float *c, unsigned int m_size)
    {
        for (int i = 0; i < m_size; i++)
            for (int j = 0; j < m_size; j++)
            {
                c[i * m_size + j] = 0;

                for (int w = 0; w < m_size; w++)
                    c[i * m_size + j] += a[i * m_size + w] * b[w * m_size + j];
            }
    }

    void
    GemminiDevA::baseMMGemm(float *a, float *b, float *c, unsigned int m_size)
    {
        for (int i = 0; i < m_size; i++)
            for (int j = 0; j < m_size; j++)
            {
                c[i * m_size + j] = 0;

                for (int w = 0; w < m_size; w++)
                    c[i * m_size + j] += a[i * m_size + w] * b[j * m_size + w];
            }
    }

    void 
    GemminiDevA::printMatrix_2D(float *m, unsigned int m_size)
    {
        for (int i = 0; i < m_size; i++)
        {
            for (int j = 0; j < m_size; j++)
                printf("%3.0f ", m[i * m_size + j]);

            printf("\n");
        }
    }

    void
    GemminiDevA::printMatrix_3D(float *m, unsigned int m_size)
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

    /* ===================== END SPECIFIC METHODS ==================== */

} // namespace gem5

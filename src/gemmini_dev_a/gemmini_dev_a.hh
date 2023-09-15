#ifndef __GemminiDevA_HH__
#define __GemminiDevA_HH__

#include "ndp/ndp.hh"

#include "params/GemminiDevA.hh"
#include "debug/GemminiDevA.hh"
#include "debug/GemminiDevAPI.hh"
#include "debug/GemminiDevAMem.hh"

namespace gem5
{
    class GemminiDevA : public NDP
    {
    private:

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

        uint64_t pi_addr_m = 0,
                 pi_addr_k = 0,
                 pi_addr_o = 0,
                 pi_size_m = 0,
                 pi_size_k = 0,
                 pi_opcode = 0,
                 pi_status = 1;

        unsigned int has_operands = 0;

        float *m, *k, *o;
        size_t m_size, k_size, o_size;

        void process_fsm();

    public:

        GemminiDevA(const GemminiDevAParams &params);

        uint64_t readPI(uint64_t ridx) override;

        void writePI(uint64_t ridx, uint64_t data) override;

        void recvData(Addr addr, uint8_t *data, size_t size) override;

        /* ==================== BEGIN SPECIFIC METHODS ==================== */

        void baseConv_2D(float *m, float *k, float *o, unsigned int m_size, unsigned int k_size);
        void baseConv_3D(float *m, float *k, float *o, unsigned int m_size, unsigned int k_size);
        void baseConvGemm(float *a, float *k, float *o, unsigned int o_size, unsigned int k_size);
        void baseMaxPool(float *m, float *o, unsigned int m_size, unsigned int k_size);
        void baseMaxPoolGemm(float *a, float *o, unsigned int m_size, unsigned int k_size);
        void baseRelu(float *m, float *o, unsigned int m_size);
        void baseMM(float *a, float *b, float *c, unsigned int m_size);
        void baseMMGemm(float *a, float *b, float *c, unsigned int m_size);
        void printMatrix_2D(float *m, unsigned int m_size);
        void printMatrix_3D(float *m, unsigned int m_size);

    };

}

#endif //__GemminiDevA_HH__

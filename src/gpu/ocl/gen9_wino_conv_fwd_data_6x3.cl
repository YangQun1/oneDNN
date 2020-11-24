/*******************************************************************************
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#define IW_BLOCK (OW_BLOCK + KW - 1)

#define WINO_D (WINO_M + WINO_R - 1)

#define TO_TYPE(value) ((DATA_T)value)

#define UTRANS_BLOCK 8
#define UTRANS_DATA_T CONCAT2(DATA_T, UTRANS_BLOCK)
#define AS_UTRANS_DATA_T CONCAT2(as_, COMP_DATA_T)
#define UTRANS_BLOCK_READ(ptr) \
    AS_UTRANS_DATA_T(BLOCK_READ8((const __global BLOCK_DATA_T *)ptr))
#define UTRANS_BLOCK_WRITE(data, ptr) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)ptr, AS_BLOCK_DATA8_T(data))

#define TRANS_BLOCK 4 // = (WINO_IC_BLOCK / (LWS_0 * LWS_1 / WINO_IW_BLOCK))
#define TRANS_DATA_T CONCAT2(DATA_T, TRANS_BLOCK)

#define COMP_BLOCK 8
#define COMP_DATA_T CONCAT2(DATA_T, COMP_BLOCK)
#define AS_COMP_DATA_T CONCAT2(as_, COMP_DATA_T)
#define COMP_READ(ptr) CONCAT2(vload, COMP_BLOCK)(0, ptr)
#define COMP_WRITE(data, ptr) CONCAT2(vstore, COMP_BLOCK)(data, 0, ptr)
#define COMP_BLOCK_READ(ptr) \
    AS_COMP_DATA_T(BLOCK_READ8((const __global BLOCK_DATA_T *)ptr))

#define COMP_UNROLL 2

#define OUT_TYPE_BLOCK 2 // = (WINO_OW_BLOCK / 7)
#define OUT_BLOCK_DATA_T CONCAT2(DATA_T, OUT_TYPE_BLOCK)

#define OUT_BLOCK_READ(ptr) CONCAT2(vload, OUT_TYPE_BLOCK)(0, ptr)
#define OUT_BLOCK_WRITE(data, ptr) \
    do { \
        OUT_BLOCK_DATA_T result = data; \
        unroll_for(int _i = 0; _i < OUT_TYPE_BLOCK; _i++) { \
            (ptr)[_i] = result[_i]; \
        } \
    } while (0)

static inline int off_nCdhw16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += n * (C / 16) * D * H * W * 16;
    off += (c / 16) * D * H * W * 16;
    off += d * H * W * 16;
    off += h * W * 16;
    off += w * 16;
    off += c % 16;
    return off;
}

static inline int off_NCdhw16n16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += (n / 16) * (C / 16) * D * H * W * 16 * 16;
    off += (c / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (n % 16) * 16;
    off += (c % 16);
    return off;
}

static inline int off_gIOdhw16i16o(int g, int o, int i, int d, int h, int w,
        int O, int I, int D, int H, int W) {
    int off = 0;
    off += g * (I / 16) * (O / 16) * D * H * W * 16 * 16;
    off += (i / 16) * (O / 16) * D * H * W * 16 * 16;
    off += (o / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (i % 16) * 16;
    off += (o % 16);
    return off;
}

static inline int src_off(int n, int c, int d, int h, int w) {
    if (SRC_W16C) return off_nCdhw16c(n, c, d, h, w, G * IC, 1, IH, IW);
    if (SRC_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * IC, 1, IH, IW);
    return 0;
}

static inline int wei_off(int g, int o, int i, int d, int h, int w) {
    return off_gIOdhw16i16o(g, o, i, d, h, w, OC, IC, 1, KH, KW);
}

static inline int U_off(int o, int i, int z, int w) {

    //  OIw8h16i16o
    const int ic_internal_block = 16;
    const int oc_internal_block = 16;
    int icb = i / ic_internal_block;
    int ic = i % ic_internal_block;
    int ocb = o / oc_internal_block;
    int oc = o % oc_internal_block;

    int off = ocb * (WINO_IC / ic_internal_block) * KW * ic_internal_block
            * WINO_D * oc_internal_block;
    off += icb * KW * ic_internal_block * WINO_D * oc_internal_block;
    off += w * ic_internal_block * WINO_D * oc_internal_block;
    off += z * ic_internal_block * oc_internal_block;
    off += ic * oc_internal_block;
    off += oc;

    return off;
}

static inline int V_off(int i, int z, int w, int block_size) {

    //V data format is 2C8h16w16c
    const int ic_internal_block = 16;
    const int iw_block = 16;

    int icb = i / ic_internal_block;
    int ic = i % ic_internal_block;
    int off = icb * WINO_D * iw_block * ic_internal_block;
    off += z * iw_block * ic_internal_block;
    off += w * ic_internal_block;
    off += ic;
    return off / block_size;
}

static inline int M_off(int o, int z, int w, int block_size) {

    //M data format is 8h16W16c2w
    const int ow_internal_block = 2;
    int owb = w / ow_internal_block;
    int ow = w % ow_internal_block;
    int off = z * OW_BLOCK / ow_internal_block * OC_BLOCK * ow_internal_block;
    off += owb * OC_BLOCK * ow_internal_block;
    off += o * ow_internal_block;
    off += ow;
    return off / block_size;
}

static inline int dst_off(int n, int c, int d, int h, int w) {
    if (DST_W16C) return off_nCdhw16c(n, c, d, h, w, G * OC, 1, OH, OW);
    if (DST_16N16C) return off_NCdhw16n16c(n, c, d, h, w, G * OC, 1, OH, OW);
    return 0;
}

__attribute__((reqd_work_group_size(OC_BLOCK, 1, 1)))
__attribute__((intel_reqd_sub_group_size(OC_BLOCK))) __kernel void
gen9_wino_wei_transform_6x3(
        __global DATA_T *U, const __global DATA_T *weights) {
    const uint weights_tile_width = 1;
    const uint weights_tile_height = WINO_M;
    const uint in_kw = get_global_id(1) * weights_tile_width;
    const uint in_kh = get_global_id(2) * weights_tile_height;

    const uint U_tile_width = 1;
    const uint U_tile_height = WINO_D;

    const uint out_kw = get_global_id(1) * U_tile_width;
    const uint out_kh = get_global_id(2) * U_tile_height;
    const uint oc0 = (get_group_id(0) % (WINO_OC / OC_BLOCK)) * OC_BLOCK;
    const uint ic = (get_group_id(0) / (WINO_OC / OC_BLOCK)) * 8;

    uint in_idx = wei_off(0, oc0, ic, 0, in_kh, in_kw);
    bool is_valid = ic < IC && oc0 < OC;

    UTRANS_DATA_T g[WINO_R];
    for (int i = 0; i < WINO_R; i++) {
        g[i] = is_valid ? UTRANS_BLOCK_READ(&weights[in_idx]) : 0;
        in_idx += wei_off(0, 0, 0, 0, 1, 0);
    }

    UTRANS_DATA_T out_tile[WINO_D];
    out_tile[0] = g[0];
    out_tile[1] = TO_TYPE(-2.0 / 9) * (g[0] + g[1] + g[2]);
    out_tile[2] = TO_TYPE(2.0 / 9) * (-g[0] + g[1] - g[2]);
    out_tile[3] = TO_TYPE(1.0 / 90) * g[0] + TO_TYPE(2.0 / 90) * g[1]
            + TO_TYPE(4.0 / 90) * g[2];
    out_tile[4] = TO_TYPE(1.0 / 90) * g[0] - TO_TYPE(2.0 / 90) * g[1]
            + TO_TYPE(4.0 / 90) * g[2];
    out_tile[5] = TO_TYPE(64.0 / 90) * g[0] + TO_TYPE(32.0 / 90) * g[1]
            + TO_TYPE(16.0 / 90) * g[2];
    out_tile[6] = TO_TYPE(64.0 / 90) * g[0] - TO_TYPE(32.0 / 90) * g[1]
            + TO_TYPE(16.0 / 90) * g[2];
    out_tile[7] = g[2];

    uint out_idx = U_off(oc0, ic, out_kh, out_kw);

    unroll_for(int i = 0; i < WINO_D; i++) {
        UTRANS_BLOCK_WRITE(out_tile[i], &U[out_idx]);
        out_idx += U_off(0, 0, 1, 0);
    }
}

#define DOTi(_result, _A, _B) \
    { _result = mad(_A, _B, _result); }

__attribute__((reqd_work_group_size(16, 8, 1)))
__attribute__((intel_reqd_sub_group_size(16))) __kernel void
gen9_wino_conv_fwd_6x3(__global DATA_T *dst, const __global DATA_T *src,
        const __global DATA_T *U_param,
        const __global DATA_T *bias POST_OP_ARGS) {
    //               (DxC2)x(UxWx8c)
    const uint slm_size = (WINO_IC_BLOCK * WINO_D * IW_BLOCK) / TRANS_BLOCK;
    __local TRANS_DATA_T V[slm_size]; // 8 KB

    const DATA_T sc = TO_TYPE(0.1);
    const DATA_T scl = TO_TYPE(1.0) / sc;
    const TRANS_DATA_T scl_vec = (TRANS_DATA_T)(sc, sc, sc, sc);

    const int ow0 = get_group_id(0) * OW_BLOCK;
    const int oh = get_group_id(1) * OH_BLOCK;
    const int gid2 = get_group_id(2);
    const int oc0 = (gid2 % (OC / OC_BLOCK)) * OC_BLOCK;
    const int mb = gid2 / (OC / OC_BLOCK);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    uint lxd8 = lx / 8;
    uint lxm8 = lx % 8;
    uint lxd2 = lx / 2;
    uint lxm2 = lx % 2;

    const int oc = oc0 + lx;
    const int ow = ow0 + 2 * ly;

    // Load ic32ih8iw16 input tile, with 2 pixel overlap in ih and iw.
    // Compute oc16oh6ow14 output tile.

    int iw0_write = ly * 2 + lxd8;
    int iw0_read = lxd2;
    int iw = ow0 + iw0_write - PW;
    int ih = oh - PH;
    int ic0_write = lxm8 * TRANS_BLOCK;
    int ic0_read = 8 * lxm2;

    // Initialize variables to accumulate intermediate output tile
    const int M_size = OW_BLOCK;
    DATA_T M[M_size];

    for (int i = 0; i < M_size; i++) {
        M[i] = 0;
    }

    // Computation is separated into three main stages, load/transform input,
    // compute intermediate output block, and transform/store final output.
    // Between these stages, the dimensions handled by local work groups
    // changes.

    // Buffers used to load and transform ic32ih8iw16 src tile into V
    // Each local thread transforms a block with dimensions c4h8w1
    // For the computation, src_i traverses ih dimension, ly * 2 + lx/8
    // traverses iw dimension, and lx % 8 traverses ic dimension
    const __global DATA_T *src_load = src + src_off(mb, ic0_write, 0, ih, iw);
    const int V_write_idx = V_off(ic0_write, 0, iw0_write, TRANS_BLOCK);
    __local TRANS_DATA_T *V_write = &V[V_write_idx];

    // Buffers used to compute oc16oh8ow14 intermediate output tile. Each
    // local thread transforms a block with dimensions c1h1w14. For the
    // computed output, M_i traverses ow dimension, ly traverses oh
    // dimension, and lx traverses oc dimension.
    const __global DATA_T *U = U_param + U_off(oc, 0, ly, 0);
    const int V_read_idx = V_off(ic0_read, ly, iw0_read, TRANS_BLOCK);
    __local const COMP_DATA_T *V_read
            = (__local const COMP_DATA_T *)&V[V_read_idx]; // ly * 64 + lx * 2;

    __attribute__((opencl_unroll_hint(1))) for (uint c = 0; c < IC;
                                                c += WINO_IC_BLOCK) {
        // Load and transform ic32ih8iw16 src tile into V
        {
            bool x_in = 0 <= iw && iw < IW && ic0_read + c < IC;
            TRANS_DATA_T src[WINO_D];
            for (int index = 0; index < WINO_D; index++) {
                bool y_in = 0 <= (ih + index) && (ih + index) < IH && x_in;
                src[index] = y_in ? *((const __global TRANS_DATA_T *)(src_load
                                     + src_off(0, 0, 0, index, 0)))
                                  : 0;

                //Scale input to prevent intermediate computations overflow in
                //some cases, output is adjusted with the same scale factor
                //after main computation
                src[index] = src[index] * scl_vec;
            }

            // Compute Winograd f6x3 data transform and store components in SLM.
            V_write[V_off(0, 0, 0, TRANS_BLOCK)] = src[0]
                    - TO_TYPE(5.25) * src[2] + TO_TYPE(5.25) * src[4] - src[6];

            TRANS_DATA_T x0 = src[1] - TO_TYPE(4.25) * src[3] + src[5];
            TRANS_DATA_T x1 = src[2] - TO_TYPE(4.25) * src[4] + src[6];

            V_write[V_off(0, 1, 0, TRANS_BLOCK)] = x1 + x0;
            V_write[V_off(0, 2, 0, TRANS_BLOCK)] = x1 - x0;

            TRANS_DATA_T x2 = TO_TYPE(-5) * src[3] + src[1];
            TRANS_DATA_T x3 = TO_TYPE(4) * src[5] + x2;
            TRANS_DATA_T x4 = TO_TYPE(0.25) * src[2] + src[6];
            TRANS_DATA_T x5 = TO_TYPE(-1.25) * src[4] + x4;

            V_write[V_off(0, 3, 0, TRANS_BLOCK)] = TO_TYPE(0.5) * x3 + x5;
            V_write[V_off(0, 4, 0, TRANS_BLOCK)] = TO_TYPE(-0.5) * x3 + x5;

            TRANS_DATA_T x6 = TO_TYPE(4) * src[1] + src[5];
            TRANS_DATA_T x7 = TO_TYPE(-5) * src[3] + x6;
            TRANS_DATA_T x8 = TO_TYPE(4) * src[2] + src[6];
            TRANS_DATA_T x9 = TO_TYPE(-5) * src[4] + x8;

            V_write[V_off(0, 5, 0, TRANS_BLOCK)] = TO_TYPE(+0.5) * x7 + x9;
            V_write[V_off(0, 6, 0, TRANS_BLOCK)] = TO_TYPE(-0.5) * x7 + x9;

            V_write[V_off(0, 7, 0, TRANS_BLOCK)] = -src[1]
                    + TO_TYPE(5.25) * src[3] - TO_TYPE(5.25) * src[5] + src[7];
        }

        src_load += src_off(0, WINO_IC_BLOCK, 0, 0, 0);
        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate oc16oh8ow14 intermediate output tile stored in the M_i
        __local const COMP_DATA_T *V_read_outer = V_read;

        const int outer_c_blocking = COMP_UNROLL * COMP_BLOCK;
        __attribute__((opencl_unroll_hint(
                1))) for (uint c_outer = 0; c_outer < WINO_IC_BLOCK;
                          c_outer += outer_c_blocking) {
            // Fetch 16 input components, spread across subgroup.
            DATA_T V_block[outer_c_blocking];
            unroll_for(int i = 0; i < outer_c_blocking; i++) {
                COMP_WRITE(
                        V_read_outer[V_off(0, 0, i * COMP_BLOCK, COMP_BLOCK)],
                        &V_block[i * COMP_BLOCK]);
            }
            V_read_outer += V_off(outer_c_blocking, 0, 0, COMP_BLOCK);

#define V_BLOCK(ic, iw) \
    sub_group_broadcast( \
            V_block[(ic) % 8 + 8 * ((iw) / 8)], 2 * ((iw) % 8) + ((ic) / 8))

            unroll_for(int c_inner = 0; c_inner < outer_c_blocking;
                       c_inner += COMP_BLOCK) {
                unroll_for(int kw_in = 0; kw_in < KW; kw_in++) {
                    const COMP_DATA_T f0
                            = COMP_BLOCK_READ(&U[U_off(0, 0, 0, kw_in)]);

                    unroll_for(int c_in = 0; c_in < COMP_BLOCK; c_in++) {
                        unroll_for(int ow_in = 0; ow_in < OW_BLOCK; ow_in++) {
                            DOTi(M[ow_in], f0[c_in],
                                    V_BLOCK(c_in + c_inner, kw_in + ow_in));
                        }
                    }
                }

                U += U_off(0, COMP_BLOCK, 0, 0);
            }
            U += U_off(0, COMP_UNROLL * COMP_BLOCK, 0, 0)
                    - COMP_UNROLL * U_off(0, COMP_BLOCK, 0, 0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store intermediate output tile to SLM.
    {
        __local DATA_T *M_write = (__local DATA_T *)&V[M_off(0, ly, 0, 4)];
        M_write += M_off(lx, 0, 0, 1);

        for (int i = 0; i < M_size; i++) {
            M_write[M_off(0, 0, i, 1)] = M[i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Transform and store final oc16oh6ow14 output tile.
    if (ly < OW_BLOCK / OUT_TYPE_BLOCK) {
        // Load multiplies from SLM.
        __local const OUT_BLOCK_DATA_T *M_read
                = (__local OUT_BLOCK_DATA_T *)&V[M_off(0, 0, ly * 2, 4)];
        M_read += M_off(lx, 0, 0, OUT_TYPE_BLOCK);

        OUT_BLOCK_DATA_T M[WINO_D];
        for (int i = 0; i < WINO_D; i++) {
            M[i] = M_read[M_off(0, i, 0, OUT_TYPE_BLOCK)];
        }

        // Inverse Transform.
        OUT_BLOCK_DATA_T x0 = M[1] + M[2];
        OUT_BLOCK_DATA_T x1 = M[1] - M[2];

        OUT_BLOCK_DATA_T x2 = M[3] + M[4];
        OUT_BLOCK_DATA_T x3 = M[3] - M[4];

        OUT_BLOCK_DATA_T x4 = M[5] + M[6];
        OUT_BLOCK_DATA_T x5 = M[5] - M[6];

        OUT_BLOCK_DATA_T C[WINO_M];
        DATA_T *C_dat = C;

        C[0] = M[0] + x0 + x2 + x4;
        C[1] = x1 + TO_TYPE(2) * x3 + TO_TYPE(0.5f) * x5;
        C[2] = x0 + TO_TYPE(4.f) * x2 + TO_TYPE(0.25f) * x4;
        C[3] = x1 + TO_TYPE(8.f) * x3 + TO_TYPE(0.125f) * x5;
        C[4] = x0 + TO_TYPE(16.f) * x2 + TO_TYPE(0.0625f) * x4;
        C[5] = x1 + TO_TYPE(32.f) * x3 + TO_TYPE(0.03125f) * x5 + M[7];

        unroll_for(int i = 0; i < WINO_M; i++) { C[i] = C[i] * scl; }

        // Write data
        int dst_idx = dst_off(mb, oc, 0, oh, ow);
        const int w_size = dst_off(0, 0, 0, 0, 1);
        const int h_size = dst_off(0, 0, 0, 1, 0);

        if (WITH_BIAS || WITH_POST_OP) {
            const int c_size = WINO_M * OUT_TYPE_BLOCK;
            if (WITH_BIAS) {
                for (int oh_block = 0; oh_block < WINO_M; oh_block++) {
                    for (int ow_block = 0; ow_block < OUT_TYPE_BLOCK;
                            ow_block++) {
                        const int c_off = oh_block * OUT_TYPE_BLOCK + ow_block;
                        C_dat[c_off] += (OC_WO_PADDING % OC_BLOCK == 0
                                                || oc < OC_WO_PADDING)
                                ? bias[oc]
                                : DATA_ZERO;
                    }
                }
            }

            DATA_T S[c_size];
            if (WITH_SUM) {
                for (int oh_block = 0; oh_block < WINO_M; oh_block++) {
                    bool valid_oh = OH % OH_BLOCK == 0 || oh + oh_block < OH;
                    for (int ow_block = 0; ow_block < OUT_TYPE_BLOCK;
                            ow_block++) {
                        const int s_off = oh_block * OUT_TYPE_BLOCK + ow_block;
                        const int dst_off = dst_idx + oh_block * h_size
                                + ow_block * w_size;
                        bool valid_ow
                                = OW % OW_BLOCK == 0 || ow + ow_block < OW;
                        S[s_off] = valid_oh && valid_ow
                                ? dst[dst_idx + oh_block * h_size
                                        + ow_block * w_size]
                                : 0;
                    }
                }
            }

            for (int didx = 0; didx < c_size; ++didx) {
                float accum = CONVERT_FLOAT_T(C_dat[didx]);
                float sum = CONVERT_FLOAT_T(S[didx]);
                int po_oc = oc;

                APPLY_POST_OPS_SERIAL_BINARY_2D(
                        C_dat, DATA_T, S, DATA_T, mb, 1, po_oc, 1);
                C_dat[didx] = TO_DATA_T(accum);
            }
        }

        unroll_for(int h_off = 0; h_off < WINO_M; h_off++) {
            if (h_off == 0 || OH % OH_BLOCK == 0 || oh + h_off < OH) {
                unroll_for(int w_off = 0; w_off < OUT_TYPE_BLOCK; w_off++) {
                    int c_off = 2 * h_off + w_off;
                    if (OW % OW_BLOCK == 0 || ow + w_off < OW)
                        dst[dst_idx + h_off * h_size + w_off * w_size]
                                = C_dat[c_off];
                }
            }
        }
    }
}

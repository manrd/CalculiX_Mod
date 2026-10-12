/*     CalculiX - A 3-dimensional finite element program                   */
/*     GPU (CUDA Driver API) + Load-Balanced OpenMP CSR CG Solver          */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#include <immintrin.h>
#endif
#include "CalculiX.h"
#include "ccx_prof.h"

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef int CUresult;

#define CUDA_SUCCESS 0

typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*pfn_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (*pfn_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*pfn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*pfn_cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, int*, void**);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*pfn_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*pfn_cuMemFree)(CUdeviceptr);
typedef CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (*pfn_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void**, void**);
typedef CUresult (*pfn_cuCtxSynchronize)(void);

static int cuda_init_state = 0; /* 0 = untried, 1 = ready, -1 = unavailable */
static CUcontext cuda_ctx = NULL;
static CUmodule cuda_mod = NULL;
static char cuda_dev_name[128] = "CUDA GPU";

/* f32 GPU kernels */
static CUfunction fn_spmv_f32 = NULL;
static CUfunction fn_dot_f32 = NULL;
static CUfunction fn_finish_pz = NULL;
static CUfunction fn_axpy_r_reduce_f32 = NULL;
static CUfunction fn_finish_rr = NULL;
static CUfunction fn_xpay_f32 = NULL;

/* f64 GPU kernels */
static CUfunction fn_spmv_f64 = NULL;
static CUfunction fn_dot_f64 = NULL;
static CUfunction fn_axpy_r_reduce_f64 = NULL;
static CUfunction fn_xpay_f64 = NULL;

static pfn_cuInit f_cuInit;
static pfn_cuDeviceGet f_cuDeviceGet;
static pfn_cuDeviceGetName f_cuDeviceGetName;
static pfn_cuCtxCreate f_cuCtxCreate;
static pfn_cuCtxSetCurrent f_cuCtxSetCurrent;
static pfn_cuModuleLoadDataEx f_cuModuleLoadDataEx;
static pfn_cuModuleGetFunction f_cuModuleGetFunction;
static pfn_cuMemAlloc f_cuMemAlloc;
static pfn_cuMemFree f_cuMemFree;
static pfn_cuMemcpyHtoD f_cuMemcpyHtoD;
static pfn_cuMemcpyDtoH f_cuMemcpyDtoH;
static pfn_cuMemsetD32 f_cuMemsetD32;
static pfn_cuLaunchKernel f_cuLaunchKernel;
static pfn_cuCtxSynchronize f_cuCtxSynchronize;

/* Device-resident CG scalar state buffer (40 bytes in GPU global memory):
   offset 0:  double rr     (current r^T r)
   offset 8:  double pz     (current p^T z)
   offset 16: double qk     (step size alpha = rr / pz)
   offset 24: double ekm1   (beta = rr_new / rr_old)
   offset 32: double ram    (max_i |r_i|)
*/
typedef struct {
  double rr;
  double pz;
  double qk;
  double ekm1;
  double ram;
} CgGpuState;

/* Hand-tuned PTX 6.5 (Turing sm_75) with non-coherent read-only cache loads (ld.global.nc / __ldg)
   for both pure-float32 (f32) and full-float64 (f64) CSR Conjugate Gradient */
static const char *cuda_ptx_src =
".version 6.5\n"
".target sm_75\n"
".address_size 64\n"
"\n"
"/* === 1a. Subwarp-8 CSR SpMV (pure float32 with ld.global.nc read-only cache) === */\n"
".visible .entry csr_spmv_subwarp8_f32(\n"
"    .param .u32 param_n,\n"
"    .param .u64 param_row_ptr,\n"
"    .param .u64 param_col_idx,\n"
"    .param .u64 param_val,\n"
"    .param .u64 param_x,\n"
"    .param .u64 param_y\n"
")\n"
"{\n"
"    .reg .pred %p<3>;\n"
"    .reg .b32 %r<16>;\n"
"    .reg .b64 %rd<16>;\n"
"    .reg .f32 %f<6>;\n"
"\n"
"    ld.param.u32 %r1, [param_n];\n"
"    mov.u32 %r2, %ctaid.x;\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %tid.x;\n"
"    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
"    shr.u32 %r6, %r5, 3;\n"
"    and.b32 %r15, %r5, 7;\n"
"    setp.ge.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_SPMV32_EXIT;\n"
"\n"
"    ld.param.u64 %rd1, [param_row_ptr];\n"
"    ld.param.u64 %rd2, [param_col_idx];\n"
"    ld.param.u64 %rd3, [param_val];\n"
"    ld.param.u64 %rd4, [param_x];\n"
"    ld.param.u64 %rd5, [param_y];\n"
"\n"
"    mul.wide.s32 %rd6, %r6, 4;\n"
"    add.u64 %rd6, %rd1, %rd6;\n"
"    ld.global.nc.u32 %r7, [%rd6];\n"
"    ld.global.nc.u32 %r8, [%rd6+4];\n"
"\n"
"    mov.f32 %f1, 0f00000000;\n"
"    add.s32 %r9, %r7, %r15;\n"
"    setp.ge.s32 %p2, %r9, %r8;\n"
"    @%p2 bra L_SPMV32_REDUCE;\n"
"\n"
"    mul.wide.s32 %rd7, %r9, 4;\n"
"    add.u64 %rd8, %rd2, %rd7;\n"
"    add.u64 %rd10, %rd3, %rd7;\n"
"\n"
"L_SPMV32_LOOP:\n"
"    ld.global.nc.u32 %r10, [%rd8];\n"
"    ld.global.nc.f32 %f2, [%rd10];\n"
"    mul.wide.s32 %rd11, %r10, 4;\n"
"    add.u64 %rd12, %rd4, %rd11;\n"
"    ld.global.nc.f32 %f3, [%rd12];\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"    add.u64 %rd8, %rd8, 32;\n"
"    add.u64 %rd10, %rd10, 32;\n"
"    add.s32 %r9, %r9, 8;\n"
"    setp.lt.s32 %p2, %r9, %r8;\n"
"    @%p2 bra L_SPMV32_LOOP;\n"
"\n"
"L_SPMV32_REDUCE:\n"
"    shfl.sync.down.b32 %f4, %f1, 4, 31, 0xffffffff;\n"
"    add.f32 %f1, %f1, %f4;\n"
"    shfl.sync.down.b32 %f4, %f1, 2, 31, 0xffffffff;\n"
"    add.f32 %f1, %f1, %f4;\n"
"    shfl.sync.down.b32 %f4, %f1, 1, 31, 0xffffffff;\n"
"    add.f32 %f1, %f1, %f4;\n"
"\n"
"    setp.ne.u32 %p1, %r15, 0;\n"
"    @%p1 bra L_SPMV32_EXIT;\n"
"\n"
"    mul.wide.s32 %rd13, %r6, 4;\n"
"    add.u64 %rd14, %rd5, %rd13;\n"
"    st.global.f32 [%rd14], %f1;\n"
"\n"
"L_SPMV32_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 1b. Subwarp-8 CSR SpMV (full float64 with ld.global.nc read-only cache) === */\n"
".visible .entry csr_spmv_subwarp8_f64(\n"
"    .param .u32 param_n,\n"
"    .param .u64 param_row_ptr,\n"
"    .param .u64 param_col_idx,\n"
"    .param .u64 param_val,\n"
"    .param .u64 param_x,\n"
"    .param .u64 param_y\n"
")\n"
"{\n"
"    .reg .pred %p<3>;\n"
"    .reg .b32 %r<16>;\n"
"    .reg .b64 %rd<16>;\n"
"    .reg .f64 %fd<6>;\n"
"\n"
"    ld.param.u32 %r1, [param_n];\n"
"    mov.u32 %r2, %ctaid.x;\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %tid.x;\n"
"    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
"    shr.u32 %r6, %r5, 3;\n"
"    and.b32 %r15, %r5, 7;\n"
"    setp.ge.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_SPMV64_EXIT;\n"
"\n"
"    ld.param.u64 %rd1, [param_row_ptr];\n"
"    ld.param.u64 %rd2, [param_col_idx];\n"
"    ld.param.u64 %rd3, [param_val];\n"
"    ld.param.u64 %rd4, [param_x];\n"
"    ld.param.u64 %rd5, [param_y];\n"
"\n"
"    mul.wide.s32 %rd6, %r6, 4;\n"
"    add.u64 %rd6, %rd1, %rd6;\n"
"    ld.global.nc.u32 %r7, [%rd6];\n"
"    ld.global.nc.u32 %r8, [%rd6+4];\n"
"\n"
"    mov.f64 %fd1, 0d0000000000000000;\n"
"    add.s32 %r9, %r7, %r15;\n"
"    setp.ge.s32 %p2, %r9, %r8;\n"
"    @%p2 bra L_SPMV64_REDUCE;\n"
"\n"
"    mul.wide.s32 %rd7, %r9, 4;\n"
"    add.u64 %rd8, %rd2, %rd7;\n"
"    mul.wide.s32 %rd9, %r9, 8;\n"
"    add.u64 %rd10, %rd3, %rd9;\n"
"\n"
"L_SPMV64_LOOP:\n"
"    ld.global.nc.u32 %r10, [%rd8];\n"
"    ld.global.nc.f64 %fd2, [%rd10];\n"
"    mul.wide.s32 %rd11, %r10, 8;\n"
"    add.u64 %rd12, %rd4, %rd11;\n"
"    ld.global.nc.f64 %fd3, [%rd12];\n"
"    fma.rn.f64 %fd1, %fd2, %fd3, %fd1;\n"
"    add.u64 %rd8, %rd8, 32;\n"
"    add.u64 %rd10, %rd10, 64;\n"
"    add.s32 %r9, %r9, 8;\n"
"    setp.lt.s32 %p2, %r9, %r8;\n"
"    @%p2 bra L_SPMV64_LOOP;\n"
"\n"
"L_SPMV64_REDUCE:\n"
"    mov.b64 {%r11, %r12}, %fd1;\n"
"    shfl.sync.down.b32 %r13, %r11, 4, 31, 0xffffffff;\n"
"    shfl.sync.down.b32 %r14, %r12, 4, 31, 0xffffffff;\n"
"    mov.b64 %fd4, {%r13, %r14};\n"
"    add.f64 %fd1, %fd1, %fd4;\n"
"\n"
"    mov.b64 {%r11, %r12}, %fd1;\n"
"    shfl.sync.down.b32 %r13, %r11, 2, 31, 0xffffffff;\n"
"    shfl.sync.down.b32 %r14, %r12, 2, 31, 0xffffffff;\n"
"    mov.b64 %fd4, {%r13, %r14};\n"
"    add.f64 %fd1, %fd1, %fd4;\n"
"\n"
"    mov.b64 {%r11, %r12}, %fd1;\n"
"    shfl.sync.down.b32 %r13, %r11, 1, 31, 0xffffffff;\n"
"    shfl.sync.down.b32 %r14, %r12, 1, 31, 0xffffffff;\n"
"    mov.b64 %fd4, {%r13, %r14};\n"
"    add.f64 %fd1, %fd1, %fd4;\n"
"\n"
"    setp.ne.u32 %p1, %r15, 0;\n"
"    @%p1 bra L_SPMV64_EXIT;\n"
"\n"
"    mul.wide.s32 %rd13, %r6, 8;\n"
"    add.u64 %rd14, %rd5, %rd13;\n"
"    st.global.f64 [%rd14], %fd1;\n"
"\n"
"L_SPMV64_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 2a. Partial dot product (p, z) in float32 -> float64 === */\n"
".visible .entry vec_dot_f32(\n"
"    .param .u32 param_n,\n"
"    .param .u64 param_a,\n"
"    .param .u64 param_b,\n"
"    .param .u64 param_out\n"
")\n"
"{\n"
"    .shared .align 8 .b8 s_mem[2048];\n"
"    .reg .pred %p<3>;\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<12>;\n"
"    .reg .f32 %f<3>;\n"
"    .reg .f64 %fd<5>;\n"
"\n"
"    ld.param.u32 %r1, [param_n];\n"
"    ld.param.u64 %rd1, [param_a];\n"
"    ld.param.u64 %rd2, [param_b];\n"
"    ld.param.u64 %rd3, [param_out];\n"
"\n"
"    mov.u32 %r2, %tid.x;\n"
"    mov.u32 %r3, %ctaid.x;\n"
"    mov.u32 %r4, %ntid.x;\n"
"    mov.u32 %r5, %nctaid.x;\n"
"    mad.lo.s32 %r6, %r3, %r4, %r2;\n"
"    mul.lo.s32 %r7, %r4, %r5;\n"
"\n"
"    mov.f64 %fd1, 0d0000000000000000;\n"
"    setp.ge.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_DOT32_REDUCE;\n"
"\n"
"L_DOT32_LOOP:\n"
"    mul.wide.s32 %rd4, %r6, 4;\n"
"    add.u64 %rd5, %rd1, %rd4;\n"
"    add.u64 %rd6, %rd2, %rd4;\n"
"    ld.global.nc.f32 %f1, [%rd5];\n"
"    ld.global.nc.f32 %f2, [%rd6];\n"
"    cvt.f64.f32 %fd2, %f1;\n"
"    cvt.f64.f32 %fd3, %f2;\n"
"    fma.rn.f64 %fd1, %fd2, %fd3, %fd1;\n"
"    add.s32 %r6, %r6, %r7;\n"
"    setp.lt.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_DOT32_LOOP;\n"
"\n"
"L_DOT32_REDUCE:\n"
"    mov.u64 %rd7, s_mem;\n"
"    mul.wide.u32 %rd8, %r2, 8;\n"
"    add.u64 %rd9, %rd7, %rd8;\n"
"    st.shared.f64 [%rd9], %fd1;\n"
"    bar.sync 0;\n"
"\n"
"    mov.u32 %r8, 128;\n"
"L_DOT32_TREE:\n"
"    setp.ge.u32 %p2, %r2, %r8;\n"
"    @%p2 bra L_DOT32_SYNC;\n"
"    mul.wide.u32 %rd10, %r8, 8;\n"
"    add.u64 %rd10, %rd9, %rd10;\n"
"    ld.shared.f64 %fd4, [%rd10];\n"
"    ld.shared.f64 %fd1, [%rd9];\n"
"    add.f64 %fd1, %fd1, %fd4;\n"
"    st.shared.f64 [%rd9], %fd1;\n"
"L_DOT32_SYNC:\n"
"    bar.sync 0;\n"
"    shr.u32 %r8, %r8, 1;\n"
"    setp.gt.u32 %p2, %r8, 0;\n"
"    @%p2 bra L_DOT32_TREE;\n"
"\n"
"    setp.ne.u32 %p1, %r2, 0;\n"
"    @%p1 bra L_DOT32_EXIT;\n"
"    ld.shared.f64 %fd1, [%rd7];\n"
"    mul.wide.u32 %rd11, %r3, 8;\n"
"    add.u64 %rd11, %rd3, %rd11;\n"
"    st.global.f64 [%rd11], %fd1;\n"
"L_DOT32_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 2b. Partial dot product (p, z) in float64 === */\n"
".visible .entry vec_dot_f64(\n"
"    .param .u32 param_n,\n"
"    .param .u64 param_a,\n"
"    .param .u64 param_b,\n"
"    .param .u64 param_out\n"
")\n"
"{\n"
"    .shared .align 8 .b8 s_mem[2048];\n"
"    .reg .pred %p<3>;\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<12>;\n"
"    .reg .f64 %fd<5>;\n"
"\n"
"    ld.param.u32 %r1, [param_n];\n"
"    ld.param.u64 %rd1, [param_a];\n"
"    ld.param.u64 %rd2, [param_b];\n"
"    ld.param.u64 %rd3, [param_out];\n"
"\n"
"    mov.u32 %r2, %tid.x;\n"
"    mov.u32 %r3, %ctaid.x;\n"
"    mov.u32 %r4, %ntid.x;\n"
"    mov.u32 %r5, %nctaid.x;\n"
"    mad.lo.s32 %r6, %r3, %r4, %r2;\n"
"    mul.lo.s32 %r7, %r4, %r5;\n"
"\n"
"    mov.f64 %fd1, 0d0000000000000000;\n"
"    setp.ge.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_DOT64_REDUCE;\n"
"\n"
"L_DOT64_LOOP:\n"
"    mul.wide.s32 %rd4, %r6, 8;\n"
"    add.u64 %rd5, %rd1, %rd4;\n"
"    add.u64 %rd6, %rd2, %rd4;\n"
"    ld.global.nc.f64 %fd2, [%rd5];\n"
"    ld.global.nc.f64 %fd3, [%rd6];\n"
"    fma.rn.f64 %fd1, %fd2, %fd3, %fd1;\n"
"    add.s32 %r6, %r6, %r7;\n"
"    setp.lt.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_DOT64_LOOP;\n"
"\n"
"L_DOT64_REDUCE:\n"
"    mov.u64 %rd7, s_mem;\n"
"    mul.wide.u32 %rd8, %r2, 8;\n"
"    add.u64 %rd9, %rd7, %rd8;\n"
"    st.shared.f64 [%rd9], %fd1;\n"
"    bar.sync 0;\n"
"\n"
"    mov.u32 %r8, 128;\n"
"L_DOT64_TREE:\n"
"    setp.ge.u32 %p2, %r2, %r8;\n"
"    @%p2 bra L_DOT64_SYNC;\n"
"    mul.wide.u32 %rd10, %r8, 8;\n"
"    add.u64 %rd10, %rd9, %rd10;\n"
"    ld.shared.f64 %fd4, [%rd10];\n"
"    ld.shared.f64 %fd1, [%rd9];\n"
"    add.f64 %fd1, %fd1, %fd4;\n"
"    st.shared.f64 [%rd9], %fd1;\n"
"L_DOT64_SYNC:\n"
"    bar.sync 0;\n"
"    shr.u32 %r8, %r8, 1;\n"
"    setp.gt.u32 %p2, %r8, 0;\n"
"    @%p2 bra L_DOT64_TREE;\n"
"\n"
"    setp.ne.u32 %p1, %r2, 0;\n"
"    @%p1 bra L_DOT64_EXIT;\n"
"    ld.shared.f64 %fd1, [%rd7];\n"
"    mul.wide.u32 %rd11, %r3, 8;\n"
"    add.u64 %rd11, %rd3, %rd11;\n"
"    st.global.f64 [%rd11], %fd1;\n"
"L_DOT64_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 3. Single-block final reduction of pz and computation of qk = rr / pz === */\n"
".visible .entry finish_dot_pz(\n"
"    .param .u32 param_nblocks,\n"
"    .param .u64 param_buf,\n"
"    .param .u64 param_state\n"
")\n"
"{\n"
"    .shared .align 8 .b8 s_pz[512];\n"
"    .reg .pred %p<3>;\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<8>;\n"
"    .reg .f64 %fd<5>;\n"
"\n"
"    ld.param.u32 %r1, [param_nblocks];\n"
"    ld.param.u64 %rd1, [param_buf];\n"
"    ld.param.u64 %rd2, [param_state];\n"
"\n"
"    mov.u32 %r2, %tid.x;\n"
"    mov.f64 %fd1, 0d0000000000000000;\n"
"    setp.ge.u32 %p1, %r2, %r1;\n"
"    @%p1 bra L_FPZ_STORE;\n"
"    mul.wide.u32 %rd3, %r2, 8;\n"
"    add.u64 %rd3, %rd1, %rd3;\n"
"    ld.global.f64 %fd1, [%rd3];\n"
"L_FPZ_STORE:\n"
"    mov.u64 %rd4, s_pz;\n"
"    mul.wide.u32 %rd5, %r2, 8;\n"
"    add.u64 %rd6, %rd4, %rd5;\n"
"    st.shared.f64 [%rd6], %fd1;\n"
"    bar.sync 0;\n"
"\n"
"    mov.u32 %r3, 32;\n"
"L_FPZ_TREE:\n"
"    setp.ge.u32 %p2, %r2, %r3;\n"
"    @%p2 bra L_FPZ_SYNC;\n"
"    mul.wide.u32 %rd7, %r3, 8;\n"
"    add.u64 %rd7, %rd6, %rd7;\n"
"    ld.shared.f64 %fd2, [%rd7];\n"
"    ld.shared.f64 %fd1, [%rd6];\n"
"    add.f64 %fd1, %fd1, %fd2;\n"
"    st.shared.f64 [%rd6], %fd1;\n"
"L_FPZ_SYNC:\n"
"    bar.sync 0;\n"
"    shr.u32 %r3, %r3, 1;\n"
"    setp.gt.u32 %p2, %r3, 0;\n"
"    @%p2 bra L_FPZ_TREE;\n"
"\n"
"    setp.ne.u32 %p1, %r2, 0;\n"
"    @%p1 bra L_FPZ_EXIT;\n"
"    ld.shared.f64 %fd1, [%rd4];\n"
"    st.global.f64 [%rd2+8], %fd1;\n"
"    ld.global.f64 %fd3, [%rd2];\n"
"    div.rn.f64 %fd4, %fd3, %fd1;\n"
"    st.global.f64 [%rd2+16], %fd4;\n"
"L_FPZ_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 4a. Fused x += qk*p, r -= qk*z, and partial (r, r) & max|r| in float32 === */\n"
".visible .entry cg_axpy_r_reduce_f32(\n"
"    .param .u32 param_n,\n"
"    .param .u64 param_state,\n"
"    .param .u64 param_x,\n"
"    .param .u64 param_r,\n"
"    .param .u64 param_p,\n"
"    .param .u64 param_z,\n"
"    .param .u64 param_out_rr,\n"
"    .param .u64 param_out_ram\n"
")\n"
"{\n"
"    .shared .align 8 .b8 s_rr[2048];\n"
"    .shared .align 8 .b8 s_ram[2048];\n"
"    .reg .pred %p<4>;\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<18>;\n"
"    .reg .f32 %f<8>;\n"
"    .reg .f64 %fd<8>;\n"
"\n"
"    ld.param.u32 %r1, [param_n];\n"
"    ld.param.u64 %rd1, [param_state];\n"
"    ld.param.u64 %rd2, [param_x];\n"
"    ld.param.u64 %rd3, [param_r];\n"
"    ld.param.u64 %rd4, [param_p];\n"
"    ld.param.u64 %rd5, [param_z];\n"
"    ld.param.u64 %rd6, [param_out_rr];\n"
"    ld.param.u64 %rd7, [param_out_ram];\n"
"\n"
"    ld.global.f64 %fd1, [%rd1+16];\n"
"    cvt.rn.f32.f64 %f1, %fd1;\n"
"    neg.f32 %f2, %f1;\n"
"\n"
"    mov.u32 %r2, %tid.x;\n"
"    mov.u32 %r3, %ctaid.x;\n"
"    mov.u32 %r4, %ntid.x;\n"
"    mov.u32 %r5, %nctaid.x;\n"
"    mad.lo.s32 %r6, %r3, %r4, %r2;\n"
"    mul.lo.s32 %r7, %r4, %r5;\n"
"\n"
"    mov.f64 %fd2, 0d0000000000000000;\n"
"    mov.f64 %fd3, 0d0000000000000000;\n"
"    setp.ge.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_AXPY32_REDUCE;\n"
"\n"
"L_AXPY32_LOOP:\n"
"    mul.wide.s32 %rd8, %r6, 4;\n"
"    add.u64 %rd9, %rd2, %rd8;\n"
"    add.u64 %rd10, %rd3, %rd8;\n"
"    add.u64 %rd11, %rd4, %rd8;\n"
"    add.u64 %rd12, %rd5, %rd8;\n"
"\n"
"    ld.global.f32 %f3, [%rd9];\n"
"    ld.global.f32 %f4, [%rd10];\n"
"    ld.global.nc.f32 %f5, [%rd11];\n"
"    ld.global.nc.f32 %f6, [%rd12];\n"
"\n"
"    fma.rn.f32 %f3, %f1, %f5, %f3;\n"
"    fma.rn.f32 %f4, %f2, %f6, %f4;\n"
"\n"
"    st.global.f32 [%rd9], %f3;\n"
"    st.global.f32 [%rd10], %f4;\n"
"\n"
"    cvt.f64.f32 %fd4, %f4;\n"
"    fma.rn.f64 %fd2, %fd4, %fd4, %fd2;\n"
"    abs.f64 %fd5, %fd4;\n"
"    max.f64 %fd3, %fd3, %fd5;\n"
"\n"
"    add.s32 %r6, %r6, %r7;\n"
"    setp.lt.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_AXPY32_LOOP;\n"
"\n"
"L_AXPY32_REDUCE:\n"
"    mov.u64 %rd13, s_rr;\n"
"    mov.u64 %rd14, s_ram;\n"
"    mul.wide.u32 %rd15, %r2, 8;\n"
"    add.u64 %rd13, %rd13, %rd15;\n"
"    add.u64 %rd14, %rd14, %rd15;\n"
"    st.shared.f64 [%rd13], %fd2;\n"
"    st.shared.f64 [%rd14], %fd3;\n"
"    bar.sync 0;\n"
"\n"
"    mov.u32 %r8, 128;\n"
"L_AXPY32_TREE:\n"
"    setp.ge.u32 %p2, %r2, %r8;\n"
"    @%p2 bra L_AXPY32_SYNC;\n"
"    mul.wide.u32 %rd16, %r8, 8;\n"
"    add.u64 %rd17, %rd13, %rd16;\n"
"    ld.shared.f64 %fd6, [%rd17];\n"
"    ld.shared.f64 %fd2, [%rd13];\n"
"    add.f64 %fd2, %fd2, %fd6;\n"
"    st.shared.f64 [%rd13], %fd2;\n"
"\n"
"    add.u64 %rd17, %rd14, %rd16;\n"
"    ld.shared.f64 %fd7, [%rd17];\n"
"    ld.shared.f64 %fd3, [%rd14];\n"
"    max.f64 %fd3, %fd3, %fd7;\n"
"    st.shared.f64 [%rd14], %fd3;\n"
"L_AXPY32_SYNC:\n"
"    bar.sync 0;\n"
"    shr.u32 %r8, %r8, 1;\n"
"    setp.gt.u32 %p2, %r8, 0;\n"
"    @%p2 bra L_AXPY32_TREE;\n"
"\n"
"    setp.ne.u32 %p1, %r2, 0;\n"
"    @%p1 bra L_AXPY32_EXIT;\n"
"    mov.u64 %rd13, s_rr;\n"
"    mov.u64 %rd14, s_ram;\n"
"    ld.shared.f64 %fd2, [%rd13];\n"
"    ld.shared.f64 %fd3, [%rd14];\n"
"    mul.wide.u32 %rd15, %r3, 8;\n"
"    add.u64 %rd16, %rd6, %rd15;\n"
"    add.u64 %rd17, %rd7, %rd15;\n"
"    st.global.f64 [%rd16], %fd2;\n"
"    st.global.f64 [%rd17], %fd3;\n"
"L_AXPY32_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 4b. Fused x += qk*p, r -= qk*z, and partial (r, r) & max|r| in float64 === */\n"
".visible .entry cg_axpy_r_reduce_f64(\n"
"    .param .u32 param_n,\n"
"    .param .u64 param_state,\n"
"    .param .u64 param_x,\n"
"    .param .u64 param_r,\n"
"    .param .u64 param_p,\n"
"    .param .u64 param_z,\n"
"    .param .u64 param_out_rr,\n"
"    .param .u64 param_out_ram\n"
")\n"
"{\n"
"    .shared .align 8 .b8 s_rr[2048];\n"
"    .shared .align 8 .b8 s_ram[2048];\n"
"    .reg .pred %p<4>;\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<18>;\n"
"    .reg .f64 %fd<12>;\n"
"\n"
"    ld.param.u32 %r1, [param_n];\n"
"    ld.param.u64 %rd1, [param_state];\n"
"    ld.param.u64 %rd2, [param_x];\n"
"    ld.param.u64 %rd3, [param_r];\n"
"    ld.param.u64 %rd4, [param_p];\n"
"    ld.param.u64 %rd5, [param_z];\n"
"    ld.param.u64 %rd6, [param_out_rr];\n"
"    ld.param.u64 %rd7, [param_out_ram];\n"
"\n"
"    ld.global.f64 %fd1, [%rd1+16];\n"
"    neg.f64 %fd8, %fd1;\n"
"\n"
"    mov.u32 %r2, %tid.x;\n"
"    mov.u32 %r3, %ctaid.x;\n"
"    mov.u32 %r4, %ntid.x;\n"
"    mov.u32 %r5, %nctaid.x;\n"
"    mad.lo.s32 %r6, %r3, %r4, %r2;\n"
"    mul.lo.s32 %r7, %r4, %r5;\n"
"\n"
"    mov.f64 %fd2, 0d0000000000000000;\n"
"    mov.f64 %fd3, 0d0000000000000000;\n"
"    setp.ge.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_AXPY64_REDUCE;\n"
"\n"
"L_AXPY64_LOOP:\n"
"    mul.wide.s32 %rd8, %r6, 8;\n"
"    add.u64 %rd9, %rd2, %rd8;\n"
"    add.u64 %rd10, %rd3, %rd8;\n"
"    add.u64 %rd11, %rd4, %rd8;\n"
"    add.u64 %rd12, %rd5, %rd8;\n"
"\n"
"    ld.global.f64 %fd9, [%rd9];\n"
"    ld.global.f64 %fd4, [%rd10];\n"
"    ld.global.nc.f64 %fd10, [%rd11];\n"
"    ld.global.nc.f64 %fd11, [%rd12];\n"
"\n"
"    fma.rn.f64 %fd9, %fd1, %fd10, %fd9;\n"
"    fma.rn.f64 %fd4, %fd8, %fd11, %fd4;\n"
"\n"
"    st.global.f64 [%rd9], %fd9;\n"
"    st.global.f64 [%rd10], %fd4;\n"
"\n"
"    fma.rn.f64 %fd2, %fd4, %fd4, %fd2;\n"
"    abs.f64 %fd5, %fd4;\n"
"    max.f64 %fd3, %fd3, %fd5;\n"
"\n"
"    add.s32 %r6, %r6, %r7;\n"
"    setp.lt.s32 %p1, %r6, %r1;\n"
"    @%p1 bra L_AXPY64_LOOP;\n"
"\n"
"L_AXPY64_REDUCE:\n"
"    mov.u64 %rd13, s_rr;\n"
"    mov.u64 %rd14, s_ram;\n"
"    mul.wide.u32 %rd15, %r2, 8;\n"
"    add.u64 %rd13, %rd13, %rd15;\n"
"    add.u64 %rd14, %rd14, %rd15;\n"
"    st.shared.f64 [%rd13], %fd2;\n"
"    st.shared.f64 [%rd14], %fd3;\n"
"    bar.sync 0;\n"
"\n"
"    mov.u32 %r8, 128;\n"
"L_AXPY64_TREE:\n"
"    setp.ge.u32 %p2, %r2, %r8;\n"
"    @%p2 bra L_AXPY64_SYNC;\n"
"    mul.wide.u32 %rd16, %r8, 8;\n"
"    add.u64 %rd17, %rd13, %rd16;\n"
"    ld.shared.f64 %fd6, [%rd17];\n"
"    ld.shared.f64 %fd2, [%rd13];\n"
"    add.f64 %fd2, %fd2, %fd6;\n"
"    st.shared.f64 [%rd13], %fd2;\n"
"\n"
"    add.u64 %rd17, %rd14, %rd16;\n"
"    ld.shared.f64 %fd7, [%rd17];\n"
"    ld.shared.f64 %fd3, [%rd14];\n"
"    max.f64 %fd3, %fd3, %fd7;\n"
"    st.shared.f64 [%rd14], %fd3;\n"
"L_AXPY64_SYNC:\n"
"    bar.sync 0;\n"
"    shr.u32 %r8, %r8, 1;\n"
"    setp.gt.u32 %p2, %r8, 0;\n"
"    @%p2 bra L_AXPY64_TREE;\n"
"\n"
"    setp.ne.u32 %p1, %r2, 0;\n"
"    @%p1 bra L_AXPY64_EXIT;\n"
"    mov.u64 %rd13, s_rr;\n"
"    mov.u64 %rd14, s_ram;\n"
"    ld.shared.f64 %fd2, [%rd13];\n"
"    ld.shared.f64 %fd3, [%rd14];\n"
"    mul.wide.u32 %rd15, %r3, 8;\n"
"    add.u64 %rd16, %rd6, %rd15;\n"
"    add.u64 %rd17, %rd7, %rd15;\n"
"    st.global.f64 [%rd16], %fd2;\n"
"    st.global.f64 [%rd17], %fd3;\n"
"L_AXPY64_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 5. Single-block final reduction of rr_new and ram, computing ekm1 = rr_new / rr === */\n"
".visible .entry finish_axpy_reduce(\n"
"    .param .u32 param_nblocks,\n"
"    .param .u64 param_buf_rr,\n"
"    .param .u64 param_buf_ram,\n"
"    .param .u64 param_state\n"
")\n"
"{\n"
"    .shared .align 8 .b8 s_rr[512];\n"
"    .shared .align 8 .b8 s_ram[512];\n"
"    .reg .pred %p<3>;\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<12>;\n"
"    .reg .f64 %fd<7>;\n"
"\n"
"    ld.param.u32 %r1, [param_nblocks];\n"
"    ld.param.u64 %rd1, [param_buf_rr];\n"
"    ld.param.u64 %rd2, [param_buf_ram];\n"
"    ld.param.u64 %rd3, [param_state];\n"
"\n"
"    mov.u32 %r2, %tid.x;\n"
"    mov.f64 %fd1, 0d0000000000000000;\n"
"    mov.f64 %fd2, 0d0000000000000000;\n"
"    setp.ge.u32 %p1, %r2, %r1;\n"
"    @%p1 bra L_FRR_STORE;\n"
"    mul.wide.u32 %rd4, %r2, 8;\n"
"    add.u64 %rd5, %rd1, %rd4;\n"
"    add.u64 %rd6, %rd2, %rd4;\n"
"    ld.global.f64 %fd1, [%rd5];\n"
"    ld.global.f64 %fd2, [%rd6];\n"
"L_FRR_STORE:\n"
"    mov.u64 %rd7, s_rr;\n"
"    mov.u64 %rd8, s_ram;\n"
"    mul.wide.u32 %rd4, %r2, 8;\n"
"    add.u64 %rd7, %rd7, %rd4;\n"
"    add.u64 %rd8, %rd8, %rd4;\n"
"    st.shared.f64 [%rd7], %fd1;\n"
"    st.shared.f64 [%rd8], %fd2;\n"
"    bar.sync 0;\n"
"\n"
"    mov.u32 %r3, 32;\n"
"L_FRR_TREE:\n"
"    setp.ge.u32 %p2, %r2, %r3;\n"
"    @%p2 bra L_FRR_SYNC;\n"
"    mul.wide.u32 %rd9, %r3, 8;\n"
"    add.u64 %rd10, %rd7, %rd9;\n"
"    ld.shared.f64 %fd3, [%rd10];\n"
"    ld.shared.f64 %fd1, [%rd7];\n"
"    add.f64 %fd1, %fd1, %fd3;\n"
"    st.shared.f64 [%rd7], %fd1;\n"
"\n"
"    add.u64 %rd10, %rd8, %rd9;\n"
"    ld.shared.f64 %fd4, [%rd10];\n"
"    ld.shared.f64 %fd2, [%rd8];\n"
"    max.f64 %fd2, %fd2, %fd4;\n"
"    st.shared.f64 [%rd8], %fd2;\n"
"L_FRR_SYNC:\n"
"    bar.sync 0;\n"
"    shr.u32 %r3, %r3, 1;\n"
"    setp.gt.u32 %p2, %r3, 0;\n"
"    @%p2 bra L_FRR_TREE;\n"
"\n"
"    setp.ne.u32 %p1, %r2, 0;\n"
"    @%p1 bra L_FRR_EXIT;\n"
"    mov.u64 %rd7, s_rr;\n"
"    mov.u64 %rd8, s_ram;\n"
"    ld.shared.f64 %fd1, [%rd7];\n"
"    ld.shared.f64 %fd2, [%rd8];\n"
"    ld.global.f64 %fd5, [%rd3];\n"
"    div.rn.f64 %fd6, %fd1, %fd5;\n"
"    st.global.f64 [%rd3], %fd1;\n"
"    st.global.f64 [%rd3+24], %fd6;\n"
"    st.global.f64 [%rd3+32], %fd2;\n"
"L_FRR_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 6a. Vector update p = r + ekm1 * p in float32 === */\n"
".visible .entry cg_xpay_f32(\n"
"    .param .u32 param_n,\n"
"    .param .u64 param_state,\n"
"    .param .u64 param_r,\n"
"    .param .u64 param_p\n"
")\n"
"{\n"
"    .reg .pred %p1;\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<7>;\n"
"    .reg .f32 %f<4>;\n"
"    .reg .f64 %fd1;\n"
"\n"
"    ld.param.u32 %r1, [param_n];\n"
"    mov.u32 %r2, %ctaid.x;\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %tid.x;\n"
"    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
"    setp.ge.s32 %p1, %r5, %r1;\n"
"    @%p1 bra L_XPAY32_EXIT;\n"
"\n"
"    ld.param.u64 %rd1, [param_state];\n"
"    ld.param.u64 %rd2, [param_r];\n"
"    ld.param.u64 %rd3, [param_p];\n"
"\n"
"    ld.global.f64 %fd1, [%rd1+24];\n"
"    cvt.rn.f32.f64 %f1, %fd1;\n"
"    mul.wide.s32 %rd4, %r5, 4;\n"
"    add.u64 %rd5, %rd2, %rd4;\n"
"    add.u64 %rd6, %rd3, %rd4;\n"
"    ld.global.nc.f32 %f2, [%rd5];\n"
"    ld.global.f32 %f3, [%rd6];\n"
"    fma.rn.f32 %f3, %f1, %f3, %f2;\n"
"    st.global.f32 [%rd6], %f3;\n"
"L_XPAY32_EXIT:\n"
"    ret;\n"
"}\n"
"\n"
"/* === 6b. Vector update p = r + ekm1 * p in float64 === */\n"
".visible .entry cg_xpay_f64(\n"
"    .param .u32 param_n,\n"
"    .param .u64 param_state,\n"
"    .param .u64 param_r,\n"
"    .param .u64 param_p\n"
")\n"
"{\n"
"    .reg .pred %p1;\n"
"    .reg .b32 %r<6>;\n"
"    .reg .b64 %rd<7>;\n"
"    .reg .f64 %fd<4>;\n"
"\n"
"    ld.param.u32 %r1, [param_n];\n"
"    mov.u32 %r2, %ctaid.x;\n"
"    mov.u32 %r3, %ntid.x;\n"
"    mov.u32 %r4, %tid.x;\n"
"    mad.lo.s32 %r5, %r2, %r3, %r4;\n"
"    setp.ge.s32 %p1, %r5, %r1;\n"
"    @%p1 bra L_XPAY64_EXIT;\n"
"\n"
"    ld.param.u64 %rd1, [param_state];\n"
"    ld.param.u64 %rd2, [param_r];\n"
"    ld.param.u64 %rd3, [param_p];\n"
"\n"
"    ld.global.f64 %fd1, [%rd1+24];\n"
"    mul.wide.s32 %rd4, %r5, 8;\n"
"    add.u64 %rd5, %rd2, %rd4;\n"
"    add.u64 %rd6, %rd3, %rd4;\n"
"    ld.global.nc.f64 %fd2, [%rd5];\n"
"    ld.global.f64 %fd3, [%rd6];\n"
"    fma.rn.f64 %fd3, %fd1, %fd3, %fd2;\n"
"    st.global.f64 [%rd6], %fd3;\n"
"L_XPAY64_EXIT:\n"
"    ret;\n"
"}\n";

static int init_cuda_driver(void) {
  if (cuda_init_state != 0) return (cuda_init_state == 1);

  const char *disable_gpu = getenv("CCX_NO_GPU");
  if (disable_gpu && atoi(disable_gpu) != 0) {
    cuda_init_state = -1;
    return 0;
  }

#ifdef _WIN32
  HMODULE hCuda = LoadLibraryA("nvcuda.dll");
  if (!hCuda) {
    cuda_init_state = -1;
    return 0;
  }

  f_cuInit = (pfn_cuInit)GetProcAddress(hCuda, "cuInit");
  f_cuDeviceGet = (pfn_cuDeviceGet)GetProcAddress(hCuda, "cuDeviceGet");
  f_cuDeviceGetName = (pfn_cuDeviceGetName)GetProcAddress(hCuda, "cuDeviceGetName");
  f_cuCtxCreate = (pfn_cuCtxCreate)GetProcAddress(hCuda, "cuCtxCreate_v2");
  f_cuCtxSetCurrent = (pfn_cuCtxSetCurrent)GetProcAddress(hCuda, "cuCtxSetCurrent");
  f_cuModuleLoadDataEx = (pfn_cuModuleLoadDataEx)GetProcAddress(hCuda, "cuModuleLoadDataEx");
  f_cuModuleGetFunction = (pfn_cuModuleGetFunction)GetProcAddress(hCuda, "cuModuleGetFunction");
  f_cuMemAlloc = (pfn_cuMemAlloc)GetProcAddress(hCuda, "cuMemAlloc_v2");
  f_cuMemFree = (pfn_cuMemFree)GetProcAddress(hCuda, "cuMemFree_v2");
  f_cuMemcpyHtoD = (pfn_cuMemcpyHtoD)GetProcAddress(hCuda, "cuMemcpyHtoD_v2");
  f_cuMemcpyDtoH = (pfn_cuMemcpyDtoH)GetProcAddress(hCuda, "cuMemcpyDtoH_v2");
  f_cuMemsetD32 = (pfn_cuMemsetD32)GetProcAddress(hCuda, "cuMemsetD32_v2");
  f_cuLaunchKernel = (pfn_cuLaunchKernel)GetProcAddress(hCuda, "cuLaunchKernel");
  f_cuCtxSynchronize = (pfn_cuCtxSynchronize)GetProcAddress(hCuda, "cuCtxSynchronize");

  if (!f_cuInit || !f_cuDeviceGet || !f_cuCtxCreate || !f_cuModuleLoadDataEx ||
      !f_cuModuleGetFunction || !f_cuMemAlloc || !f_cuMemFree ||
      !f_cuMemcpyHtoD || !f_cuMemcpyDtoH || !f_cuLaunchKernel || !f_cuCtxSynchronize) {
    cuda_init_state = -1;
    return 0;
  }

  if (f_cuInit(0) != CUDA_SUCCESS) {
    cuda_init_state = -1;
    return 0;
  }
  CUdevice dev = 0;
  if (f_cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
    cuda_init_state = -1;
    return 0;
  }
  if (f_cuDeviceGetName) {
    f_cuDeviceGetName(cuda_dev_name, sizeof(cuda_dev_name), dev);
  }
  if (f_cuCtxCreate(&cuda_ctx, 0, dev) != CUDA_SUCCESS) {
    cuda_init_state = -1;
    return 0;
  }

  char jit_log[4096];
  jit_log[0] = '\0';
  int jit_opts[2] = { 3 /* CU_JIT_ERROR_LOG_BUFFER */, 4 /* CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES */ };
  void *jit_vals[2] = { (void*)jit_log, (void*)(size_t)sizeof(jit_log) };
  CUresult res = f_cuModuleLoadDataEx(&cuda_mod, cuda_ptx_src, 2, jit_opts, jit_vals);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "[GPU] cuModuleLoadDataEx failed (%d): %s\n", res, jit_log);
    cuda_init_state = -1;
    return 0;
  }

  if (f_cuModuleGetFunction(&fn_spmv_f32, cuda_mod, "csr_spmv_subwarp8_f32") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_dot_f32, cuda_mod, "vec_dot_f32") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_finish_pz, cuda_mod, "finish_dot_pz") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_axpy_r_reduce_f32, cuda_mod, "cg_axpy_r_reduce_f32") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_finish_rr, cuda_mod, "finish_axpy_reduce") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_xpay_f32, cuda_mod, "cg_xpay_f32") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_spmv_f64, cuda_mod, "csr_spmv_subwarp8_f64") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_dot_f64, cuda_mod, "vec_dot_f64") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_axpy_r_reduce_f64, cuda_mod, "cg_axpy_r_reduce_f64") != CUDA_SUCCESS ||
      f_cuModuleGetFunction(&fn_xpay_f64, cuda_mod, "cg_xpay_f64") != CUDA_SUCCESS) {
    fprintf(stderr, "[GPU] cuModuleGetFunction failed\n");
    cuda_init_state = -1;
    return 0;
  }
  cuda_init_state = 1;
  return 1;
#else
  cuda_init_state = -1;
  return 0;
#endif
}

/* Lock-free parallel O(nzs) builder for the full symmetric diagonally-scaled CSR matrix A_scaled = D * A * D.
   Uses per-thread column range buckets so every row's column indices are 100% sorted with zero atomic contention! */
static void build_scaled_full_csr(const double *ad, const double *au, const ITG *icol, const ITG *irow,
                                  ITG neq, ITG nzs, int nthreads,
                                  double *d_scale, int *row_ptr, int *col_idx,
                                  double *val_f64, float *val_f32) {
  #pragma omp parallel for schedule(static) num_threads(nthreads)
  for (ITG i = 0; i < neq; i++) {
    double aii = fabs(ad[i]);
    d_scale[i] = (aii > 0.0) ? (1.0 / sqrt(aii)) : 1.0;
  }

  ITG *col_offset = NULL;
  int *thread_counts = NULL;
  NNEW(col_offset, ITG, neq + 1);
  NNEW(thread_counts, int, (size_t)nthreads * (size_t)neq);

  col_offset[0] = 0;
  for (ITG i = 0; i < neq; i++) {
    col_offset[i + 1] = col_offset[i] + icol[i];
  }

  /* Pass 1: Each thread t counts lower-triangular entries in its contiguous column chunk [c_start, c_end) */
  #pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    ITG c_start = ((ITG)tid * neq) / nthreads;
    ITG c_end = ((ITG)(tid + 1) * neq) / nthreads;
    int *my_counts = &thread_counts[(size_t)tid * (size_t)neq];
    ITG k_start = col_offset[c_start];
    ITG k_end = col_offset[c_end];
    for (ITG k = k_start; k < k_end; k++) {
      my_counts[irow[k] - 1]++;
    }
  }

  /* Compute row lengths and per-thread write cursors in parallel */
  #pragma omp parallel for schedule(static) num_threads(nthreads)
  for (ITG r = 0; r < neq; r++) {
    int sum = 0;
    for (int t = 0; t < nthreads; t++) {
      int c = thread_counts[(size_t)t * (size_t)neq + r];
      thread_counts[(size_t)t * (size_t)neq + r] = sum;
      sum += c;
    }
    row_ptr[r + 1] = sum + 1 + (int)icol[r];
  }

  row_ptr[0] = 0;
  for (ITG r = 0; r < neq; r++) {
    row_ptr[r + 1] += row_ptr[r];
  }

  /* Pass 2: Each thread scatters its sorted column chunk into the lower triangle AND fills diagonal + upper triangle */
  #pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    ITG c_start = ((ITG)tid * neq) / nthreads;
    ITG c_end = ((ITG)(tid + 1) * neq) / nthreads;
    int *my_offsets = &thread_counts[(size_t)tid * (size_t)neq];

    for (ITG c = c_start; c < c_end; c++) {
      int u_pos = (row_ptr[c + 1] - (int)icol[c]) - 1;
      double diag_v = (ad[c] >= 0.0) ? 1.0 : -1.0;
      col_idx[u_pos] = (int)c;
      if (val_f64) val_f64[u_pos] = diag_v;
      if (val_f32) val_f32[u_pos] = (float)diag_v;
      u_pos++;

      ITG kstart = col_offset[c];
      ITG kend = col_offset[c + 1];
      double dc = d_scale[c];
      for (ITG k = kstart; k < kend; k++) {
        ITG r = irow[k] - 1;
        double v = au[k] * dc * d_scale[r];
        col_idx[u_pos] = (int)r;
        if (val_f64) val_f64[u_pos] = v;
        if (val_f32) val_f32[u_pos] = (float)v;
        u_pos++;

        int l_pos = row_ptr[r] + (my_offsets[r]++);
        col_idx[l_pos] = (int)c;
        if (val_f64) val_f64[l_pos] = v;
        if (val_f32) val_f32[l_pos] = (float)v;
      }
    }
  }

  SFREE(col_offset);
  SFREE(thread_counts);
}

static CUdeviceptr g_d_row_ptr = 0, g_d_col_idx = 0, g_d_val = 0;
static CUdeviceptr g_d_x = 0, g_d_r = 0, g_d_p = 0, g_d_z = 0, g_d_red_buf = 0, g_d_state = 0;
static size_t g_cap_n_bytes = 0, g_cap_val_bytes = 0;
static int g_cap_n = 0, g_cap_nnz = 0;

/* Solve on GPU in either pure float32 (use_f64=0) or full float64 (use_f64=1) */
static int solve_cg_gpu(int n, int nnz, const int *row_ptr, const int *col_idx,
                        const float *val_f32, const double *val_f64, int use_f64,
                        const double *b_scaled, double *x_scaled,
                        double tol_limit, ITG max_iter, ITG *out_iter,
                        double *out_rr, int nthreads, double *t_h2d_ms, double *t_kern_ms) {
  double t0 = ccx_prof_ms();
  f_cuCtxSetCurrent(cuda_ctx);

  int num_red_blocks = (n + 255) / 256;
  if (num_red_blocks > 64) num_red_blocks = 64;
  if (num_red_blocks < 1) num_red_blocks = 1;

  size_t elem_sz = use_f64 ? sizeof(double) : sizeof(float);
  size_t vec_bytes = (size_t)n * elem_sz;
  size_t val_bytes = (size_t)nnz * elem_sz;
  size_t max_red_bytes = 64 * 2 * sizeof(double);

  if (n > g_cap_n || vec_bytes > g_cap_n_bytes) {
    if (g_d_row_ptr) f_cuMemFree(g_d_row_ptr);
    if (g_d_x) f_cuMemFree(g_d_x);
    if (g_d_r) f_cuMemFree(g_d_r);
    if (g_d_p) f_cuMemFree(g_d_p);
    if (g_d_z) f_cuMemFree(g_d_z);
    if (!g_d_red_buf) f_cuMemAlloc(&g_d_red_buf, max_red_bytes);
    if (!g_d_state) f_cuMemAlloc(&g_d_state, sizeof(CgGpuState));
    if (f_cuMemAlloc(&g_d_row_ptr, (size_t)(n + 1) * sizeof(int)) != CUDA_SUCCESS ||
        f_cuMemAlloc(&g_d_x, vec_bytes) != CUDA_SUCCESS ||
        f_cuMemAlloc(&g_d_r, vec_bytes) != CUDA_SUCCESS ||
        f_cuMemAlloc(&g_d_p, vec_bytes) != CUDA_SUCCESS ||
        f_cuMemAlloc(&g_d_z, vec_bytes) != CUDA_SUCCESS) {
      g_cap_n = 0;
      g_cap_n_bytes = 0;
      return -1;
    }
    g_cap_n = n;
    g_cap_n_bytes = vec_bytes;
  }

  if (nnz > g_cap_nnz || val_bytes > g_cap_val_bytes) {
    if (g_d_col_idx) f_cuMemFree(g_d_col_idx);
    if (g_d_val) f_cuMemFree(g_d_val);
    if (f_cuMemAlloc(&g_d_col_idx, (size_t)nnz * sizeof(int)) != CUDA_SUCCESS ||
        f_cuMemAlloc(&g_d_val, val_bytes) != CUDA_SUCCESS) {
      g_cap_nnz = 0;
      g_cap_val_bytes = 0;
      return -1;
    }
    g_cap_nnz = nnz;
    g_cap_val_bytes = val_bytes;
  }

  CUdeviceptr d_row_ptr = g_d_row_ptr, d_col_idx = g_d_col_idx, d_val = g_d_val;
  CUdeviceptr d_x = g_d_x, d_r = g_d_r, d_p = g_d_p, d_z = g_d_z;
  CUdeviceptr d_red_rr = g_d_red_buf;
  CUdeviceptr d_red_ram = g_d_red_buf + (size_t)num_red_blocks * sizeof(double);
  CUdeviceptr d_state = g_d_state;

  f_cuMemcpyHtoD(d_row_ptr, row_ptr, (size_t)(n + 1) * sizeof(int));
  f_cuMemcpyHtoD(d_col_idx, col_idx, (size_t)nnz * sizeof(int));
  if (use_f64) {
    f_cuMemcpyHtoD(d_val, val_f64, val_bytes);
  } else {
    f_cuMemcpyHtoD(d_val, val_f32, val_bytes);
  }

  double rr = 0.0;
  float *h_vec_f32 = NULL;
  if (!use_f64) {
    NNEW(h_vec_f32, float, n);
  }

  #pragma omp parallel for reduction(+:rr) schedule(static) num_threads(nthreads)
  for (int i = 0; i < n; i++) {
    x_scaled[i] = 0.0;
    double bi = b_scaled[i];
    if (!use_f64) h_vec_f32[i] = (float)bi;
    rr += bi * bi;
  }

  CgGpuState h_state;
  memset(&h_state, 0, sizeof(h_state));
  h_state.rr = rr;
  h_state.ram = 1e30;

  f_cuMemsetD32(d_x, 0, use_f64 ? ((size_t)n * 2) : (size_t)n);
  if (use_f64) {
    f_cuMemcpyHtoD(d_r, b_scaled, vec_bytes);
    f_cuMemcpyHtoD(d_p, b_scaled, vec_bytes);
  } else {
    f_cuMemcpyHtoD(d_r, h_vec_f32, vec_bytes);
    f_cuMemcpyHtoD(d_p, h_vec_f32, vec_bytes);
  }
  f_cuMemcpyHtoD(d_state, &h_state, sizeof(h_state));
  double t1 = ccx_prof_ms();
  *t_h2d_ms = t1 - t0;

  unsigned int spmv_threads = 256;
  unsigned int spmv_blocks = ((unsigned int)n * 8 + spmv_threads - 1) / spmv_threads;
  unsigned int vec_threads = 256;
  unsigned int vec_blocks = ((unsigned int)n + vec_threads - 1) / vec_threads;

  CUfunction k_spmv = use_f64 ? fn_spmv_f64 : fn_spmv_f32;
  CUfunction k_dot  = use_f64 ? fn_dot_f64  : fn_dot_f32;
  CUfunction k_fpz  = fn_finish_pz;
  CUfunction k_axpy = use_f64 ? fn_axpy_r_reduce_f64 : fn_axpy_r_reduce_f32;
  CUfunction k_frr  = fn_finish_rr;
  CUfunction k_xpay = use_f64 ? fn_xpay_f64 : fn_xpay_f32;

  void *spmv_args[] = { &n, &d_row_ptr, &d_col_idx, &d_val, &d_p, &d_z };
  void *dot_args[] = { &n, &d_p, &d_z, &d_red_rr };
  void *fpz_args[] = { &num_red_blocks, &d_red_rr, &d_state };
  void *axpy_args[] = { &n, &d_state, &d_x, &d_r, &d_p, &d_z, &d_red_rr, &d_red_ram };
  void *frr_args[] = { &num_red_blocks, &d_red_rr, &d_red_ram, &d_state };
  void *xpay_args[] = { &n, &d_state, &d_r, &d_p };

  ITG total_iters = 0;
  const ITG batch_size = 16;

  for (ITG k = 1; k <= max_iter; ) {
    ITG batch_end = k + batch_size - 1;
    if (batch_end > max_iter) batch_end = max_iter;

    for (ITG step = k; step <= batch_end; step++) {
      f_cuLaunchKernel(k_spmv, spmv_blocks, 1, 1, spmv_threads, 1, 1, 0, NULL, spmv_args, NULL);
      f_cuLaunchKernel(k_dot, (unsigned int)num_red_blocks, 1, 1, 256, 1, 1, 0, NULL, dot_args, NULL);
      f_cuLaunchKernel(k_fpz, 1, 1, 1, 64, 1, 1, 0, NULL, fpz_args, NULL);
      f_cuLaunchKernel(k_axpy, (unsigned int)num_red_blocks, 1, 1, 256, 1, 1, 0, NULL, axpy_args, NULL);
      f_cuLaunchKernel(k_frr, 1, 1, 1, 64, 1, 1, 0, NULL, frr_args, NULL);
      f_cuLaunchKernel(k_xpay, vec_blocks, 1, 1, vec_threads, 1, 1, 0, NULL, xpay_args, NULL);
    }

    total_iters = batch_end;
    f_cuMemcpyDtoH(&h_state, d_state, sizeof(h_state));
    if (h_state.ram <= tol_limit || h_state.pz == 0.0) {
      break;
    }
    k = batch_end + 1;
  }

  if (use_f64) {
    f_cuMemcpyDtoH(x_scaled, d_x, vec_bytes);
  } else {
    f_cuMemcpyDtoH(h_vec_f32, d_x, vec_bytes);
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; i++) {
      x_scaled[i] = (double)h_vec_f32[i];
    }
    SFREE(h_vec_f32);
  }

  double t2 = ccx_prof_ms();
  *t_kern_ms = t2 - t1;
  *out_iter = total_iters;
  *out_rr = h_state.rr;
  return 0;
}

/* Lock-free atomic sense-reversing spin-barrier to avoid winpthreads OS semaphore overhead in CPU CG */
typedef struct {
  volatile int count;
  volatile int sense;
} SpinBarrier;

static inline void spin_barrier_wait(SpinBarrier *b, int nthreads, int *local_sense) {
  int s = !(*local_sense);
  *local_sense = s;
  if (__atomic_add_fetch(&b->count, 1, __ATOMIC_ACQ_REL) == nthreads) {
    __atomic_store_n(&b->count, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&b->sense, s, __ATOMIC_RELEASE);
  } else {
    int spins = 0;
    while (__atomic_load_n(&b->sense, __ATOMIC_ACQUIRE) != s) {
#ifdef _WIN32
      if (++spins < 2000) {
        _mm_pause();
      } else {
        SwitchToThread();
        spins = 0;
      }
#endif
    }
  }
}

/* Load-balanced Multi-threaded OpenMP Full-CSR CG Solver (CPU fallback when CCX_NO_GPU=1 or no GPU) */
static int solve_cg_cpu_omp(int n, int nnz, const int *row_ptr, const int *col_idx,
                            const double *val_f64, const double *b_scaled, double *x_scaled,
                            double tol_limit, ITG max_iter, ITG *out_iter,
                            double *out_rr, int nthreads) {
  double *r = NULL, *p = NULL, *z = NULL;
  int *t_row_start = NULL;
  double *t_red = NULL; /* 16 doubles (128 bytes) per thread to avoid false sharing */
  NNEW(r, double, n);
  NNEW(p, double, n);
  NNEW(z, double, n);
  NNEW(t_row_start, int, nthreads + 1);
  NNEW(t_red, double, (size_t)nthreads * 16);

  /* Partition rows so each thread processes an equal number of nonzeros (nnz / nthreads) */
  t_row_start[0] = 0;
  t_row_start[nthreads] = n;
  for (int t = 1; t < nthreads; t++) {
    int target_nnz = (int)(((long long)t * nnz) / nthreads);
    int lo = 0, hi = n;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (row_ptr[mid] < target_nnz) lo = mid + 1;
      else hi = mid;
    }
    t_row_start[t] = lo;
  }

  double rr_init = 0.0;
  #pragma omp parallel for reduction(+:rr_init) schedule(static) num_threads(nthreads)
  for (int i = 0; i < n; i++) {
    x_scaled[i] = 0.0;
    double bi = b_scaled[i];
    r[i] = bi;
    p[i] = bi;
    rr_init += bi * bi;
  }

  ITG total_iters = 0;
  double final_rr = rr_init;
  SpinBarrier bar = { 0, 0 };

  #pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int local_sense = 0;
    int rb = t_row_start[tid];
    int re = t_row_start[tid + 1];
    int vb = (int)(((long long)tid * n) / nthreads);
    int ve = (int)(((long long)(tid + 1) * n) / nthreads);
    double rr = rr_init;

    for (ITG k = 1; k <= max_iter; k++) {
      /* 1. SpMV z = A * p and partial dot product (p, z) */
      double local_pz = 0.0;
      for (int i = rb; i < re; i++) {
        int kstart = row_ptr[i];
        int kend = row_ptr[i + 1];
        double sum = 0.0;
        for (int jj = kstart; jj < kend; jj++) {
          sum += val_f64[jj] * p[col_idx[jj]];
        }
        z[i] = sum;
        local_pz += p[i] * sum;
      }
      t_red[tid * 16 + 0] = local_pz;

      spin_barrier_wait(&bar, nthreads, &local_sense);

      double pz = 0.0;
      for (int t = 0; t < nthreads; t++) {
        pz += t_red[t * 16 + 0];
      }
      if (pz == 0.0) {
        if (tid == 0) {
          total_iters = k;
          final_rr = rr;
        }
        break;
      }
      double qk = rr / pz;

      /* 2. Update x and r, accumulate (r, r) and max |r| */
      double local_rr = 0.0;
      double local_ram = 0.0;
      for (int i = vb; i < ve; i++) {
        x_scaled[i] += qk * p[i];
        double ri = r[i] - qk * z[i];
        r[i] = ri;
        local_rr += ri * ri;
        double ari = fabs(ri);
        if (ari > local_ram) local_ram = ari;
      }
      t_red[tid * 16 + 1] = local_rr;
      t_red[tid * 16 + 2] = local_ram;

      spin_barrier_wait(&bar, nthreads, &local_sense);

      double rr_new = 0.0;
      double ram = 0.0;
      for (int t = 0; t < nthreads; t++) {
        rr_new += t_red[t * 16 + 1];
        double tr = t_red[t * 16 + 2];
        if (tr > ram) ram = tr;
      }

      if (ram <= tol_limit) {
        if (tid == 0) {
          total_iters = k;
          final_rr = rr_new;
        }
        break;
      }

      double ekm1 = rr_new / rr;
      rr = rr_new;

      /* 3. Update search direction p */
      for (int i = vb; i < ve; i++) {
        p[i] = r[i] + ekm1 * p[i];
      }

      spin_barrier_wait(&bar, nthreads, &local_sense);
    }
  }

  SFREE(t_red);
  SFREE(t_row_start);
  SFREE(r);
  SFREE(p);
  SFREE(z);
  *out_iter = total_iters;
  *out_rr = final_rr;
  return 0;
}

ITG ccx_fast_cg_solve(const double *ad, const double *au, double *b,
                      const ITG *icol, const ITG *irow, ITG neq, ITG nzs,
                      double *eps, ITG *niter) {
  int nthreads = omp_get_max_threads();
  const char *env_omp = getenv("OMP_NUM_THREADS");
  if (env_omp && atoi(env_omp) > 0) nthreads = atoi(env_omp);
  if (nthreads < 1) nthreads = 1;

  int use_gpu = init_cuda_driver();
  double t0 = ccx_prof_ms();
  int n = (int)neq;
  int nnz_full = (int)(neq + 2 * nzs);

  double *d_scale = NULL;
  double *b_scaled = NULL;
  double *x_scaled = NULL;
  NNEW(d_scale, double, neq);
  NNEW(b_scaled, double, neq);
  NNEW(x_scaled, double, neq);

  double qam = 0.0;
  ITG iam = 0;
  #pragma omp parallel for reduction(+:qam,iam) schedule(static) num_threads(nthreads)
  for (ITG i = 0; i < neq; i++) {
    double aii = fabs(ad[i]);
    double ds = (aii > 0.0) ? (1.0 / sqrt(aii)) : 1.0;
    d_scale[i] = ds;
    double bs = b[i] * ds;
    b_scaled[i] = bs;
    double abs_bs = fabs(bs);
    if (abs_bs > 1.e-20) {
      qam += abs_bs;
      iam++;
    }
  }

  if (iam > 0) {
    qam = qam / (double)neq;
  } else {
    *niter = 0;
    memset(b, 0, (size_t)neq * sizeof(double));
    SFREE(d_scale);
    SFREE(b_scaled);
    SFREE(x_scaled);
    return 0;
  }

  double tol_limit = 0.005 * qam;
  /* Select float32 on GPU for moderate-sized systems (neq <= 100000) or when CCX_GPU_FP32=1 is set;
     use float64 on GPU for large ill-conditioned 3D systems (neq > 100000) to avoid Krylov/eigenvalue stagnation */
  int gpu_use_f64 = (neq > 100000) ? 1 : 0;
  const char *env_gpu_fp32 = getenv("CCX_GPU_FP32");
  if (env_gpu_fp32) {
    gpu_use_f64 = (atoi(env_gpu_fp32) != 0) ? 0 : 1;
  }

  int *row_ptr = NULL;
  int *col_idx = NULL;
  float *val_f32 = NULL;
  double *val_f64 = NULL;

  NNEW(row_ptr, int, neq + 1);
  NNEW(col_idx, int, nnz_full);
  if (use_gpu && !gpu_use_f64) {
    NNEW(val_f32, float, nnz_full);
  } else {
    NNEW(val_f64, double, nnz_full);
  }

  build_scaled_full_csr(ad, au, icol, irow, neq, nzs, nthreads,
                        d_scale, row_ptr, col_idx, val_f64, val_f32);

  double t_csr = ccx_prof_ms();
  double t_h2d = 0.0, t_kern = 0.0;
  ITG iters_done = 0;
  double rr_final = 0.0;
  int gpu_ok = -1;

  if (use_gpu) {
    printf("Solving the system of equations using the GPU %s CSR iterative solver (%s)\n\n",
           gpu_use_f64 ? "float64" : "float32", cuda_dev_name);
    gpu_ok = solve_cg_gpu(n, nnz_full, row_ptr, col_idx, val_f32, val_f64, gpu_use_f64,
                          b_scaled, x_scaled, tol_limit, *niter, &iters_done, &rr_final,
                          nthreads, &t_h2d, &t_kern);
  }

  if (gpu_ok != 0) {
    if (!val_f64) {
      NNEW(val_f64, double, nnz_full);
      build_scaled_full_csr(ad, au, icol, irow, neq, nzs, nthreads,
                            d_scale, row_ptr, col_idx, val_f64, NULL);
    }
    printf("Solving the system of equations using the OpenMP CSR iterative solver (%d threads)\n\n", nthreads);
    solve_cg_cpu_omp(n, nnz_full, row_ptr, col_idx, val_f64, b_scaled, x_scaled,
                     tol_limit, *niter, &iters_done, &rr_final, nthreads);
  }

  #pragma omp parallel for schedule(static) num_threads(nthreads)
  for (ITG i = 0; i < neq; i++) {
    b[i] = x_scaled[i] * d_scale[i];
  }

  double t_end = ccx_prof_ms();
  CCX_PROF_LOG("fast_cg_solve: total=%.2f ms (csr_build=%.2f ms, gpu_h2d=%.2f ms, cg_loop=%.2f ms) [gpu=%d, f64=%d, nt=%d, niter=%d, neq=%d, nnz=%d]",
               t_end - t0, t_csr - t0, t_h2d, (gpu_ok == 0) ? t_kern : (t_end - t_csr),
               (gpu_ok == 0) ? 1 : 0, gpu_use_f64, nthreads, (int)iters_done, (int)neq, nnz_full);

  *eps = rr_final;
  *niter = iters_done;

  SFREE(d_scale);
  SFREE(row_ptr);
  SFREE(col_idx);
  if (val_f32) SFREE(val_f32);
  if (val_f64) SFREE(val_f64);
  SFREE(b_scaled);
  SFREE(x_scaled);
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
// 32 time rows x 64 output columns, 32 reduction values, 256 threads.
// Maxwell FP32 FMA; each thread owns a 2x4 register tile.
extern "C" __global__ void q8_gemm(const unsigned char* w,const float* x,float* y,int m,int k,int n) {
    __shared__ float xs[32][33];
    __shared__ float ws[64][33];
    int tid=threadIdx.x,tx=tid%16,ty=tid/16;
    int tiles_n=(n+63)/64,bm=blockIdx.x/tiles_n,bn=blockIdx.x%tiles_n;
    float acc[2][4]={{0}};
    for(int start=0;start<k;start+=32) {
        for(int i=tid;i<32*32;i+=256) {
            int row=bm*32+i/32,col=start+i%32;
            xs[i/32][i%32]=(row<m&&col<k)?x[row*k+col]:0;
        }
        for(int i=tid;i<64*32;i+=256) {
            int row=bn*64+i/32,col=start+i%32;
            float value=0;
            if(row<n&&col<k) {
                int block=(row*k+col)/32;
                float scale=__half2float(*reinterpret_cast<const __half*>(w+block*34));
                value=scale*reinterpret_cast<const signed char*>(w+block*34+2)[col%32];
            }
            ws[i/32][i%32]=value;
        }
        __syncthreads();
        #pragma unroll
        for(int d=0;d<32;d++) {
            float a0=xs[ty][d],a1=xs[ty+16][d];
            #pragma unroll
            for(int j=0;j<4;j++) {
                float b=ws[tx+j*16][d];acc[0][j]+=a0*b;acc[1][j]+=a1*b;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<2;i++)for(int j=0;j<4;j++) {
        int row=bm*32+ty+i*16,col=bn*64+tx+j*16;
        if(row<m&&col<n)y[row*n+col]=acc[i][j];
    }
}

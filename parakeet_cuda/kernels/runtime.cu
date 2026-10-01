// SPDX-License-Identifier: Apache-2.0
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ float sum_block(float v) {
    __shared__ float tmp[256];
    tmp[threadIdx.x] = v; __syncthreads();
    for (int d=blockDim.x/2; d; d>>=1) {
        if (threadIdx.x<d) tmp[threadIdx.x] += tmp[threadIdx.x+d];
        __syncthreads();
    }
    return tmp[0];
}
__device__ float max_block(float v) {
    __shared__ float tmp[256];
    tmp[threadIdx.x] = v; __syncthreads();
    for (int d=blockDim.x/2; d; d>>=1) {
        if (threadIdx.x<d) tmp[threadIdx.x] = fmaxf(tmp[threadIdx.x],tmp[threadIdx.x+d]);
        __syncthreads();
    }
    return tmp[0];
}

extern "C" __global__ void dequant_q8(const unsigned char* x, float* y, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) {
        int b=i/32;
        float scale=__half2float(*reinterpret_cast<const __half*>(x+b*34));
        y[i]=scale * static_cast<float>(reinterpret_cast<const signed char*>(x+b*34+2)[i%32]);
    }
}
extern "C" __global__ void expand_half(const __half* x, float* y, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=__half2float(x[i]);
}
extern "C" __global__ void layer_norm(const float* x, const float* w, const float* b,
                                float* y, int width) {
    int row=blockIdx.x, t=threadIdx.x;
    float s=0;
    for(int c=t;c<width;c+=blockDim.x) s+=x[row*width+c];
    float mean=sum_block(s)/width;
    s=0;
    for(int c=t;c<width;c+=blockDim.x) {float d=x[row*width+c]-mean;s+=d*d;}
    float inv=rsqrtf(sum_block(s)/width+1e-5f);
    for(int c=t;c<width;c+=blockDim.x) y[row*width+c]=(x[row*width+c]-mean)*inv*w[c]+b[c];
}
extern "C" __global__ void silu(float* x,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=sigmoid(x[i]);
}
extern "C" __global__ void residual(float* x,const float* y,float scale,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=scale*y[i];
}
extern "C" __global__ void bias_act(float* x,const float* bias,int width,int n,int relu) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {float v=x[i]+bias[i%width];x[i]=relu?fmaxf(v,0.f):v;}
}
extern "C" __global__ void glu(const float* x,float* y,int width,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {int j=(i/width)*2*width+i%width;y[i]=x[j]*sigmoid(x[j+width]);}
}
extern "C" __global__ void depthwise1(const float* x,const float* w,const float* gamma,
 const float* beta,const float* mean,const float* var,float* y,int input_t,int width,int start,int out_t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<out_t*width) {
        int t=i/width+start,c=i%width;float v=0;
        for(int k=0;k<9;k++) {int j=t+k-4;if(j>=0&&j<input_t)v+=x[j*width+c]*w[c*9+k];}
        v=(v-mean[c])/sqrtf(var[c]+1e-5f)*gamma[c]+beta[c];y[i]=v*sigmoid(v);
    }
}
// Time-major, frequency-middle, channel-inner buffers. Stored last axis is frequency.
extern "C" __global__ void conv_first(const float* x,const float* w,const float* b,float* y,
 int input_t,int input_f,int start,int out_t,int out_f,int channels) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<out_t*out_f*channels) {
        int c=i%channels,f=(i/channels)%out_f,t=i/(channels*out_f)+start;float v=b[c];
        for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++) {
            int it=2*t+kx-1,jf=2*f+ky-1;
            if(it>=0&&it<input_t&&jf>=0&&jf<input_f)v+=x[it*input_f+jf]*w[c*9+kx*3+ky];
        }
        y[i]=fmaxf(v,0.f);
    }
}
extern "C" __global__ void conv_dw2(const float* x,const float* w,const float* b,float* y,
 int input_t,int input_f,int input_start,int output_start,int out_t,int out_f,int channels) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<out_t*out_f*channels) {
        int c=i%channels,f=(i/channels)%out_f,t=i/(channels*out_f)+output_start;float v=b[c];
        for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++) {
            int it=2*t+kx-1,jf=2*f+ky-1;
            if(it>=input_start&&it<input_start+input_t&&jf>=0&&jf<input_f)
                v+=x[((it-input_start)*input_f+jf)*channels+c]*w[c*9+kx*3+ky];
        }
        y[i]=v;
    }
}
extern "C" __global__ void flatten_stem(const float* x,float* y,int t,int f,int c) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*f*c) {int ch=(i/f)%c,fr=i%f,ti=i/(f*c);y[i]=x[(ti*f+fr)*c+ch];}
}
extern "C" __global__ void split_qkv(const float* x,float* q,float* k,float* v,int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*1024) {
        int d=i%128,ti=(i/128)%t,h=i/(128*t),src=ti*3072+h*128+d;
        q[i]=x[src];k[i]=x[src+1024];v[i]=x[src+2048];
    }
}
extern "C" __global__ void split_heads(const float* x,float* y,int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*1024) {int d=i%128,ti=(i/128)%t,h=i/(128*t);y[i]=x[ti*1024+h*128+d];}
}
extern "C" __global__ void merge_heads(const float* x,float* y,int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*1024) {int d=i%128,ti=(i/128)%t,h=i/(128*t);y[ti*1024+h*128+d]=x[i];}
}
extern "C" __global__ void query_bias(const float* q,const float* u,const float* v,
 float* qu,float* qv,int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*1024) {int c=(i/(t*128))*128+i%128;qu[i]=q[i]+u[c];qv[i]=q[i]+v[c];}
}
extern "C" __global__ void rel_scores(float* scores,const float* rel,int q,int k,int p) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<8*q*k) {
        int j=i%k,qi=(i/k)%q,h=i/(q*k);
        scores[i]=(scores[i]+rel[(h*q+qi)*p+q-1+j-qi])*(1.f/sqrtf(128.f));
    }
}
extern "C" __global__ void softmax(float* scores,int k) {
    float* row=scores+blockIdx.x*k;
    float m=-INFINITY;
    for(int j=threadIdx.x;j<k;j+=blockDim.x)m=fmaxf(m,row[j]);
    m=max_block(m);float s=0;
    for(int j=threadIdx.x;j<k;j+=blockDim.x) {float v=expf(row[j]-m);row[j]=v;s+=v;}
    s=sum_block(s);
    for(int j=threadIdx.x;j<k;j+=blockDim.x)row[j]/=s;
}
extern "C" __global__ void online_probs(float* scores,float* state,float* alpha,int k,int q,int first) {
    int r=blockIdx.x;float* row=scores+r*k;
    float m=-INFINITY;
    for(int j=threadIdx.x;j<k;j+=blockDim.x)m=fmaxf(m,row[j]);
    m=max_block(m);float old=first?-INFINITY:state[r*2],old_l=first?0:state[r*2+1];
    float next=fmaxf(old,m),a=first?0:expf(old-next),s=0;
    for(int j=threadIdx.x;j<k;j+=blockDim.x) {float v=expf(row[j]-next);row[j]=v;s+=v;}
    s=sum_block(s);
    if(threadIdx.x==0) {state[r*2]=next;state[r*2+1]=a*old_l+s;alpha[r]=a;}
}
extern "C" __global__ void online_accum(float* acc,const float* part,const float* alpha,int n,int first) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)acc[i]=(first?0:acc[i]*alpha[i/128])+part[i];
}
extern "C" __global__ void online_finish(float* acc,const float* state,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)acc[i]/=state[(i/128)*2+1];
}
extern "C" __global__ void lstm(const float* ih,const float* hh,const float* bi,const float* bh,
 const float* c,float* hn,float* cn,int hidden) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<hidden) {
        float g[4];for(int j=0;j<4;j++){int k=j*hidden+i;g[j]=ih[k]+hh[k]+bi[k]+bh[k];}
        float nc=sigmoid(g[1])*c[i]+sigmoid(g[0])*tanhf(g[2]);
        cn[i]=nc;hn[i]=sigmoid(g[3])*tanhf(nc);
    }
}
extern "C" __global__ void joint_relu(const float* enc,const float* pred,float* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=fmaxf(enc[i]+pred[i],0.f);
}
extern "C" __global__ void argmax_tdt(const float* logits,int* out) {
    __shared__ float vals[256];__shared__ int ids[256];
    int t=threadIdx.x,base=blockIdx.x?8193:0,n=blockIdx.x?5:8193;
    float v=-INFINITY;int id=0;
    for(int j=t;j<n;j+=blockDim.x)if(logits[base+j]>v){v=logits[base+j];id=j;}
    vals[t]=v;ids[t]=id;__syncthreads();
    for(int d=blockDim.x/2;d;d>>=1) {
        if(t<d&&(vals[t+d]>vals[t]||(vals[t+d]==vals[t]&&ids[t+d]<ids[t]))) {
            vals[t]=vals[t+d];ids[t]=ids[t+d];
        }__syncthreads();
    }
    if(!t)out[blockIdx.x]=ids[0];
}
extern "C" __global__ void q8_gemv(const unsigned char* weights,const float* x,float* y,int k,int n) {
    int row=blockIdx.x;float s=0;
    for(int j=threadIdx.x;j<k;j+=blockDim.x) {
        int b=(row*k+j)/32;
        float scale=__half2float(*reinterpret_cast<const __half*>(weights+b*34));
        s+=scale*reinterpret_cast<const signed char*>(weights+b*34+2)[j%32]*x[j];
    }
    s=sum_block(s);if(!threadIdx.x)y[row]=s;
}

// Nemotron 3: fuse packed QKV transpose with full NEOX RoPE (split half).
extern "C" __global__ void diar_rope_qkv(const float* packed,float* q,float* k,float* v,int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*512) {
        int d=i%64,ti=(i/64)%t,h=i/(64*t);
        int pair=d%32,src=ti*1536+h*64+pair;
        float angle=ti*powf(10000.f,-2.f*pair/64.f),sn,cs;
        sincosf(angle,&sn,&cs);
        float qa=packed[src],qb=packed[src+32],ka=packed[src+512],kb=packed[src+544];
        q[i]=d<32?qa*cs-qb*sn:qa*sn+qb*cs;
        k[i]=d<32?ka*cs-kb*sn:ka*sn+kb*cs;
        v[i]=packed[ti*1536+1024+h*64+d];
    }
}
extern "C" __global__ void diar_merge_heads(const float* x,float* y,int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*512) {int d=i%64,ti=(i/64)%t,h=i/(64*t);y[ti*512+h*64+d]=x[i];}
}
extern "C" __global__ void softmax_scaled(float* scores,int k,float scale) {
    float* row=scores+blockIdx.x*k;
    float m=-INFINITY;
    for(int j=threadIdx.x;j<k;j+=blockDim.x)m=fmaxf(m,row[j]*scale);
    m=max_block(m);float s=0;
    for(int j=threadIdx.x;j<k;j+=blockDim.x) {float a=expf(row[j]*scale-m);row[j]=a;s+=a;}
    s=sum_block(s);
    for(int j=threadIdx.x;j<k;j+=blockDim.x)row[j]/=s;
}
extern "C" __global__ void gelu_bias(float* x,const float* bias,int width,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {float a=x[i]+bias[i%width];x[i]=.5f*a*(1.f+erff(a*.7071067811865475f));}
}
// im2col + cuBLAS avoids a costly 192*3 serial reduction per output.
extern "C" __global__ void diar_conv_columns(const float* x,float* columns,int t) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<t*576) {
        int ti=i/576,c=(i%576)/3,dt=i%3,j=ti+dt-1;
        columns[i]=(j>=0&&j<t)?x[j*192+c]:0.f;
    }
}
extern "C" __global__ void sigmoid_bias(float* x,const float* bias,int width,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=sigmoid(x[i]+bias[i%width]);
}

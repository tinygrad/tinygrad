typedef long unsigned int size_t;
typedef __bf16 hip_bfloat16;
typedef __bf16 bf2 __attribute__((ext_vector_type(2)));
typedef float f2 __attribute__((ext_vector_type(2)));
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) float __ocml_sqrt_f32(float);
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(128, 128)))
void ffn_residual_norm_packed_raw(hip_bfloat16* checkpoint, float* denom, hip_bfloat16* normalized, float* rrms,
                            const hip_bfloat16* residual, const hip_bfloat16* projection,
                            const hip_bfloat16* bias, const hip_bfloat16* weight) {
  const int tid=__ockl_get_local_id(0), pair=tid%32, row_local=tid/32;
  const int row=__ockl_get_group_id(0)*4+row_local;
  __attribute__((shared, aligned(16))) float partial[256];
  __attribute__((shared, aligned(16))) float inv[4];
  unsigned h[45];
  f2 sum={0.0f,0.0f};
  #pragma unroll
  for(int i=0;i<45;i++) {
    int c=pair*2+64*i;
    bf2 r=*reinterpret_cast<const bf2*>(residual+row*2880+c);
    bf2 p=*reinterpret_cast<const bf2*>(projection+row*3072+c);
    bf2 b=*reinterpret_cast<const bf2*>(bias+c);
    bf2 pb=__builtin_convertvector(__builtin_convertvector(p,f2)+__builtin_convertvector(b,f2),bf2);
    h[i]=__builtin_bit_cast(unsigned,__builtin_convertvector(__builtin_convertvector(r,f2)+__builtin_convertvector(pb,f2),bf2));
    asm volatile("" : "+v"(h[i]));
    *reinterpret_cast<bf2*>(checkpoint+row*2880+c)=__builtin_bit_cast(bf2,h[i]);
    f2 v=__builtin_convertvector(__builtin_bit_cast(bf2,h[i]),f2);
    sum=sum+v*v;
  }
  partial[row_local*64+pair*2]=sum[0];partial[row_local*64+pair*2+1]=sum[1];
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  if(pair==0) {
    float total=0.0f;
    #pragma unroll
    for(int i=0;i<16;i++) {
      #pragma unroll
      for(int j=0;j<4;j++) total=total+partial[row_local*64+i+16*j];
    }
    float d=__ocml_sqrt_f32(total*0.00034722223062999547f+9.999999747378752e-06f);
    denom[row]=d;inv[row_local]=1.0f/d;rrms[row]=1.0f/d;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  float reciprocal=inv[row_local];
  #pragma unroll
  for(int i=0;i<45;i++) {
    int c=pair*2+64*i;
    f2 w=__builtin_convertvector(*reinterpret_cast<const bf2*>(weight+c),f2);
    *reinterpret_cast<bf2*>(normalized+row*2880+c)=__builtin_convertvector(__builtin_convertvector(__builtin_bit_cast(bf2,h[i]),f2)*reciprocal*w,bf2);
  }
}

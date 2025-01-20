#include <tgmath.h>
unsigned short buf_0[2048];
unsigned char  input0[430080]; 
unsigned char buf_1[2];
unsigned char buf_2[4];
signed char buf_3[32768];
unsigned char buf_4[524288];
float   buf_5[2048];
typedef __fp16 __fp164 __attribute__((aligned(8), vector_size(8)));
typedef float  float4  __attribute__((aligned(16), vector_size(16)));
float       output0[524288];

void E_512_4(unsigned short* restrict data0, unsigned char* restrict data1) {
  for (int ridx0 = 0; ridx0 < 512; ridx0++) {
    int alu0 = (ridx0*840);
    unsigned char val0 = *(data1+(alu0+208));
    unsigned char val1 = *(data1+(alu0+209));
    unsigned char val2 = *(data1+(alu0+418));
    unsigned char val3 = *(data1+(alu0+419));
    unsigned char val4 = *(data1+(alu0+628));
    unsigned char val5 = *(data1+(alu0+629));
    unsigned char val6 = *(data1+(alu0+838));
    unsigned char val7 = *(data1+(alu0+839));
    int alu1 = (ridx0<<2);
    *(data0+alu1) = (((unsigned short)(val0))+(((unsigned short)(val1))<<((unsigned short)(8u))));
    *(data0+(alu1+1)) = (((unsigned short)(val2))+(((unsigned short)(val3))<<((unsigned short)(8u))));
    *(data0+(alu1+2)) = (((unsigned short)(val4))+(((unsigned short)(val5))<<((unsigned short)(8u))));
    *(data0+(alu1+3)) = (((unsigned short)(val6))+(((unsigned short)(val7))<<((unsigned short)(8u))));
  }
}

void E_2(unsigned char* restrict data0) {
  *(data0+0) = ((unsigned char)(1u));
  *(data0+1) = ((unsigned char)(16u));
}

void E_4(unsigned char* restrict data0) {
  *(data0+0) = ((unsigned char)(1u));
  *(data0+1) = ((unsigned char)(4u));
  *(data0+2) = ((unsigned char)(16u));
  *(data0+3) = ((unsigned char)(64u));
}

void E_2048_4_4(signed char* restrict data0, unsigned char* restrict data1) {
  for (int ridx0 = 0; ridx0 < 2048; ridx0++) {
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu0 = (ridx1<<2);
      int alu1 = ((ridx0*210)+alu0);
      unsigned char val0 = *(data1+(alu1+192));
      unsigned char val1 = *(data1+(alu1+193));
      unsigned char val2 = *(data1+(alu1+194));
      unsigned char val3 = *(data1+(alu1+195));
      unsigned char precast0 = val0;
      unsigned char precast1 = val1;
      unsigned char precast2 = val2;
      unsigned char precast3 = val3;
      int alu2 = ((ridx0<<4)+alu0);
      *(data0+alu2) = (*((signed char*)&precast0));
      *(data0+(alu2+1)) = (*((signed char*)&precast1));
      *(data0+(alu2+2)) = (*((signed char*)&precast2));
      *(data0+(alu2+3)) = (*((signed char*)&precast3));
    }
  }
}

void E_2048_2_32_4(unsigned char* restrict data0, unsigned char* restrict data1, unsigned char* restrict data2, unsigned char* restrict data3) {
  for (int ridx0 = 0; ridx0 < 2048; ridx0++) {
    int alu0 = (ridx0*210);
    for (int ridx1 = 0; ridx1 < 2; ridx1++) {
      for (int ridx2 = 0; ridx2 < 32; ridx2++) {
        unsigned char val0 = *(data3+(ridx2>>3));
        unsigned char val1 = *(data2+(ridx2>>4));
        int alu1 = (alu0+(ridx1<<5)+((ridx2&7)<<2));
        unsigned char val2 = *(data1+(alu1+128));
        unsigned char val3 = *(data1+(alu1+129));
        unsigned char val4 = *(data1+(alu1+130));
        unsigned char val5 = *(data1+(alu1+131));
        int alu2 = (alu0+(ridx1<<6)+((ridx2&15)<<2));
        unsigned char val6 = *(data1+alu2);
        unsigned char val7 = *(data1+(alu2+1));
        unsigned char val8 = *(data1+(alu2+2));
        unsigned char val9 = *(data1+(alu2+3));
        int alu3 = ((ridx0<<8)+(ridx1<<7)+(ridx2<<2));
        *(data0+(alu3+1)) = ((((val3/val0)&((unsigned char)(3u)))<<((unsigned char)(4u)))|((val7/val1)&((unsigned char)(15u))));
        *(data0+(alu3+2)) = ((((val4/val0)&((unsigned char)(3u)))<<((unsigned char)(4u)))|((val8/val1)&((unsigned char)(15u))));
        *(data0+(alu3+3)) = ((((val5/val0)&((unsigned char)(3u)))<<((unsigned char)(4u)))|((val9/val1)&((unsigned char)(15u))));
        *(data0+alu3) = ((((val2/val0)&((unsigned char)(3u)))<<((unsigned char)(4u)))|((val6/val1)&((unsigned char)(15u))));
      }
    }
  }
}
typedef __fp16 __fp164 __attribute__((aligned(8),vector_size(8)));
typedef float float4 __attribute__((aligned(16),vector_size(16)));
void E_512_4n1(float* restrict data0, __fp16* restrict data1) {
  for (int ridx0 = 0; ridx0 < 512; ridx0++) {
    int alu0 = (ridx0<<2);
    __fp164 val0 = *((__fp164*)((data1+alu0)));
    *((float4*)((data0+alu0))) = (float4){((float)(val0[0])),((float)(val0[1])),((float)(val0[2])),((float)(val0[3]))};
  }
}
typedef float float4 __attribute__((aligned(16),vector_size(16)));
void E_2048_64_4(float* restrict data0, float* restrict data1, signed char* restrict data2, signed char* restrict data3) {
  for (int ridx0 = 0; ridx0 < 2048; ridx0++) {
    float val0 = *(data1+ridx0);
    for (int ridx1 = 0; ridx1 < 64; ridx1++) {
      signed char val1 = *(data3+((ridx0<<4)+(ridx1>>2)));
      int alu0 = ((ridx0<<8)+(ridx1<<2));
      signed char val2 = *(data2+alu0);
      signed char val3 = *(data2+(alu0+1));
      signed char val4 = *(data2+(alu0+2));
      signed char val5 = *(data2+(alu0+3));
      float cast0 = ((float)(val1));
      *((float4*)((data0+alu0))) = (float4){(cast0*((float)((val2+((signed char)(-32)))))*val0),(cast0*((float)((val3+((signed char)(-32)))))*val0),(cast0*((float)((val4+((signed char)(-32)))))*val0),(cast0*((float)((val5+((signed char)(-32)))))*val0)};
    }
  }
}

void net(unsigned char* input0, float* output0) {
E_512_4(buf_0, input0);
E_2(buf_1);
E_4(buf_2);
E_2048_4_4(buf_3, input0);
E_2048_2_32_4(buf_4, input0, buf_1, buf_2);
__fp16* buf_6 = (__fp16*)buf_0;
E_512_4n1(buf_5, buf_6);
signed char* buf_7 = (signed char*)buf_4;
E_2048_64_4(output0, buf_5, buf_7, buf_3);
}
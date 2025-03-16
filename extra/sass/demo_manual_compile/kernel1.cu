#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(9) r_3_3_3_3_9(float* data0, float* data1, float* data2) {
  int gidx0 = blockIdx.x; /* 3 */
  int gidx1 = blockIdx.y; /* 3 */
  int lidx0 = threadIdx.x; /* 3 */
  int lidx1 = threadIdx.y; /* 3 */
  int alu0 = (gidx0*3);
  int alu1 = (gidx1*27);
  int alu2 = (lidx0*9);
  int alu3 = (lidx1+alu0);
  float val0 = *(data2+alu3);
  int alu4 = (alu1+alu2);
  float val1 = *(data1+alu4);
  float val2 = *(data2+(alu3+9));
  float val3 = *(data2+(alu3+18));
  float val4 = *(data2+(alu3+27));
  float val5 = *(data2+(alu3+36));
  float val6 = *(data2+(alu3+45));
  float val7 = *(data2+(alu3+54));
  float val8 = *(data2+(alu3+63));
  float val9 = *(data2+(alu3+72));
  float val10 = *(data1+(alu4+1));
  float val11 = *(data1+(alu4+2));
  float val12 = *(data1+(alu4+3));
  float val13 = *(data1+(alu4+4));
  float val14 = *(data1+(alu4+5));
  float val15 = *(data1+(alu4+6));
  float val16 = *(data1+(alu4+7));
  float val17 = *(data1+(alu4+8));
  *(data0+(lidx1+alu2+alu0+alu1)) = ((val1*val0)+(val10*val2)+(val11*val3)+(val12*val4)+(val13*val5)+(val14*val6)+(val15*val7)+(val16*val8)+(val17*val9));
}
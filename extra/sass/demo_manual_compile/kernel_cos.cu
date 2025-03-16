#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
extern "C" __global__ void __launch_bounds__(3) E_325_3_3(float* data0, float* data1) {
  int gidx0 = blockIdx.x; /* 325 */
  int lidx0 = threadIdx.x; /* 3 */
  int alu0 = ((gidx0*9)+(lidx0*3));
  float val0 = *(data1+alu0);
  int alu1 = (alu0+1);
  float val1 = *(data1+alu1);
  int alu2 = (alu0+2);
  float val2 = *(data1+alu2);
  float alu3 = (((bool)(val0))?((val0<0.0f)?-1.0f:1.0f):0.0f);
  float alu4 = (val0*alu3);
  *(data0+alu0) = (1.5707963267948966f-(alu3*(1.5707963267948966f-(sqrt((1.0f-alu4))*((((((((((((((-0.0012624911f*alu4)+0.0066700901f)*alu4)+-0.0170881256f)*alu4)+0.030891881f)*alu4)+-0.0501743046f)*alu4)+0.0889789874f)*alu4)+-0.2145988016f)*alu4)+1.570796305f)))));
  float alu6 = (((bool)(val1))?((val1<0.0f)?-1.0f:1.0f):0.0f);
  float alu7 = (val1*alu6);
  *(data0+alu1) = (1.5707963267948966f-(alu6*(1.5707963267948966f-(sqrt((1.0f-alu7))*((((((((((((((-0.0012624911f*alu7)+0.0066700901f)*alu7)+-0.0170881256f)*alu7)+0.030891881f)*alu7)+-0.0501743046f)*alu7)+0.0889789874f)*alu7)+-0.2145988016f)*alu7)+1.570796305f)))));
  float alu9 = (((bool)(val2))?((val2<0.0f)?-1.0f:1.0f):0.0f);
  float alu10 = (val2*alu9);
  *(data0+alu2) = (1.5707963267948966f-(alu9*(1.5707963267948966f-(sqrt((1.0f-alu10))*((((((((((((((-0.0012624911f*alu10)+0.0066700901f)*alu10)+-0.0170881256f)*alu10)+0.030891881f)*alu10)+-0.0501743046f)*alu10)+0.0889789874f)*alu10)+-0.2145988016f)*alu10)+1.570796305f)))));
}
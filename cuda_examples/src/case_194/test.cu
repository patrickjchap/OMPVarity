
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, float var_1,float var_2,int var_3,float var_4,float var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12,float var_13,float var_14,float var_15,float var_16,float var_17,float var_18,float var_19,float var_20,float var_21,float var_22,float var_23,float var_24,float var_25,float var_26,float var_27,float var_28,float var_29) {
if (comp == (var_1 - -1.2209E-37f - var_2)) {
  comp = var_4 - (+1.1637E-41f * var_5);
for (int i=0; i < var_3; ++i) {
  comp = -1.6746E-35f + (var_6 / (-1.6653E-42f + var_7 / (-1.8159E-18f + var_8)));
comp = var_9 + var_10 * (-1.3638E-36f - +1.9566E-6f);
comp += +1.9819E-37f + var_11 * log10f(var_12 * (+1.4948E34f * var_13 + var_14 + powf((var_15 * (var_16 / var_17 / var_18)), var_19 * var_20 + atan2f(-1.5268E-43f, +1.8931E23f - coshf(-1.2911E-36f / var_21 - +1.9380E35f)))));
}
if (comp == var_22 + (var_23 * var_24)) {
  comp += (var_25 - sinf(var_26 + (var_27 / asinf(var_28 - -1.9587E-36f - log10f((var_29 / (-1.4373E34f / +1.1303E34f * (-0.0f - -0.0f))))))));
}
}
   printf("%.17g\n", comp);

}

float* initPointer(float v) {
  float *ret = (float*) malloc(sizeof(float)*10);
  for(int i=0; i < 10; ++i)
    ret[i] = v;
  return ret;
}

int main(int argc, char** argv) {
/* Program variables */

  float tmp_1 = atof(argv[1]);
  float tmp_2 = atof(argv[2]);
  float tmp_3 = atof(argv[3]);
  int tmp_4 = atoi(argv[4]);
  float tmp_5 = atof(argv[5]);
  float tmp_6 = atof(argv[6]);
  float tmp_7 = atof(argv[7]);
  float tmp_8 = atof(argv[8]);
  float tmp_9 = atof(argv[9]);
  float tmp_10 = atof(argv[10]);
  float tmp_11 = atof(argv[11]);
  float tmp_12 = atof(argv[12]);
  float tmp_13 = atof(argv[13]);
  float tmp_14 = atof(argv[14]);
  float tmp_15 = atof(argv[15]);
  float tmp_16 = atof(argv[16]);
  float tmp_17 = atof(argv[17]);
  float tmp_18 = atof(argv[18]);
  float tmp_19 = atof(argv[19]);
  float tmp_20 = atof(argv[20]);
  float tmp_21 = atof(argv[21]);
  float tmp_22 = atof(argv[22]);
  float tmp_23 = atof(argv[23]);
  float tmp_24 = atof(argv[24]);
  float tmp_25 = atof(argv[25]);
  float tmp_26 = atof(argv[26]);
  float tmp_27 = atof(argv[27]);
  float tmp_28 = atof(argv[28]);
  float tmp_29 = atof(argv[29]);
  float tmp_30 = atof(argv[30]);

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18,tmp_19,tmp_20,tmp_21,tmp_22,tmp_23,tmp_24,tmp_25,tmp_26,tmp_27,tmp_28,tmp_29,tmp_30);
  cudaDeviceSynchronize();

  return 0;
}
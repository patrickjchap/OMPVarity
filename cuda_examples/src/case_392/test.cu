
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,float var_2,float var_3,float var_4,float var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12,float var_13,float var_14,float var_15,float var_16,float var_17,float var_18,float var_19,float var_20,float var_21,float var_22,float var_23,float var_24,float var_25,float var_26) {
if (comp < (-0.0f / fmodf(+1.8753E-41f + var_2 / -1.0982E-43f, (+1.1450E-41f / var_3)))) {
  float tmp_1 = (var_4 - (+1.3056E34f / -1.7195E35f));
comp += tmp_1 * var_5 * (var_6 / asinf(-1.2615E25f));
comp += var_7 - -1.8671E-24f;
for (int i=0; i < var_1; ++i) {
  comp = log10f(var_8 / (var_9 + tanhf(ceilf(-0.0f))));
comp += sinhf((var_10 / var_11 / -0.0f));
}
if (comp > var_12 * var_13) {
  float tmp_2 = -1.4037E-37f;
comp += tmp_2 / (var_14 - (var_15 / asinf((-1.5623E-19f + var_16))));
comp += powf(tanhf((var_17 + (+1.3386E-41f - -1.2844E-35f + (var_18 - +1.2629E18f)))), (var_19 / var_20 - -1.0365E-42f * +0.0f * -0.0f));
comp = (var_21 - -0.0f * var_22 - +1.5954E36f + +0.0f / var_23);
}
if (comp < +1.9744E-36f - -1.6597E9f + +0.0f) {
  comp += (var_24 - var_25 / -1.6978E-42f / var_26);
float tmp_3 = +1.3585E-35f - (-0.0f / +0.0f);
comp += tmp_3 * (+1.5991E34f + atan2f(+1.9960E-37f, -0.0f));
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
  int tmp_2 = atoi(argv[2]);
  float tmp_3 = atof(argv[3]);
  float tmp_4 = atof(argv[4]);
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

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18,tmp_19,tmp_20,tmp_21,tmp_22,tmp_23,tmp_24,tmp_25,tmp_26,tmp_27);
  cudaDeviceSynchronize();

  return 0;
}


/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,int var_2,float var_3,float var_4,float var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12,float var_13,float var_14,float var_15,float var_16,float var_17,float var_18,float var_19,float var_20,float var_21,float var_22,float var_23,float var_24,float var_25,float var_26,float var_27,float var_28,float var_29,float var_30,float var_31,float var_32) {
for (int i=0; i < var_1; ++i) {
  if (comp >= var_3 - (var_4 + var_5)) {
    for (int i=0; i < var_2; ++i) {
      if (comp == -1.0505E36f / (var_6 - (var_7 * var_8))) {
        comp += floorf(-0.0f);
comp += +1.5245E36f + logf(var_9 + +1.2094E35f / acosf(var_10 + (-1.1743E-28f + (-1.9368E34f + sqrtf((var_11 / fmodf(-1.5370E-43f, tanhf(+1.4469E-37f))))))));
if (comp > (+0.0f + var_12 + (-1.8527E-36f + -1.7724E-36f))) {
  comp = (-1.1195E-37f / var_13 - -1.8615E36f / sinhf(var_14 / +1.9936E-36f));
comp += acosf(powf(acosf((-0.0f + (var_15 + (var_16 / (-1.8716E-42f * cosf(+0.0f)))))), var_17 + (-1.3047E35f * +1.0484E34f * var_18 - var_19)));
}
if (comp < fabsf(+1.5676E-41f)) {
  float tmp_1 = +1.4288E-35f;
float tmp_2 = +0.0f;
comp = tmp_2 * tmp_1 * var_20 - (+1.5325E-43f - var_21);
}
if (comp > (+1.1002E-44f - (-0.0f * (+1.9985E-36f - (var_22 / (-1.6789E35f - var_23)))))) {
  comp += var_24 * (var_25 * (var_26 * ceilf(var_27 * var_28)));
comp = (var_29 + asinf((var_30 / +0.0f + -1.5086E-41f - (var_31 + var_32))));
}
}
}
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
  int tmp_3 = atoi(argv[3]);
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
  float tmp_28 = atof(argv[28]);
  float tmp_29 = atof(argv[29]);
  float tmp_30 = atof(argv[30]);
  float tmp_31 = atof(argv[31]);
  float tmp_32 = atof(argv[32]);
  float tmp_33 = atof(argv[33]);

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18,tmp_19,tmp_20,tmp_21,tmp_22,tmp_23,tmp_24,tmp_25,tmp_26,tmp_27,tmp_28,tmp_29,tmp_30,tmp_31,tmp_32,tmp_33);
  cudaDeviceSynchronize();

  return 0;
}

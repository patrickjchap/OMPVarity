
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,float var_2,float var_3,int var_4,int var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12,float var_13,float var_14,float var_15,float var_16,float var_17,float var_18,float var_19,float var_20,float var_21,float var_22,float var_23,float var_24,float var_25,float var_26,float var_27) {
if (comp < -1.7421E20f / (var_2 + (-1.9119E-29f + (+1.6852E36f - -1.5755E36f * var_3)))) {
  for (int i=0; i < var_1; ++i) {
    comp += var_6 * var_7 * (var_8 - var_9);
if (comp <= (+1.4077E29f - +0.0f * coshf(var_10 + cosf((var_11 / (var_12 + (var_13 - var_14))))))) {
  float tmp_1 = +1.3231E1f;
comp += tmp_1 * ceilf(-0.0f);
}
for (int i=0; i < var_4; ++i) {
  comp = floorf(fmodf(var_15 / (-1.0474E-36f * fmodf(-1.4453E-37f, (var_16 / var_17))), (-1.4296E-37f + +0.0f * var_18)));
comp += var_19 / -1.4939E-36f * (var_20 + (-1.2341E-15f / var_21 * +1.6801E-35f));
}
for (int i=0; i < var_5; ++i) {
  float tmp_2 = (-0.0f / -1.2244E-37f / var_22 / +1.2306E-42f);
comp = tmp_2 * +1.6979E-43f + (-1.0369E34f - (var_23 * sinf(tanhf(var_24 + (var_25 - (-1.1242E-44f + (var_26 + (var_27 - +1.0765E-37f))))))));
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
  float tmp_3 = atof(argv[3]);
  float tmp_4 = atof(argv[4]);
  int tmp_5 = atoi(argv[5]);
  int tmp_6 = atoi(argv[6]);
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

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18,tmp_19,tmp_20,tmp_21,tmp_22,tmp_23,tmp_24,tmp_25,tmp_26,tmp_27,tmp_28);
  cudaDeviceSynchronize();

  return 0;
}


/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,int var_2,float var_3,float var_4,int var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12,float var_13,float var_14,float var_15,float var_16,float var_17,float var_18,float var_19,float var_20,float var_21,float var_22,float var_23,float var_24,float var_25,float var_26,float var_27,float var_28,float var_29,float var_30) {
if (comp == (-1.3650E-35f - -1.6966E35f + (var_3 - (var_4 * +0.0f)))) {
  for (int i=0; i < var_1; ++i) {
    for (int i=0; i < var_2; ++i) {
      comp += sinf((-1.7156E35f + coshf(var_6 - (-0.0f + var_7))));
comp += -1.8569E9f - -1.0959E-36f + var_8 * var_9 + -1.9780E-36f;
if (comp > (var_10 * var_11)) {
  comp = (+1.6505E-28f - (+1.6974E2f - (var_12 * var_13 + (var_14 + var_15))));
comp += (var_16 / logf(-1.4145E-36f / +1.8719E34f / -1.6747E-17f + var_17 + (+1.6221E34f + var_18)));
comp = (var_19 / atan2f((var_20 - (var_21 - var_22 - var_23)), -1.1830E-37f - +1.0270E-44f + (var_24 / var_25)));
comp += (-1.3441E-42f * (var_26 * (+1.0655E12f * -1.6187E-37f)));
}
for (int i=0; i < var_5; ++i) {
  comp += (+0.0f / log10f(-0.0f * (var_27 + +0.0f - (var_28 * (+0.0f * var_29)))));
float tmp_1 = +1.3321E36f;
comp = tmp_1 + (var_30 * (+1.2326E26f + (-1.2799E20f / +1.1198E-42f)));
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
  float tmp_29 = atof(argv[29]);
  float tmp_30 = atof(argv[30]);
  float tmp_31 = atof(argv[31]);

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18,tmp_19,tmp_20,tmp_21,tmp_22,tmp_23,tmp_24,tmp_25,tmp_26,tmp_27,tmp_28,tmp_29,tmp_30,tmp_31);
  cudaDeviceSynchronize();

  return 0;
}

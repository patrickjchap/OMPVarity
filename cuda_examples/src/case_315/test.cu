
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,float var_2,float var_3,float var_4,float var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12,float var_13,float var_14,float var_15,float var_16,float var_17,float var_18,float var_19,float var_20,float var_21,float var_22,float var_23,float var_24,float var_25,float* var_26,float var_27,float var_28,float var_29) {
if (comp >= +1.2098E-44f - +1.3675E36f + var_2 * (var_3 + +1.8233E-42f)) {
  if (comp == (var_4 / -1.6941E-36f / var_5)) {
    if (comp <= (-0.0f / var_6 + expf(var_7 * var_8 - +1.1490E-44f))) {
      float tmp_1 = +0.0f;
float tmp_2 = var_9 * var_10;
comp += tmp_2 / tmp_1 - (+1.6737E-37f / ceilf((var_11 / var_12 - (-1.9241E-43f + var_13 - (var_14 + var_15)))));
if (comp == fabsf(acosf(-1.0522E10f + (-0.0f + sinf((-1.3195E-20f / (var_16 * var_17))))))) {
  float tmp_3 = (-1.7483E34f - (var_18 * (+1.9849E-35f / -0.0f - var_19 + var_20)));
comp = tmp_3 - (var_21 - cosf(-1.1098E36f));
}
if (comp <= (var_22 - (-1.6237E-16f - var_23 - (var_24 * -1.0733E28f)))) {
  comp = (var_25 + -1.2367E34f - -1.3208E-42f + (-1.8013E34f - +1.4686E-43f));
}
for (int i=0; i < var_1; ++i) {
  var_26[i] = +1.1237E-44f;
comp += var_26[i] - atanf((var_27 * -1.7977E12f));
float tmp_4 = +1.5908E-42f;
comp = tmp_4 + (-1.5652E-23f / var_28 - var_29);
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
  float* tmp_27 = initPointer( atof(argv[27]) );
  float tmp_28 = atof(argv[28]);
  float tmp_29 = atof(argv[29]);
  float tmp_30 = atof(argv[30]);

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18,tmp_19,tmp_20,tmp_21,tmp_22,tmp_23,tmp_24,tmp_25,tmp_26,tmp_27,tmp_28,tmp_29,tmp_30);
  cudaDeviceSynchronize();

  return 0;
}

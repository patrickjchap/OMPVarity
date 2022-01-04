
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,float var_2,int var_3,int var_4,float var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float* var_12,float var_13,float var_14,float var_15,float var_16,float var_17,float var_18,float var_19,float var_20,float var_21,float* var_22,float* var_23,float var_24,float var_25,float var_26,float var_27,float var_28,float var_29,float var_30,float var_31,float var_32,float var_33) {
for (int i=0; i < var_1; ++i) {
  if (comp > (var_2 + -1.5747E-35f)) {
    comp += coshf((+0.0f / +1.3473E-43f));
if (comp >= -1.3494E36f - (-0.0f * (+1.7416E-44f - var_5))) {
  comp += asinf((var_6 + var_7));
float tmp_1 = -0.0f + (+0.0f * var_8);
float tmp_2 = -1.9823E34f / +0.0f - var_9;
comp = tmp_2 * tmp_1 + (var_10 * var_11);
}
for (int i=0; i < var_3; ++i) {
  var_12[i] = var_13 + var_14 * fabsf((var_15 - var_16 * ldexpf(floorf(tanhf((-1.9774E-35f * var_17 + var_18))), 2)));
float tmp_3 = +1.5409E26f;
comp = tmp_3 / var_12[i] / -1.2900E-35f + var_19 / (var_20 / (-1.3379E-15f - -1.8224E-35f - var_21));
}
for (int i=0; i < var_4; ++i) {
  var_22[i] = sinhf(acosf(var_24 + +1.4771E-35f));
var_23[i] = +1.8351E-35f * (-1.0405E10f * (var_25 / (var_26 * fmodf((-1.5535E35f + -1.3174E-24f * -1.2706E-41f - (-1.4619E-35f - var_27)), (var_28 + (var_29 - (var_30 / ceilf(+1.0588E-43f))))))));
comp += var_23[i] + var_22[i] + powf(var_31 - (var_32 / -1.1886E34f * -0.0f - (var_33 - +0.0f)), -1.0284E-35f);
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
  int tmp_4 = atoi(argv[4]);
  int tmp_5 = atoi(argv[5]);
  float tmp_6 = atof(argv[6]);
  float tmp_7 = atof(argv[7]);
  float tmp_8 = atof(argv[8]);
  float tmp_9 = atof(argv[9]);
  float tmp_10 = atof(argv[10]);
  float tmp_11 = atof(argv[11]);
  float tmp_12 = atof(argv[12]);
  float* tmp_13 = initPointer( atof(argv[13]) );
  float tmp_14 = atof(argv[14]);
  float tmp_15 = atof(argv[15]);
  float tmp_16 = atof(argv[16]);
  float tmp_17 = atof(argv[17]);
  float tmp_18 = atof(argv[18]);
  float tmp_19 = atof(argv[19]);
  float tmp_20 = atof(argv[20]);
  float tmp_21 = atof(argv[21]);
  float tmp_22 = atof(argv[22]);
  float* tmp_23 = initPointer( atof(argv[23]) );
  float* tmp_24 = initPointer( atof(argv[24]) );
  float tmp_25 = atof(argv[25]);
  float tmp_26 = atof(argv[26]);
  float tmp_27 = atof(argv[27]);
  float tmp_28 = atof(argv[28]);
  float tmp_29 = atof(argv[29]);
  float tmp_30 = atof(argv[30]);
  float tmp_31 = atof(argv[31]);
  float tmp_32 = atof(argv[32]);
  float tmp_33 = atof(argv[33]);
  float tmp_34 = atof(argv[34]);

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18,tmp_19,tmp_20,tmp_21,tmp_22,tmp_23,tmp_24,tmp_25,tmp_26,tmp_27,tmp_28,tmp_29,tmp_30,tmp_31,tmp_32,tmp_33,tmp_34);
  cudaDeviceSynchronize();

  return 0;
}


/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,float var_2,float* var_3,int var_4,float var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12,float var_13,float var_14,float var_15,float var_16,float var_17,float var_18,float var_19,float var_20,float var_21,float var_22,float var_23,float var_24) {
if (comp > (-1.1746E-37f * +1.8329E-44f / floorf(var_2 - +0.0f))) {
  for (int i=0; i < var_1; ++i) {
    comp = +0.0f - var_5 + var_6 * var_7;
comp = ceilf(-0.0f);
var_3[i] = (var_8 / (+1.1739E-10f * var_9));
comp += var_3[i] + +1.4254E-36f / (+1.6838E-44f + +1.7860E-42f - var_10 / (-1.0179E35f - var_11));
for (int i=0; i < var_4; ++i) {
  comp = var_12 + +1.9376E34f * (var_13 / (var_14 / var_15));
}
if (comp <= -1.5750E35f - var_16 / (var_17 + var_18)) {
  float tmp_1 = var_19 * (+1.4672E-37f - +1.4490E-37f * +1.1942E-37f);
comp += tmp_1 - -1.0597E-35f + (+1.6730E36f / +0.0f + (var_20 - -1.3493E-7f));
float tmp_2 = +0.0f + +1.6589E15f - var_21 / +1.3478E-36f;
comp = tmp_2 + (var_22 + sinf((var_23 / (-1.0234E-1f * var_24))));
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
  float* tmp_4 = initPointer( atof(argv[4]) );
  int tmp_5 = atoi(argv[5]);
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

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18,tmp_19,tmp_20,tmp_21,tmp_22,tmp_23,tmp_24,tmp_25);
  cudaDeviceSynchronize();

  return 0;
}

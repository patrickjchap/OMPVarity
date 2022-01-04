
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,float var_2,float var_3,int var_4,float var_5,float* var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12,float var_13,float var_14,float var_15,float var_16,float var_17) {
for (int i=0; i < var_1; ++i) {
  if (comp > var_2 / (-1.6664E-27f * (+0.0f - var_3))) {
    comp += (+1.8770E35f * var_5);
for (int i=0; i < var_4; ++i) {
  var_6[i] = -1.6712E35f;
float tmp_1 = -1.8464E-36f;
comp = tmp_1 / var_6[i] + (var_7 + var_8);
}
if (comp > fmodf(+1.6640E-41f, +1.5258E-35f * logf(logf(-0.0f * (var_9 / (+0.0f * var_10 - -1.9280E-26f)))))) {
  comp = -1.5239E18f - var_11;
comp += -0.0f / +1.4122E36f / var_12 / (var_13 + (var_14 * var_15));
float tmp_2 = -1.5238E-43f;
comp = tmp_2 / (-1.8835E-44f - var_16 - var_17);
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
  float tmp_6 = atof(argv[6]);
  float* tmp_7 = initPointer( atof(argv[7]) );
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

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13,tmp_14,tmp_15,tmp_16,tmp_17,tmp_18);
  cudaDeviceSynchronize();

  return 0;
}

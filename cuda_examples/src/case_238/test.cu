
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,float var_2,float var_3,float* var_4,int var_5,float var_6,float var_7,float var_8,float var_9,float var_10,float var_11,float var_12) {
if (comp <= (var_2 * sinhf(+1.9369E-37f / (-1.2784E-36f + coshf(-1.5536E-42f + +0.0f + (var_3 * +1.1114E-35f)))))) {
  for (int i=0; i < var_1; ++i) {
    comp += var_6 + (-1.7184E34f + -1.3381E10f);
var_4[i] = +1.5118E35f;
comp += var_4[i] / (var_7 - (var_8 / (var_9 * -1.9421E-37f / atanf(-1.0252E-36f - +1.2662E-19f / var_10 - (+1.6486E-36f * +0.0f)))));
comp += +1.7407E36f / (-1.1066E-30f + +1.1145E-35f);
for (int i=0; i < var_5; ++i) {
  comp = var_11 * (-1.4938E-13f + var_12);
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
  float* tmp_5 = initPointer( atof(argv[5]) );
  int tmp_6 = atoi(argv[6]);
  float tmp_7 = atof(argv[7]);
  float tmp_8 = atof(argv[8]);
  float tmp_9 = atof(argv[9]);
  float tmp_10 = atof(argv[10]);
  float tmp_11 = atof(argv[11]);
  float tmp_12 = atof(argv[12]);
  float tmp_13 = atof(argv[13]);

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9,tmp_10,tmp_11,tmp_12,tmp_13);
  cudaDeviceSynchronize();

  return 0;
}

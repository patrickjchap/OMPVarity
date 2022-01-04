
/* This is a automatically generated test. Do not modify */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void compute(float comp, int var_1,float var_2,float var_3,float var_4,float* var_5,float* var_6,float var_7,float var_8) {
if (comp == (var_2 - var_3 / var_4 - atan2f(-0.0f, +1.1260E-43f))) {
  for (int i=0; i < var_1; ++i) {
    var_5[i] = -1.8635E-44f;
var_6[i] = -1.9821E-43f;
comp += var_6[i] * var_5[i] / (var_7 - -1.0268E-41f * var_8);
if (comp >= (+0.0f * -1.9777E35f)) {
  comp = ceilf(+1.3120E-41f);
comp += (+1.8972E-9f - +1.4179E-43f);
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
  float* tmp_6 = initPointer( atof(argv[6]) );
  float* tmp_7 = initPointer( atof(argv[7]) );
  float tmp_8 = atof(argv[8]);
  float tmp_9 = atof(argv[9]);

  compute<<<1,1>>>(tmp_1,tmp_2,tmp_3,tmp_4,tmp_5,tmp_6,tmp_7,tmp_8,tmp_9);
  cudaDeviceSynchronize();

  return 0;
}

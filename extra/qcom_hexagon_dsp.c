// main.c
#include <stdio.h>
#include <hexagon_standalone.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
 
#define STACK_SIZE 0x8000
 
int main(int argc, char* argv[]) {
    char *error;
 
    printf("Hexagon Tinygrad execution started\n");
 
    char *builtin[]={"libgcc.so", "libc.so", "libstdc++.so"};
    dlinit(3, builtin);
 
    //Load the shared object dynamically - Tinygrad Generated and Hexagon Compiled
    void *handle = dlopen("testprg.so", RTLD_LAZY);
 
    if (!handle) {
        fprintf(stderr, "Error loading shared object: %s\n", dlerror());
        return 1;
    }
 
    printf("Test Program so file loaded");
 
    if (!handle) {
        fputs (dlerror(), stderr);
        printf("dlopen error\n") ;
    }
 
    void (*r_2_2_3)() = dlsym(handle, "r_2_2_3");
    printf("Test Program func pointer loaded");
    if (((error = dlerror()) != NULL) ||
	(r_2_2_3 == NULL))
    {
        fputs(error, stderr);
    }
    else {
        int data1[2][3] = {{1,2,3},{4,5,6}};
        int data2[3][2] = {{7,8},{9,10},{11,12}};
        int data0[2][2] = {};
        r_2_2_3(data0, data1, data2);
        printf("Result\n");
        for(int i=0; i < 2; i++) {
            for(int j=0; j < 2; j++) {
                printf("%d\n", data2[i][j]);
            }
        }
        (*r_2_2_3)(data0, data1, data2);
    }
 
    // Unload the shared object
    dlclose(handle);
 
    return 0;
}

/* testprg.so file contents
#include<stdio.h>
//Tinygrad Generated
void r_2_2_3(int* restrict data0, const int* restrict data1, const int* restrict data2) {
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    int alu0 = (ridx0*3);
    int val0 = data1[alu0];
    int val1 = data1[alu0+1];
    int val2 = data1[alu0+2];
    for (int ridx1 = 0; ridx1 < 2; ridx1++) {
      int acc0 = 0;
      int val3 = data2[ridx1];
      int val4 = data2[ridx1+2];
      int val5 = data2[ridx1+4];
      data0[(ridx0*2)+ridx1] = ((val2*val5)+((val1*val4)+((val0*val3)+acc0)));
    }
  }
}


*/

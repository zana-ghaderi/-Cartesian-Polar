#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>

#define VERBOSE 0 
#define OFFSET (((float)1.0)/(1<<15))		// Offset that we will not care about the output
#define THRESHOLD 4	
#define NMAX 1000

struct test{
	void *array;
	int N;
	float theta;
	float R;
	void *result;
};
typedef struct test TESTCASE;

typedef signed short SHORT16;

FILE * results = fopen("Results.txt", "w");
float gR, gTheta;
int gN;

texture<unsigned int, 1, cudaReadModeElementType> tex;

unsigned int array0[NMAX][NMAX];
unsigned int array1[NMAX][NMAX];

TESTCASE test[2];		// Initialize basic sample test cases.  There will be more.

unsigned int Refresult[NMAX][NMAX];
unsigned int testin[NMAX][NMAX];
unsigned int result[NMAX][NMAX];

__host__ __device__ SHORT16 floatToSHORT16(float val){
	SHORT16 i;
	val=val*256;
	if (val>=0) {
		i=val+0.5;
	} else i=val-0.5;
	return i;
}

__host__ __device__ float SHORT16Tofloat(SHORT16 val) { return ((float)val)*(float)(1.0/256.0); }

unsigned int mychecksum(unsigned int sum, SHORT16 new1) {
	if (sum & 0x80000000) {
		sum<<=1;
		sum|=1;
	} else sum<<=1;

	sum=sum^new1;
	return sum;
}

__host__ __device__ SHORT16 refGetRPOL(void *base, int r, int t, int N){
	if(r<0||r>=N||t<0||t>=N)
		return -1;
	else {
		unsigned int *ptr=(unsigned int*)base+t*N+r;
		unsigned int val=*ptr;
		val=(val>>16)&0xffff;
		return val;
	}
}

__host__ __device__ SHORT16 refGetIPOL(void *base, int r, int t, int N){
	if(r<0||r>=N||t<0||t>=N)
		return -1;
	else {
		unsigned int *ptr=(unsigned int*)base+t*N+r;
		unsigned int val=*ptr;
		val=val&0xffff;
		return val;
	}
}

__host__ __device__ SHORT16 refGetRCART(void *base, int x, int y, int N){
	if(x<0||x>=N||y<0||y>=N)
		return -1;
	else {
		unsigned int *ptr=(unsigned int*)base+y*N+x;
		unsigned int val=*ptr;
		val=(val>>16)&0xffff;
		return val;
	}
}

__host__ __device__ SHORT16 refGetICART(void *base, int x, int y, int N){
	if(x<0||x>=N||y<0||y>=N)
		return -1;
	else {
		unsigned int *ptr=(unsigned int*)base+y*N+x;
		unsigned int val=*ptr;
		val=val&0xffff;
		return val;
	}
}

__host__ __device__ void refSetRPOL(void *base, int r , int t, int N, SHORT16 val){
	if(r>=0 && r<N && t>=0 && t<N){
		unsigned int *ptr=(unsigned int*)base+t*N+r;
		unsigned int ival=*ptr;
		SHORT16 check=refGetIPOL(base, r, t, N);

		ival=ival&0xffff;
		ival=ival|(val<<16);
		*ptr=ival;
	}
}

__host__ __device__ void refSetIPOL(void *base, int r , int t, int N, SHORT16 val){
	if(r>=0 && r<N && t>=0 && t<N){
		unsigned int *ptr=(unsigned int*)base+t*N+r;
		unsigned int ival=*ptr;
		SHORT16 check=refGetRPOL(base, r, t, N);

		ival=ival&0xffff0000;
		ival=ival|(0x0000ffff&val);
		*ptr=ival;
	}
}

__host__ __device__ void refSetRCART(void *base, int x , int y, int N, SHORT16 val){
	if(x>=0 && x<N && y>=0 && y<N){
		unsigned int *ptr=(unsigned int*)base+y*N+x;
		unsigned int ival=*ptr;
		SHORT16 check=refGetICART(base, x, y, N);

		ival=ival&0xffff;
		ival=ival|(val<<16);
		*ptr=ival;
	}
}

__host__ __device__ void refSetICART(void *base, int x , int y, int N, SHORT16 val)
{
	if(x>=0 && x<N && y>=0 && y<N){
		unsigned int *ptr=(unsigned int*)base+y*N+x;
		unsigned int ival=*ptr;
		SHORT16 check=refGetRCART(base, x, y, N);

		ival=ival&0xffff0000;
		ival=ival|(0x0000ffff&val);
		*ptr=ival;
	}
}

SHORT16 getRCART(void *base, int x, int y, int N)  { return refGetRCART(base, x, y, N);}
SHORT16 getICART(void *base, int x, int y, int N)  { return refGetICART(base, x, y, N);}
SHORT16 getRPOL	(void *base, int r, int t, int N)  { return refGetRPOL(base, r, t, N);}
SHORT16 getIPOL	(void *base, int r, int t, int N)  { return refGetIPOL(base, r, t, N);}

void setRCART (void *base, int x , int y, int N, SHORT16 val) { refSetRCART(base, x , y, N, val);}
void setICART (void *base, int x , int y, int N, SHORT16 val) { refSetICART(base, x , y, N, val);}
void setRPOL  (void *base, int r , int t, int N, SHORT16 val) {	refSetRPOL(base, r , t, N, val);}
void setIPOL  (void *base, int r , int t, int N, SHORT16 val) { refSetIPOL(base, r , t, N, val);}

void initTestCases(void) {
	int i,j;

	printf("Initialization can take some time on the XUP.... \n");

	for(i=0;i<2;i++) {
		test[i].R=gR;
		test[i].theta=gTheta;
	}

	test[0].N=gN;test[1].N=gN;
	test[0].array=(void*)array0;test[1].array=(void*)array1;

	for(i=0;i<gN;i++) {
		for(j=0;j<gN;j++) {
			refSetRCART((void*)array0, i, j, gN, floatToSHORT16(i*(float)(1/(float)(j+1))));
			refSetICART((void*)array0, i, j, gN, -floatToSHORT16(i*(float)(1/(float)(j+1))));
			refSetRCART((void*)array1, i, j, gN, floatToSHORT16(pow((float)1.05,(float)(i-j))));
			refSetICART((void*)array1, i, j, gN, -floatToSHORT16(pow((float)1.05,(float)(i-j))));
		}
	}
}

void refComputeCart2Pol(void *POL, void *CART, int N, float R, float theta) {
	int i,j,r,t;
	float dx,dy,x,y, real, imag;
	int lli,llj;

	/* delta in x and y direction*/
	dx=((R+1)-(R*(cos(theta))))/(N-1);
	dy=((R+1)*(sin(theta)))/(N-1);

	for(r=0;r<N;r++) {
		for(t=0;t<N;t++) {
			/*x and y value for given r and t*/
			x=(R+(float)r/(N-1))*cos(theta*(float)t/(N-1));
			y=(R+(float)r/(N-1))*sin(theta*(float)t/(N-1));

			lli=(int)((x-R*cos(theta))/dx);
			llj=(int)(y/dy);
            /*if(r == 125 && t == 120)
                 printf("ref: y = %f  dy = %f  y/dy = %f \n",y, dy, y/dy);*/
			if((lli==(N-1))&&(llj==(N-1))) {
				/* boundary case handling, not necessary */
				real = SHORT16Tofloat(refGetRCART(CART,N-1,N-1,N));
				imag = SHORT16Tofloat(refGetICART(CART,N-1,N-1,N));
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else if(lli==(N-1)) {
				/* boundary case handling, not necessary */
				real=    (SHORT16Tofloat(refGetRCART(CART, N-1, llj+1, N))+
					SHORT16Tofloat(refGetRCART(CART, N-1, llj, N)))/2;
				imag=	(SHORT16Tofloat(refGetICART(CART, N-1, llj+1, N))+
					SHORT16Tofloat(refGetICART(CART, N-1, llj, N)))/2;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else if(llj==(N-1)) {
				/* boundary case handling, not necessary */
				real=	(SHORT16Tofloat(refGetRCART(CART, lli, N-1, N))+
					SHORT16Tofloat(refGetRCART(CART, lli+1, N-1, N)))/2;
				imag=	(SHORT16Tofloat(refGetICART(CART, lli, N-1, N))+
					SHORT16Tofloat(refGetICART(CART, lli+1, N-1, N)))/2;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else {
				/*Compute the acerage value of the enclosing cart points*/
				real=	(SHORT16Tofloat(refGetRCART(CART, lli, llj+1, N))+
					SHORT16Tofloat(refGetRCART(CART, lli, llj, N))+
					SHORT16Tofloat(refGetRCART(CART, lli+1, llj+1, N))+
					SHORT16Tofloat(refGetRCART(CART, lli+1, llj, N)))/4;
				imag=	(SHORT16Tofloat(refGetICART(CART, lli, llj+1, N))+
					SHORT16Tofloat(refGetICART(CART, lli, llj, N))+
					SHORT16Tofloat(refGetICART(CART, lli+1, llj+1, N))+
					SHORT16Tofloat(refGetICART(CART, lli+1, llj, N)))/4;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			}
		}
	}
}

void CPU_ComputeCart2Pol(void *POL, void *CART, int N, float R, float theta){
  int i,j,r,t;
  float dx,dy,x,y, cos_val, sin_val, temp1, temp2, real, imag;
  int lli,llj;

  /* delta in x and y direction*/
  dx=((R+1)-(R*(cos(theta))))/(N-1);
  dy=((R+1)*(sin(theta)))/(N-1);
  temp2 = R*cos(theta);

  for(t=0;t<N;t++) {
    cos_val = cos(theta*(float)t/(N-1));
    sin_val = sin(theta*(float)t/(N-1));
    for(r=0;r<N;r++) {
      /*x and y value for given r and t*/
      temp1 = R+(float)r/(N-1);
      x = temp1*cos_val;
      y = temp1*sin_val;

      lli=(int)((x-temp2)/dx);
      llj=(int)(y/dy);
      
      if((lli==(N-1))&&(llj==(N-1))) {
	/* boundary case handling, not necessary */
	real = SHORT16Tofloat(refGetRCART(CART,N-1,N-1,N));
	imag = SHORT16Tofloat(refGetICART(CART,N-1,N-1,N));
	refSetRPOL(POL, r, t, N, floatToSHORT16(real));
	refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
      } else if(lli==(N-1)) {
	/* boundary case handling, not necessary */
	real=    (SHORT16Tofloat(refGetRCART(CART, N-1, llj+1, N))+
			 SHORT16Tofloat(refGetRCART(CART, N-1, llj, N)))/2;
	imag=	(SHORT16Tofloat(refGetICART(CART, N-1, llj+1, N))+
			 SHORT16Tofloat(refGetICART(CART, N-1, llj, N)))/2;
	refSetRPOL(POL, r, t, N, floatToSHORT16(real));
	refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
      } else if(llj==(N-1)) {
	/* boundary case handling, not necessary */
	real=	(SHORT16Tofloat(refGetRCART(CART, lli, N-1, N))+
			 SHORT16Tofloat(refGetRCART(CART, lli+1, N-1, N)))/2;
	imag=	(SHORT16Tofloat(refGetICART(CART, lli, N-1, N))+
			 SHORT16Tofloat(refGetICART(CART, lli+1, N-1, N)))/2;
	refSetRPOL(POL, r, t, N, floatToSHORT16(real));
	refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
      } else {
	/*Compute the acerage value of the enclosing cart points*/
	real=	(SHORT16Tofloat(refGetRCART(CART, lli, llj+1, N))+
			 SHORT16Tofloat(refGetRCART(CART, lli, llj, N))+
			 SHORT16Tofloat(refGetRCART(CART, lli+1, llj+1, N))+
			 SHORT16Tofloat(refGetRCART(CART, lli+1, llj, N)))/4;
	imag=	(SHORT16Tofloat(refGetICART(CART, lli, llj+1, N))+
			 SHORT16Tofloat(refGetICART(CART, lli, llj, N))+
			 SHORT16Tofloat(refGetICART(CART, lli+1, llj+1, N))+
			 SHORT16Tofloat(refGetICART(CART, lli+1, llj, N)))/4;
	refSetRPOL(POL, r, t, N, floatToSHORT16(real));
	refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
      }
    }
  }
}

__global__ void Kernel(void *POL, void *CART, int N, float R, float theta){
    
    int t = blockIdx.x;
    int r=blockIdx.y*128+threadIdx.x;
    int lli = tex1Dfetch(tex, (t*N+r)*2);
    int llj = tex1Dfetch(tex, (t*N+r)*2+1);

      if((lli==(N-1))&&(llj==(N-1))) {
	/* boundary case handling, not necessary */
	float real = SHORT16Tofloat(refGetRCART(CART,N-1,N-1,N));
	float imag = SHORT16Tofloat(refGetICART(CART,N-1,N-1,N));
	refSetRPOL(POL, r, t, N, floatToSHORT16(real));
	refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
      } else if(lli==(N-1)) {
	/* boundary case handling, not necessary */
	float real=    (SHORT16Tofloat(refGetRCART(CART, N-1, llj+1, N))+
			 SHORT16Tofloat(refGetRCART(CART, N-1, llj, N)))/2;
	float imag=	(SHORT16Tofloat(refGetICART(CART, N-1, llj+1, N))+
			 SHORT16Tofloat(refGetICART(CART, N-1, llj, N)))/2;
	refSetRPOL(POL, r, t, N, floatToSHORT16(real));
	refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
      } else if(llj==(N-1)) {
	/* boundary case handling, not necessary */
	float real=	(SHORT16Tofloat(refGetRCART(CART, lli, N-1, N))+
			 SHORT16Tofloat(refGetRCART(CART, lli+1, N-1, N)))/2;
	float imag=	(SHORT16Tofloat(refGetICART(CART, lli, N-1, N))+
			 SHORT16Tofloat(refGetICART(CART, lli+1, N-1, N)))/2;
	refSetRPOL(POL, r, t, N, floatToSHORT16(real));
	refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
      } else {
	/*Compute the acerage value of the enclosing cart points*/
	float real=	(SHORT16Tofloat(refGetRCART(CART, lli, llj+1, N))+
			 SHORT16Tofloat(refGetRCART(CART, lli, llj, N))+
			 SHORT16Tofloat(refGetRCART(CART, lli+1, llj+1, N))+
			 SHORT16Tofloat(refGetRCART(CART, lli+1, llj, N)))/4;
	float imag=	(SHORT16Tofloat(refGetICART(CART, lli, llj+1, N))+
			 SHORT16Tofloat(refGetICART(CART, lli, llj, N))+
			 SHORT16Tofloat(refGetICART(CART, lli+1, llj+1, N))+
			 SHORT16Tofloat(refGetICART(CART, lli+1, llj, N)))/4;
	refSetRPOL(POL, r, t, N, floatToSHORT16(real));
	refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
    }     
}

void callComputeCart2Pol(void *POL, void *CART, int N, float R, float theta){

  int i,j, t, r;
  float dx,dy,x,y,cos_val,sin_val,temp1,temp2;
  int *indexes = (int*)malloc(2*N*N*sizeof(int));
  int* mem;
  struct timeval tempo1, tempo2;
  
  /* delta in x and y direction*/
  dx=((R+1)-(R*(cos(theta))))/(N-1);
  dy=((R+1)*(sin(theta)))/(N-1);

  size_t arraySize = N*N*sizeof(unsigned int);
  unsigned int* CART_d; 
  unsigned int* POL_d;
  unsigned int* indexes_d;
  
  int blockSize = 128;
  //int nblocks;
  int gridX,gridY;
  if(N%blockSize == 0)
         gridY = N/blockSize;
  else
         gridY = N/blockSize + 1;
         
         gridX = N;

  dim3 dimGrid ( gridX,gridY);
  
    // allocate arrays on device
    cudaMalloc((void **) &mem, 2*N*N*sizeof(int));
    cudaMalloc((void **) &CART_d, arraySize);
    cudaMalloc((void **) &POL_d, arraySize);
   
    // Set texture parameters (default)
    tex.normalized = false; // do not normalize coordinates   
    
    //memcpy global CART
    gettimeofday(&tempo1, NULL);
    
    temp2 = R*cos(theta);
    for(t=0;t<N;t++){
         cos_val = cos(theta*(float)t/(N-1));
         sin_val = sin(theta*(float)t/(N-1));
         for(r=0;r<N;r++) {
               temp1 = R+(float)r/(N-1);
               x = temp1*cos_val;
               y = temp1*sin_val;
               indexes[(t*N+r)*2]=(int)((x-temp2)/dx);
               indexes[(t*N+r)*2+1]=(int)(y/dy);
         }
    }
    cudaMemcpy(CART_d, CART, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(mem, indexes, 2*N*N*sizeof(int), cudaMemcpyHostToDevice );
    // Bind mem to the texture
    cudaBindTexture(0, tex, mem, 2*N*N*sizeof(int));
    
      //kernel call
      Kernel<<<dimGrid,blockSize>>>(POL_d, CART_d, N, R, theta);
      
    //memcpy device host POL 
   cudaMemcpy(POL, POL_d, arraySize, cudaMemcpyDeviceToHost);
   gettimeofday(&tempo2, NULL);
   printf("GPU Elapsed Time: %d us \n", tempo2.tv_usec - tempo1.tv_usec + 1000000*(tempo2.tv_sec - tempo1.tv_sec));
   fprintf(results, "GPU Elapsed Time: %d us \n", tempo2.tv_usec - tempo1.tv_usec + 1000000*(tempo2.tv_sec - tempo1.tv_sec));
   
    cudaFree(CART_d);
    cudaFree(POL_d);
    cudaUnbindTexture(tex);
}

void computeCart2Pol(void *POL, void *CART, int N, float R, float theta)
{
  if(N<=15){
           struct timeval tempo1, tempo2;
		   gettimeofday(&tempo1, NULL);
			
			CPU_ComputeCart2Pol(POL, CART, N, R, theta);
			
			gettimeofday(&tempo2, NULL);
			printf("CPU Elapsed Time: %d us \n", tempo2.tv_usec - tempo1.tv_usec + 1000000*(tempo2.tv_sec - tempo1.tv_sec));
			fprintf(results, "CPU Elapsed Time: %d us \n", tempo2.tv_usec - tempo1.tv_usec + 1000000*(tempo2.tv_sec - tempo1.tv_sec));
			FILE * outfile = fopen("POL_CPU.txt", "w");
	        for(int i=0; i<N; i++)
                     for(int j=0; j<N; j++)
                              fprintf(outfile, "real(%d, %d): %f imag(%d, %d): %f\n",i, j, i, j, SHORT16Tofloat(refGetRPOL(POL, i, j, N)), SHORT16Tofloat(refGetIPOL(POL, i, j, N)));      

           }
  else
           callComputeCart2Pol(POL, CART, N, R, theta);         
}

void testCart2pol(int index) {
	FILE * outfile;
    unsigned int refsum=0, ressum=0;
    long long start, finish;
	int okay=1;
	initTestCases();

		printf("****************************************\n");
		printf("Begin Test %d N=%d, R=%d/1000000, theta=%d/1000000\n", index, test[index].N, (int)(test[index].R*1000000), (int)(test[index].theta*1000000));
		{
			struct timeval tempo1, tempo2;
			
            gettimeofday(&tempo1, NULL);		

			/* compute "ground truth" using float precision reference implementation. */
			refComputeCart2Pol((void *)Refresult, (void *)test[index].array, test[index].N, test[index].R, test[index].theta);
			
            gettimeofday(&tempo2, NULL);
			printf("Ref Code Elapsed Time: %d us \n", tempo2.tv_usec - tempo1.tv_usec + 1000000*(tempo2.tv_sec - tempo1.tv_sec));
			fprintf(results, "Ref Code Elapsed Time: %d us \n", tempo2.tv_usec - tempo1.tv_usec + 1000000*(tempo2.tv_sec - tempo1.tv_sec));
			outfile = fopen("POL_ref.txt", "w");
	        for(int i=0; i<test[index].N; i++)
                     for(int j=0; j<test[index].N; j++)
                              fprintf(outfile, "real(%d, %d): %f imag(%d, %d): %f\n",i, j, i, j, SHORT16Tofloat(refGetRPOL(Refresult, i, j, test[index].N)), SHORT16Tofloat(refGetIPOL(Refresult, i, j, test[index].N)));    
          }   

		{
			int i,j;
			int N=test[index].N;

			for(i=0;i<N;i++) {
				for(j=0;j<N;j++) {
					setRCART((void *)testin, i, j, N, refGetRCART(test[index].array, i, j, N));
					setICART((void *)testin, i, j, N, refGetICART(test[index].array, i, j, N));
				}
			}
		}
        computeCart2Pol((void *)result, (void *)testin, test[index].N, test[index].R, test[index].theta);
        outfile = fopen("POL_GPU.txt", "w");
        for(int i=0; i<test[index].N; i++)
                for(int j=0; j<test[index].N; j++)
                        fprintf(outfile, "real(%d, %d): %f imag(%d, %d): %f\n",i, j, i, j, SHORT16Tofloat(refGetRPOL(result, i, j, test[index].N)), SHORT16Tofloat(refGetIPOL(result, i, j, test[index].N)));             
        {
			double dx=((test[index].R+1)-(test[index].R*(cos((double)test[index].theta))))/(test[index].N-1);
			double dy=((test[index].R+1)*(sin((double)test[index].theta)))/(test[index].N-1);
			int i, j, thisokay=1;

			printf("\tChecking for correctness\n");

			/* check your answer against the ground truth */
			for(i=1;i<(test[index].N-1);i++) {
				for(j=1;j<(test[index].N-1);j++) {
					double posx=(test[index].R+(double)i/(test[index].N-1))*cos(test[index].theta*(double)j/(test[index].N-1));
					double posy=(test[index].R+(double)i/(test[index].N-1))*sin(test[index].theta*(double)j/(test[index].N-1));
					int omit = 	(((int)(posy/dy)+1) < (posy/dy + OFFSET))||
						(((int)(posy/dy)) > (posy/dy - OFFSET))||
						(((int)((posx-test[index].R*cos(test[index].theta))/dx)+1) < ((posx-test[index].R*cos(test[index].theta))/dx + OFFSET))||
						(((int)((posx-test[index].R*cos(test[index].theta))/dx)) > ((posx-test[index].R*cos(test[index].theta))/dx - OFFSET));

					if(VERBOSE) {
						SHORT16 ref,res;
						ref=refGetRPOL(Refresult,i,j,test[index].N);
						res=getRPOL(result,i,j,test[index].N);
						printf("\t**the real value at (%d, %d) differs by %d(SHORT16 format), %x, %x\n",i,j,ref-res, ref, res);

						ref=refGetIPOL(Refresult,i,j,test[index].N);
						res=getIPOL(result,i,j,test[index].N);
						printf("\t**the imaginary value at (%d, %d) differs by %d(SHORT16 format) %x, %x\n",i,j,ref-res, ref, res);
					} else {
						SHORT16 refRPol=refGetRPOL(Refresult,i,j,test[index].N);
						SHORT16 resRPol=getRPOL(result,i,j,test[index].N);
						SHORT16 refIPol=refGetIPOL(Refresult,i,j,test[index].N);
						SHORT16 resIPol=getIPOL(result,i,j,test[index].N);

						SHORT16 diffRPol=refRPol-resRPol;
						SHORT16 diffIPol=refIPol-resIPol;

						if (!omit) {
							refsum=mychecksum(refsum,refRPol);
							refsum=mychecksum(refsum,refIPol);
							ressum=mychecksum(ressum,resRPol);
							ressum=mychecksum(ressum,resIPol);
						}

						if((diffRPol>THRESHOLD)||(diffRPol<(-THRESHOLD))) {
							printf("\t**%s The real value at (%d, %d) differs by %d(SHORT16 format)\n",omit?"(okay)":"(not okay)",i,j,diffRPol);
							if (!omit) {
								okay=0;
								thisokay=0;
							}
						}
						if((diffIPol>THRESHOLD)||(diffIPol<(-THRESHOLD))) {
							printf("\t**%s The imaginary value at (%d, %d) differs by %d i(SHORT16 format)\n",omit?"(okay)":"(not okay)",i,j,diffIPol);
							if (!omit) { okay=0; thisokay=0; }
						}
					} // else
				} // for(j)
			} // for(i)
			printf("Test%d %s refsum=%x; ressum=%x\n", index, thisokay?"Passed":"Failed", refsum, ressum);
		}
		if (okay) 
		printf("All okay!!!\n");
        else
		printf("Some test(s) failed!!!\n");
}

int main (void) {
    
	gR = 46.416, gTheta = 0.0627, gN = 937;
	testCart2pol(1);
	return(0);
}


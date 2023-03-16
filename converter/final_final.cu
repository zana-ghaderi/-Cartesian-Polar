#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <sys/time.h>
#define myassert(A,B) 
#define VERBOSE 0 
#define OFFSET (((double)1.0)/(1<<15))		// Offset that we will not care about the output
#define THRESHOLD 4	
#define NMAX 1000
#define NMIN 10

using namespace std;
////////////////////////////////////////////////////////////////////////////////////////////////
struct test{
	void *array;
	int N;
	double theta;
	double R;
	void *result;
};

typedef signed short SHORT16;
typedef struct test TESTCASE;


double gR=10, gTheta=0.201358,gN=700;



unsigned int array0[NMAX][NMAX];
unsigned int array1[NMAX][NMAX];
unsigned int Refresult[NMAX][NMAX];
unsigned int testin[NMAX][NMAX];
unsigned int result[NMAX][NMAX];
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int mychecksum(unsigned int sum, SHORT16 newnm) {
  if (sum&0x80000000) {
    sum<<=1;
    sum|=1;
  } else {
    sum<<=1;
  }

  sum=sum^newnm;

  return sum;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ float power(int x,int y)
 {
	 float r=1;
	 float l;
	 for(int i=0;i<abs(y);i++)
		 r=r*x;
	 if(y>=0)
	 return r;
	 else
	 {
	  l=1/r;
     return l;
     }
 }
///////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ SHORT16 doubleToSHORT16(double val){
	SHORT16 i;
	val=val*power(2,8);
	if (val>=0) {
		i=val+0.5;
	} else i=val-0.5;
	return i;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ SHORT16 floatToSHORT16(float val){
	SHORT16 i;
	val=val*(256);
	if (val>=0) {
		i=val+0.5;
	} else i=val-0.5;
	return i;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ double SHORT16ToDouble(SHORT16 val) { return ((double)val)*power(2,-8); }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////////////////////
__host__ __device__ SHORT16 getRCART    (void *base, int x, int y, int N)  { return refGetRCART(base, x, y, N);}
__host__ __device__ SHORT16 getICART   (void *base, int x, int y, int N)  { return refGetICART(base, x, y, N);}
__host__ __device__ SHORT16 getRPOL	(void *base, int r, int t, int N)  { return refGetRPOL(base, r, t, N);}
__host__ __device__ SHORT16 getIPOL	(void *base, int r, int t, int N)  { return refGetIPOL(base, r, t, N);}

__host__ __device__ void setRCART (void *base, int x , int y, int N, SHORT16 val) { refSetRCART(base, x , y, N, val);}
__host__ __device__ void setICART (void *base, int x , int y, int N, SHORT16 val) { refSetICART(base, x , y, N, val);}
__host__ __device__ void setRPOL  (void *base, int r , int t, int N, SHORT16 val) {	refSetRPOL(base, r , t, N, val);}
__host__ __device__ void setIPOL  (void *base, int r , int t, int N, SHORT16 val) { refSetIPOL(base, r , t, N, val);}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////__host__  void initTestCases(void) {

	int i,j;

	printf("Initialization can take some time on the XUP.... \n");

	test[0].N=10;
	test[0].R=10.0;
	test[0].theta=0.100167;
	
	test[1].N=700;
	test[1].R=10.0;
	test[1].theta=0.100167;
	
	test[2].N=1000;
	test[2].R=10.0;
	test[2].theta=0.100167;
	
	test[3].N=10;
	test[3].R=100.0;
	test[3].theta=0.0123003;
	
	test[4].N=700;
	test[4].R=100.0;
	test[4].theta=0.0123003;
	
	test[5].N=1000;
	test[5].R=100.0;
	test[5].theta=0.0123003;
	
	test[6].N=10;
	test[6].R=100.0;
	test[6].theta=0.201358;
	
	test[7].N=700;
	test[7].R=100.0;
	test[7].theta=0.201358;
	
	test[8].N=1000;
	test[8].R=100.0;
	test[8].theta=0.201358;
	
	test[9].N=13;
	test[9].R=46.416;
	test[9].theta=0.0627 ;
	
	test[10].N=683;
	test[10].R=46.416;
	test[10].theta=0.0627 ;
	
	test[11].N=937;
	test[11].R=46.416;
	test[11].theta=0.0627 ;

	for(i=0;i<gN;i++) {
		for(j=0;j<gN;j++) {
			
			refSetRCART((void*)array0, i, j, gN, floatToSHORT16(i*(float)(1/(float)(j+1))));
			refSetICART((void*)array0, i, j, gN, -floatToSHORT16(i*(float)(1/(float)(j+1))));
			refSetRCART((void*)array1, i, j, gN, floatToSHORT16(pow((float)1.05,(float)(i-j))));
			refSetICART((void*)array1, i, j, gN, -floatToSHORT16(pow((float)1.05,(float)(i-j))));
		}
	}
	
	test[0].array=(void*)array0;test[1].array=(void*)array1;
	test[2].array=(void*)array0;test[3].array=(void*)array1;
	test[4].array=(void*)array0;test[5].array=(void*)array1;
	test[6].array=(void*)array0;test[7].array=(void*)array1;
	test[8].array=(void*)array0;test[9].array=(void*)array1;
	test[10].array=(void*)array0;test[11].array=(void*)array1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void refComputeCart2Pol(void *POL, void *CART, int N, double R, double theta) {
	int r,t;
	float dx,dy,x,y;
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
			
			//printf("r=%d,t=%d,lli=%d,llj=%d",r,t,lli,llj);

			if((lli==(N-1))&&(llj==(N-1))) {
				/* boundary case handling, not necessary */
				double real = SHORT16ToDouble(refGetRCART(CART,N-1,N-1,N));
				double imag = SHORT16ToDouble(refGetICART(CART,N-1,N-1,N));
				refSetRPOL(POL, r, t, N, doubleToSHORT16(real));
				refSetIPOL(POL, r, t, N, doubleToSHORT16(imag));
			} else if(lli==(N-1)) {
				/* boundary case handling, not necessary */
				double real=    (SHORT16ToDouble(refGetRCART(CART, N-1, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, N-1, llj, N)))/2;
				double imag=	(SHORT16ToDouble(refGetICART(CART, N-1, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, N-1, llj, N)))/2;
				refSetRPOL(POL, r, t, N, doubleToSHORT16(real));
				refSetIPOL(POL, r, t, N, doubleToSHORT16(imag));
			} else if(llj==(N-1)) {
				/* boundary case handling, not necessary */
				double real=	(SHORT16ToDouble(refGetRCART(CART, lli, N-1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, N-1, N)))/2;
				double imag=	(SHORT16ToDouble(refGetICART(CART, lli, N-1, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, N-1, N)))/2;
				refSetRPOL(POL, r, t, N, doubleToSHORT16(real));
				refSetIPOL(POL, r, t, N, doubleToSHORT16(imag));
			} else {
				/*Compute the acerage value of the enclosing cart points*/
				double real=	(SHORT16ToDouble(refGetRCART(CART, lli, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli, llj, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, llj, N)))/4;
				double imag=	(SHORT16ToDouble(refGetICART(CART, lli, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, lli, llj, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, llj, N)))/4;
				refSetRPOL(POL, r, t, N, doubleToSHORT16(real));
				refSetIPOL(POL, r, t, N, doubleToSHORT16(imag));
			}
		}
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeCart2Pol(void *POL, void *CART, int N, double R, double theta) {
	int r,t;
	double dx,dy,x,y;
	int lli,llj;

	/* delta in x and y direction*/
	double mm=(R+1);
	dx=(mm-(R*(cos(theta))))/(N-1);
	dy=(mm*(sin(theta)))/(N-1);
	
   r=blockIdx.x * blockDim.x + threadIdx.x;
   t=blockIdx.y * blockDim.y + threadIdx.y;
   
   //printf("r=%d,t=%d,lli=%d,llj=%d",r,t,lli,llj);

			/*x and y value for given r and t*/
			x=(R+(double)r/(N-1))*cos(theta*(double)t/(N-1));
			y=(R+(double)r/(N-1))*sin(theta*(double)t/(N-1));

			lli=(int)((x-R*cos(theta))/dx);
			llj=(int)(y/dy);

			if((lli==(N-1))&&(llj==(N-1))) {
				/* boundary case handling, not necessary */
				double real = SHORT16ToDouble(refGetRCART(CART,N-1,N-1,N));
				double imag = SHORT16ToDouble(refGetICART(CART,N-1,N-1,N));
				refSetRPOL(POL, r, t, N, doubleToSHORT16(real));
				refSetIPOL(POL, r, t, N, doubleToSHORT16(imag));
			} else if(lli==(N-1)) {
				/* boundary case handling, not necessary */
				double real=    (SHORT16ToDouble(refGetRCART(CART, N-1, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, N-1, llj, N)))/2;
				double imag=	(SHORT16ToDouble(refGetICART(CART, N-1, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, N-1, llj, N)))/2;
				refSetRPOL(POL, r, t, N, doubleToSHORT16(real));
				refSetIPOL(POL, r, t, N, doubleToSHORT16(imag));
			} else if(llj==(N-1)) {
				/* boundary case handling, not necessary */
				double real=	(SHORT16ToDouble(refGetRCART(CART, lli, N-1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, N-1, N)))/2;
				double imag=	(SHORT16ToDouble(refGetICART(CART, lli, N-1, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, N-1, N)))/2;
				refSetRPOL(POL, r, t, N, doubleToSHORT16(real));
				refSetIPOL(POL, r, t, N, doubleToSHORT16(imag));
			} else {
				/*Compute the acerage value of the enclosing cart points*/
				double real=	(SHORT16ToDouble(refGetRCART(CART, lli, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli, llj, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, llj, N)))/4;
				double imag=	(SHORT16ToDouble(refGetICART(CART, lli, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, lli, llj, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, llj, N)))/4;
				refSetRPOL(POL, r, t, N, doubleToSHORT16(real));
				refSetIPOL(POL, r, t, N, doubleToSHORT16(imag));
			}
		}

///////////////////////////////////////////////////////////////////////////////////////////////
void testCart2pol() {
	long long start, finish;
	int okay=1, testnum;

	initTestCases();

	for(testnum=0;testnum<12;testnum++) {
		unsigned int refsum=0, ressum=0;

		printf("****************************************\n");
		printf("Begin Test %d N=%d, R=%d/1000000, theta=%d/1000000\n",
			testnum, test[testnum].N, (int)(test[testnum].R*1000000), (int)(test[testnum].theta*1000000));
		{
		    struct timeval tempo1, tempo2;
            gettimeofday(&tempo1, NULL);
            
			/* compute "ground truth" using double precision reference implementation. */
			refComputeCart2Pol((void *)Refresult, (void *)test[testnum].array, test[testnum].N, test[testnum].R, test[testnum].theta);
        
          gettimeofday(&tempo2, NULL);
          printf(" Elapsed Time in CPU = %d us \n", tempo2.tv_usec - tempo1.tv_usec + 1000000*(tempo2.tv_sec - tempo1.tv_sec));
          
          FILE * outfile;

        //printf the cpu results
         outfile = fopen("POL_ref.txt", "w");
            for(int i=0; i<test[testnum].N; i++)
                     for(int j=0; j<test[testnum].N; j++)
                              fprintf(outfile, "real(%d, %d): %f imag(%d, %d): %f\n",i, j,
                               SHORT16ToDouble(refGetRPOL(Refresult, i, j, test[testnum].N)), i, j,
                                SHORT16ToDouble(refGetIPOL(Refresult, i, j, test[testnum].N)));    
		}

		{
			/*int i,j;
			int N=test[testnum].N;

			for(i=0;i<N;i++) {
				for(j=0;j<N;j++) {
					setRCART((void *)testin, i, j, N,refGetRCART(test[testnum].array, i, j, N));
					setICART((void *)testin, i, j, N,refGetICART(test[testnum].array, i, j, N));
				}
			}*/
		}

		{	/* run and time our implementation */
		    int N=test[testnum].N;
            int size=N*N*sizeof(unsigned int);
            void *POLd;
	        void *CARTd;
            	
		    
         	cudaMalloc(&CARTd,size);
	        cudaMalloc(&POLd,size);
	        
	        dim3 dimBlock(16,16);
            dim3 dimGrid((N -1 + dimBlock.x )/dimBlock.x,(N -1 + dimBlock.y )/dimBlock.y);
	        
			/*clock_t start;
			clock_t end;
			start = clock();*/
		    struct timeval tempo1, tempo2;
            gettimeofday(&tempo1, NULL);
			
	       cudaMemcpy(CARTd,(void *)test[testnum].array,size,cudaMemcpyHostToDevice);

           ComputeCart2Pol<<<dimGrid,dimBlock>>>(POLd,CARTd,N,test[testnum].R, test[testnum].theta);
	       
	       cudaMemcpy(result,POLd,size,cudaMemcpyDeviceToHost);
           gettimeofday(&tempo2, NULL);
           printf(" Elapsed Time in GPU = %d us \n", tempo2.tv_usec - tempo1.tv_usec + 1000000*(tempo2.tv_sec - tempo1.tv_sec));
           
           //printf the GPU results
           FILE * outfile;
            outfile = fopen("POL_GPU.txt", "w");
            for(int i=0; i<test[testnum].N; i++)
                     for(int j=0; j<test[testnum].N; j++)
                              fprintf(outfile, "real(%d, %d): %f imag(%d, %d): %f\n",i, j, 
                               SHORT16ToDouble(refGetRPOL(result, i, j, test[testnum].N)),i, j,
                                SHORT16ToDouble(refGetIPOL(result, i, j, test[testnum].N)));
	       cudaFree(POLd);
           cudaFree(CARTd);
           
		}

		{
			double dx=((test[testnum].R+1)-(test[testnum].R*(cos((double)test[testnum].theta))))/(test[testnum].N-1);
			double dy=((test[testnum].R+1)*(sin((double)test[testnum].theta)))/(test[testnum].N-1);
			int i, j, thisokay=1;

			printf("\tChecking for correctness\n");

			/* check your answer against the ground truth */
			for(i=1;i<(test[testnum].N-1);i++) {
				for(j=1;j<(test[testnum].N-1);j++) {
					double posx=(test[testnum].R+(double)i/(test[testnum].N-1))*cos(test[testnum].theta*(double)j/(test[testnum].N-1));
					double posy=(test[testnum].R+(double)i/(test[testnum].N-1))*sin(test[testnum].theta*(double)j/(test[testnum].N-1));
					int omit = 	(((int)(posy/dy)+1) < (posy/dy + OFFSET))||
						(((int)(posy/dy)) > (posy/dy - OFFSET))||
						(((int)((posx-test[testnum].R*cos(test[testnum].theta))/dx)+1) < ((posx-test[testnum].R*cos(test[testnum].theta))/dx + OFFSET))||
						(((int)((posx-test[testnum].R*cos(test[testnum].theta))/dx)) > ((posx-test[testnum].R*cos(test[testnum].theta))/dx - OFFSET));

					if(VERBOSE) {
						SHORT16 ref,res;
						ref=refGetRPOL(Refresult,i,j,test[testnum].N);
						res=getRPOL(result,i,j,test[testnum].N);
						//printf("\t**the real value at (%d, %d) differs by %d(SHORT16 format), %x, %x\n",i,j,ref-res, ref, res);

						ref=refGetIPOL(Refresult,i,j,test[testnum].N);
						res=getIPOL(result,i,j,test[testnum].N);
						//printf("\t**the imaginary value at (%d, %d) differs by %d(SHORT16 format) %x, %x\n",i,j,ref-res, ref, res);
					} else {
						SHORT16 refRPol=refGetRPOL(Refresult,i,j,test[testnum].N);
						SHORT16 resRPol=getRPOL(result,i,j,test[testnum].N);
						SHORT16 refIPol=refGetIPOL(Refresult,i,j,test[testnum].N);
						SHORT16 resIPol=getIPOL(result,i,j,test[testnum].N);

						SHORT16 diffRPol=refRPol-resRPol;
						SHORT16 diffIPol=refIPol-resIPol;

						if (!omit) {
							refsum=mychecksum(refsum,refRPol);
							refsum=mychecksum(refsum,refIPol);
							ressum=mychecksum(ressum,resRPol);
							ressum=mychecksum(ressum,resIPol);
						}

						if((diffRPol>THRESHOLD)||(diffRPol<(-THRESHOLD))) {
							//printf("\t**%s The real value at (%d, %d) differs by %d(SHORT16 format)\n",omit?"(okay)":"(not okay)",i,j,diffRPol);
							if (!omit) {
								okay=0;
								thisokay=0;
							}
						}
						if((diffIPol>THRESHOLD)||(diffIPol<(-THRESHOLD))) {
						//	printf("\t**%s The imaginary value at (%d, %d) differs by %d i(SHORT16 format)\n",omit?"(okay)":"(not okay)",i,j,diffIPol);
							if (!omit) { okay=0; thisokay=0; }
						}
					} // else
				} // for(j)
			} // for(i)
			printf("Test%d %s refsum=%x; ressum=%x\n", testnum, thisokay?"Passed":"Failed", refsum, ressum);
		}

	} // for(testnum)

	if (okay) 
		printf("All okay!!!\n");
	else
		printf("Some test(s) failed!!!\n");
}
////////////////////////////////////////////////////////////////////////////////////////////////////////
int main( )
{
    
    testCart2pol();
    system("PAUSE");
    return EXIT_SUCCESS;
}






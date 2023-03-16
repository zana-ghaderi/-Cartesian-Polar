#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <iostream>
using namespace std;

#define VERBOSE 0
#define OFFSET (((float)1.0)/(1<<15))		// Offset that we will not care about the output
#define THRESHOLD 4	
#define NMAX 1000

struct test{
	void *arrayy;
	int N;
	double theta;
	double R;
	void *result;
};
typedef struct test TESTCASE;
typedef signed short SHORT16;

float gR=100.0, gTheta=0.5, gN=1000.0;

unsigned int array0[NMAX][NMAX];
unsigned int array1[NMAX][NMAX];
unsigned int Refresult[NMAX][NMAX];
unsigned int result[NMAX][NMAX];

TESTCASE test[12];		// Initialize basic sample test cases.  There will be more.

__host__  __device__  SHORT16  floatToSHORT16(float val){
	SHORT16 i;
	val=val*256;//pow(2,8);
	if (val>=0) {
		i=val+0.5;
	} else i=val-0.5;
	return i;
}

__host__  __device__  float SHORT16ToDouble(SHORT16 val) { return ((float)val)*0.00390625;}//pow(2,-8); }

__host__  __device__  unsigned int mychecksum(unsigned int sum, SHORT16 neww) {
	if (sum & 0x80000000) {
		sum<<=1;
		sum|=1;
	} else sum<<=1;

	sum=sum^neww;
	return sum;
}

__host__  __device__  SHORT16 refGetRPOL(void *base, int r, int t, int N){
	if(r<0||r>=N||t<0||t>=N)
		return -1;
	else {
		unsigned int *ptr=(unsigned int*)base+t*N+r;
		unsigned int val=*ptr;
		val=(val>>16)&0xffff;
		return val;
	}
}

__host__  __device__  SHORT16 refGetIPOL(void *base, int r, int t, int N){
	if(r<0||r>=N||t<0||t>=N)
		return -1;
	else {
		unsigned int *ptr=(unsigned int*)base+t*N+r;
		unsigned int val=*ptr;
		val=val&0xffff;
		return val;
	}
}

__host__  __device__  SHORT16 refGetRCART(void *base, int x, int y, int N){
	if(x<0||x>=N||y<0||y>=N)
		return -1;
	else {
		unsigned int *ptr=(unsigned int*)base+y*N+x;
		unsigned int val=*ptr;
		val=(val>>16)&0xffff;
		return val;
	}
}

__host__  __device__  SHORT16 refGetICART(void *base, int x, int y, int N){
	if(x<0||x>=N||y<0||y>=N)
		return -1;
	else {
		unsigned int *ptr=(unsigned int*)base+y*N+x;
		unsigned int val=*ptr;
		val=val&0xffff;
		return val;
	}
}

__host__  __device__  void refSetRPOL(void *base, int r , int t, int N, SHORT16 val){
	if(r>=0 && r<N && t>=0 && t<N){
		unsigned int *ptr=(unsigned int*)base+t*N+r;
		unsigned int ival=*ptr;
		SHORT16 check=refGetIPOL(base, r, t, N);

		ival=ival&0xffff;
		ival=ival|(val<<16);
		*ptr=ival;
	}
}

__host__  __device__  void refSetIPOL(void *base, int r , int t, int N, SHORT16 val){
	if(r>=0 && r<N && t>=0 && t<N){
		unsigned int *ptr=(unsigned int*)base+t*N+r;
		unsigned int ival=*ptr;
		SHORT16 check=refGetRPOL(base, r, t, N);

		ival=ival&0xffff0000;
		ival=ival|(0x0000ffff&val);
		*ptr=ival;
	}
}

__host__  __device__  void refSetRCART(void *base, int x , int y, int N, SHORT16 val){
	if(x>=0 && x<N && y>=0 && y<N){
		unsigned int *ptr=(unsigned int*)base+y*N+x;
		unsigned int ival=*ptr;
		SHORT16 check=refGetICART(base, x, y, N);

		ival=ival&0xffff;
		ival=ival|(val<<16);
		*ptr=ival;
	}
}

__host__  __device__  void refSetICART(void *base, int x , int y, int N, SHORT16 val)
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

__host__  __device__  SHORT16 getRCART(void *base, int x, int y, int N)  { return refGetRCART(base, x, y, N);}
__host__  __device__  SHORT16 getICART(void *base, int x, int y, int N)  { return refGetICART(base, x, y, N);}
__host__  __device__  SHORT16 getRPOL(void *base, int r, int t, int N)  { return refGetRPOL(base, r, t, N);}
__host__  __device__  SHORT16 getIPOL(void *base, int r, int t, int N)  { return refGetIPOL(base, r, t, N);}

__host__  __device__  void setRCART (void *base, int x , int y, int N, SHORT16 val) { refSetRCART(base, x , y, N, val);}
__host__  __device__  void setICART (void *base, int x , int y, int N, SHORT16 val) { refSetICART(base, x , y, N, val);}
__host__  __device__  void setRPOL  (void *base, int r , int t, int N, SHORT16 val) { refSetRPOL(base, r , t, N, val);}
__host__  __device__  void setIPOL  (void *base, int r , int t, int N, SHORT16 val) { refSetIPOL(base, r , t, N, val);}

__host__  void initTestCases(void) {

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
	
	test[10].N=190;
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
	
	test[0].arrayy=(void*)array0;test[1].arrayy=(void*)array1;
	test[2].arrayy=(void*)array0;test[3].arrayy=(void*)array1;
	test[4].arrayy=(void*)array0;test[5].arrayy=(void*)array1;
	test[6].arrayy=(void*)array0;test[7].arrayy=(void*)array1;
	test[8].arrayy=(void*)array0;test[9].arrayy=(void*)array1;
	test[10].arrayy=(void*)array0;test[11].arrayy=(void*)array1;
}

__host__  void refComputeCart2Pol(void *POL, void *CART, int N, float R, float theta) {
	
	int r,t;
	float dx,dy,x,y;
	int lli, llj;
	
  	/* delta in x and y direction*/
	dx=((R+1)-(R*(cos(theta))))/(N-1);
	dy=((R+1)*(sin(theta)))/(N-1);

	for(r=0;r<N;r++) {
		for(t=0;t<N;t++) {

			/*x and y value for given r and t*/
			x=(R+(float)r/(N-1))*cos(theta*(float)t/(N-1));
			y=(R+(float)r/(N-1))*sin(theta*(float)t/(N-1));

			
			lli= (int)((x-R*cos(theta))/dx);			
			llj= (int)((y/dy));

			if((lli==(N-1))&&(llj==(N-1))) {
				/* boundary case handling, not necessary */
				float real = SHORT16ToDouble(refGetRCART(CART,N-1,N-1,N));
				float imag = SHORT16ToDouble(refGetICART(CART,N-1,N-1,N));
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else if(lli==(N-1)) {
				/* boundary case handling, not necessary*/
				float real=    (SHORT16ToDouble(refGetRCART(CART, N-1, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, N-1, llj, N)))/2;
				float imag=	(SHORT16ToDouble(refGetICART(CART, N-1, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, N-1, llj, N)))/2;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else if(llj==(N-1)) {
				/* boundary case handling, not necessary */
				float real=	(SHORT16ToDouble(refGetRCART(CART, lli, N-1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, N-1, N)))/2;
				float imag=	(SHORT16ToDouble(refGetICART(CART, lli, N-1, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, N-1, N)))/2;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else {
				/*Compute the acerage value of the enclosing cart points*/
				float real= 
					
					(SHORT16ToDouble(refGetRCART(CART, lli, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli, llj, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, llj, N)))/4;
					
				float imag= 
					(SHORT16ToDouble(refGetICART(CART, lli, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, lli, llj, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, llj, N)))/4;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
				
			}
		}
	}
}
//////////////////////////////////////////my code/////////////////
__global__ void ComputeCart2Pol(void *POL, void *CART, int N, float R, float theta,int WT) {
	
	int r,t;
	float dx,dy,x,y;
	int lli, llj;
	int INDEX;
	
			INDEX = threadIdx.x +  WT * (threadIdx.y + blockIdx.y) ;

			t =(INDEX / N) ;
			r =(INDEX -t * N); 
	
	
  				/* delta in x and y direction*/
				dx=((R+1)-(R*(cos(theta))))/(N-1);
				dy=((R+1)*(sin(theta)))/(N-1);


			/*x and y value for given r and t*/
			x=(R+(float)r/(N-1))*cos(theta*(float)t/(N-1));
			y=(R+(float)r/(N-1))*sin(theta*(float)t/(N-1));

			
			lli= (int)((x-R*cos(theta))/dx);			
			llj= (int)((y/dy));

			if((lli==(N-1))&&(llj==(N-1))) {
				/* boundary case handling, not necessary */
				float real = SHORT16ToDouble(refGetRCART(CART,N-1,N-1,N));
				float imag = SHORT16ToDouble(refGetICART(CART,N-1,N-1,N));
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else if(lli==(N-1)) {
				/* boundary case handling, not necessary*/
				float real=    (SHORT16ToDouble(refGetRCART(CART, N-1, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, N-1, llj, N)))/2;
				float imag=	(SHORT16ToDouble(refGetICART(CART, N-1, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, N-1, llj, N)))/2;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else if(llj==(N-1)) {
				/* boundary case handling, not necessary */
				float real=	(SHORT16ToDouble(refGetRCART(CART, lli, N-1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, N-1, N)))/2;
				float imag=	(SHORT16ToDouble(refGetICART(CART, lli, N-1, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, N-1, N)))/2;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
			} else {
				/*Compute the acerage value of the enclosing cart points*/
				float real= 
					
					(SHORT16ToDouble(refGetRCART(CART, lli, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli, llj, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, llj+1, N))+
					SHORT16ToDouble(refGetRCART(CART, lli+1, llj, N)))/4;
					
				float imag= 
					(SHORT16ToDouble(refGetICART(CART, lli, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, lli, llj, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, llj+1, N))+
					SHORT16ToDouble(refGetICART(CART, lli+1, llj, N)))/4;
				refSetRPOL(POL, r, t, N, floatToSHORT16(real));
				refSetIPOL(POL, r, t, N, floatToSHORT16(imag));
				
			}
}
////////////////////////////////////////////////
 __host__ void testCart2pol() {
	
	int okay=1, testnum;

	initTestCases();

	for(testnum=10;testnum<11;testnum++) {
		unsigned int refsum=0, ressum=0;

		printf("****************************************\n");
		printf("Begin Test %d N=%d, R=%d/1000000, theta=%d/1000000\n",
			testnum, test[testnum].N, (int)(test[testnum].R*1000000), (int)(test[testnum].theta*1000000));
		{
			clock_t start;
			clock_t end;

			start = clock();

			/* compute "ground truth" using float precision reference implementation. */
			refComputeCart2Pol((void *)Refresult, (void *)test[testnum].arrayy, test[testnum].N, test[testnum].R, test[testnum].theta);

			end = clock();
			printf("%f \n",(double)(end-start)/((double)CLOCKS_PER_SEC));
		}

		{	/* run and time your implementation */	
			
			clock_t start;
			clock_t end;

			int N = test[testnum].N;
			
			/////by me
			int NB;
			int WT;

			if ( N == 1000) { NB = 2605; WT = 384;}
			else if ( N == 700) {NB = 1277; WT = 384;}
			else if ( N == 937) {NB = 2287; WT = 384;}
			else if ( N == 683) {NB = 14578; WT = 32;}		
			else if ( N == 13) {NB = 11; WT = 16;}
			else if ( N == 10) {NB = 4; WT = 25;}
			else if (N == 190) {NB = 1129;WT = 32;}
			else {NB = (N/32)+1; WT = 32;}

			dim3 dimGrid(1,NB);
			dim3 dimBlock( WT , 1);
			
			int CART_size =2* N * N * sizeof(unsigned int);
			int POL_size = 2 * N * N * sizeof(unsigned int);
		
			void * CART_d;
			void * POL_d;
			
			cudaMalloc((void**)&CART_d, CART_size);
			cudaMalloc((void**)&POL_d, POL_size);
			cudaMemcpy(CART_d, test[testnum].arrayy , CART_size, cudaMemcpyHostToDevice);

			/////
			start = clock();
				/*GPU computation */		
			ComputeCart2Pol<<< dimGrid,dimBlock>>>((void *)POL_d, (void *)CART_d, test[testnum].N, test[testnum].R, test[testnum].theta, WT);
						
			end = clock();
			/////
			cudaMemcpy(result, POL_d, POL_size, cudaMemcpyDeviceToHost);
	
					
			cudaFree (CART_d); 
		    cudaFree (POL_d);

			printf("%f\n",(double)(end-start)/(double)(CLOCKS_PER_SEC));
			
		}



		{
			float dx=((test[testnum].R+1)-(test[testnum].R*(cos((float)test[testnum].theta))))/(test[testnum].N-1);
			float dy=((test[testnum].R+1)*(sin((float)test[testnum].theta)))/(test[testnum].N-1);
			int i, j, thisokay=1;

			printf("\tChecking for correctness\n");

			


			/* check your answer against the ground truth */
			for(i=0;i<(test[testnum].N);i++) {
				for(j=0;j<(test[testnum].N);j++) {
					float posx=(test[testnum].R+(float)i/(test[testnum].N-1))*cos(test[testnum].theta*(float)j/(test[testnum].N-1));
					float posy=(test[testnum].R+(float)i/(test[testnum].N-1))*sin(test[testnum].theta*(float)j/(test[testnum].N-1));
					int omit = 	(((int)(posy/dy)+1) < (posy/dy + OFFSET))||
						(((int)(posy/dy)) > (posy/dy - OFFSET))||
						(((int)((posx-test[testnum].R*cos(test[testnum].theta))/dx)+1) < ((posx-test[testnum].R*cos(test[testnum].theta))/dx + OFFSET))||
						(((int)((posx-test[testnum].R*cos(test[testnum].theta))/dx)) > ((posx-test[testnum].R*cos(test[testnum].theta))/dx - OFFSET));

					if(VERBOSE) {
						SHORT16 ref,res;
						ref=refGetRPOL(Refresult,i,j,test[testnum].N);
						res=getRPOL(result,i,j,test[testnum].N);
						printf("\t**the real value at (%d, %d) differs by %d(SHORT16 format), %x, %x\n",i,j,ref-res, ref, res);

						ref=refGetIPOL(Refresult,i,j,test[testnum].N);
						res=getIPOL(result,i,j,test[testnum].N);
						printf("\t**the imaginary value at (%d, %d) differs by %d(SHORT16 format) %x, %x\n",i,j,ref-res, ref, res);
						
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
							printf("\t**%s The real value at (%d, %d) differs by %d(SHORT16 format)\n",omit?"(okay)":"(not okay)",i,j,diffRPol);
								FILE * pFile;
		    pFile = fopen ("host.txt","w");

		
				if (pFile!=NULL)
					 {
							fprintf (pFile,"i=%d  ,j=%d  ,  %d + i %d\n",i,j,refRPol , resRPol);
					 }
		
				fclose (pFile);
			/*
				FILE * tFile;
		    tFile = fopen ("device.txt","w");

			for(int i = 0; i<N ; i++)/////chap
				for(int j = 0; j<N ; j++)
			{		

				if (tFile!=NULL)
					 {
							fprintf (tFile,"%f + i %f\n",SHORT16ToDouble(getRPOL((void *)result,i, j, N)),SHORT16ToDouble(getIPOL((void *)result,i, j, N)));
					 }
			}
				fclose (tFile);*/

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
			printf("Test%d %s refsum=%x; ressum=%x\n", testnum, thisokay?"Passed":"Failed", refsum, ressum);
		}

	} // for(testnum)

	if (okay) 
		printf("All okay!!!\n");
	else 
		printf("Some test(s) failed!!!\n");
}
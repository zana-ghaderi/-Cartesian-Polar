#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include "cuComplex.h"

using namespace std;

typedef cuComplex DCX;
typedef signed short SHORT16;
typedef struct test TESTCASE;


#define VERBOSE 0 
#define OFFSET (((double)1.0)/(1<<15))		// Offset that we will not care about the output
#define THRESHOLD 4	
#define NMAX 1000


struct test{
	void *array;
	int N;
	double theta;
	double R;
	void *result;
};

double gR, gTheta, gN = 1000;

unsigned int array0[1000][1000];
unsigned int array1[1000][1000];
unsigned int Refresult[1000][1000];
unsigned int result[1000][1000];

unsigned int mychecksum(unsigned int sum, SHORT16 temp) 
{
	if (sum & 0x80000000) {
		sum<<=1;
		sum|=1;
	} else sum<<=1;

	sum=sum^temp;
	return sum;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 SHORT16 doubleToSHORT16(double val)
{
  SHORT16 i;
  val=val * pow(2.0,8);
  if (val>=0) {
    i=val+0.5;
  } else {
    i=val-0.5;
  }
  return i;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double SHORT16ToDouble(SHORT16 val)
{
  double r;
  r=((double)val)*pow(2.0,-8);
 
  return r;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 SHORT16 refGetRCART(void *base, int x, int y, int N)
{
  if(x<0||x>=N||y<0||y>=N)
    return -1;
  else {
    unsigned int *ptr=(unsigned int*)base+y*N+x;
    unsigned int val=*ptr;
    val=(val>>16)&0xffff;
    return val;
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  SHORT16 refGetICART(void *base, int x, int y, int N)
{
  if(x<0||x>=N||y<0||y>=N)
    return -1;
  else {
    unsigned int *ptr=(unsigned int*)base+y*N+x;
    unsigned int val=*ptr;
    val=val&0xffff;
    return val;
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 SHORT16 refGetRPOL(void *base, int r, int t, int N)
{
  if(r<0||r>=N||t<0||t>=N)
    return -1;
  else {
    unsigned int *ptr=(unsigned int*)base+t*N+r;
    unsigned int val=*ptr;
    val=(val>>16)&0xffff;
    return val;
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  SHORT16 refGetIPOL(void *base, int r, int t, int N)
{
  if(r<0||r>=N||t<0||t>=N)
    return -1;
  else {
    unsigned int *ptr=(unsigned int*)base+t*N+r;
    unsigned int val=*ptr;
    val=val&0xffff;
    return val;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

 void refSetRCART(void *base, int x , int y, int N, SHORT16 val)
{
  if(x<0||x>=N||y<0||y>=N)
    return;

  {
    unsigned int *ptr=(unsigned int*)base+y*N+x;
    unsigned int ival=*ptr;
    SHORT16 check=refGetICART(base, x, y, N);

    ival=ival&0xffff;
    ival=ival|(val<<16);
    *ptr=ival;
   

  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////

  void refSetICART(void* base, int x , int y, int N, SHORT16 val)
{
  if(x<0||x>=N||y<0||y>=N)
    return;

  {
    unsigned int *ptr=(unsigned int*)base+y*N+x;
    unsigned int ival=*ptr;
    SHORT16 check=refGetRCART(base, x, y, N);

    ival=ival&0xffff0000;
    ival=ival|(0x0000ffff&val);
    *ptr=ival;
   
  
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////

  void refSetRPOL(void *base, int r , int t, int N, SHORT16 val)
{
  if(r<0||r>=N||t<0||t>=N)
    return;

  {
    unsigned int *ptr=(unsigned int*)base+t*N+r;
    unsigned int ival=*ptr;
    SHORT16 check=refGetIPOL(base, r, t, N);

    ival=ival&0xffff;
    ival=ival|(val<<16);
    *ptr=ival;
   
   
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////

 void  refSetIPOL(void *base, int r , int t, int N, SHORT16 val)
{
  if(r<0||r>=N||t<0||t>=N)
    return;

  {
    unsigned int *ptr=(unsigned int*)base+t*N+r;
    unsigned int ival=*ptr;
    SHORT16 check=refGetRPOL(base, r, t, N);

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
/////////////////////////////////////IN CPU///////////////////////////////////////////////////////////////////////
__host__ void refComputeCart2Pol(void *POL, void *CART, int N, double R, double theta)
 {
	int r,t;
	double dx,dy,x,y;
	int lli,llj;


	/* delta in x and y direction*/
	dx=((R+1)-(R*(cos(theta))))/(N-1);
	dy=((R+1)*(sin(theta)))/(N-1);

	for(r=0;r<N;r++) {
		for(t=0;t<N;t++) {

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
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeCart2Pol (DCX *POL_d ,DCX *CART_d ,int N ,float R ,float Tetha , int WT)
{
			float Angle , Radius;
	
			int ID_BLK_Tetha = 0;
			int ID_BLK_Rad = 0;
			
			int INDEX = 0;
			float dy = 0;
			float dx = 0;
			

			dy = ((R + 1) * sinf (Tetha)) /(N - 1) ;
			dx = ((R + 1) - R *cosf (Tetha)) /(N - 1) ;

			INDEX = threadIdx.x +  WT * (threadIdx.y + blockIdx.y) ;
			
			ID_BLK_Tetha = INDEX / N ;
			ID_BLK_Rad = INDEX - (ID_BLK_Tetha * N); 
	
			/////////IndexToPolar//////////
			Radius = R + ID_BLK_Rad /(N - 1) ;
			Angle = Tetha * ID_BLK_Tetha /(N - 1) ;


			float xx , yy ;

	       ////////PolarToCart////////////
		
			xx = Radius * __cosf(Angle) ;
			yy = Radius * __sinf(Angle) ;

			int ix , iy ;

			//////////CartToindex////////

			iy =  yy / dy ;
			ix =  (xx - (R * __cosf (Tetha))) / dx ;
			
			if ((iy == N - 1) && (ix == N - 1))
			 {
				 POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].x = CART_d [iy * N + ix].x;
				 POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].y = CART_d [iy * N + ix].y;
			 }
			else if (ix == N - 1)
			{
				 POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].x = (CART_d [(N - 1) * N + ix + 1].x + CART_d [(N - 1) * N + ix ].x)/2;
				 POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].y = (CART_d [(N - 1) * N + ix + 1].y + CART_d [(N - 1) * N + ix ].y)/2;
			}
			else if (iy == N - 1)
			{
				POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].x = (CART_d [(iy) * N + N - 1].x + CART_d [(iy + 1) * N + N - 1].x)/2;
				POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].y = (CART_d [(iy) * N + N - 1].y + CART_d [(iy + 1) * N + N - 1].y)/2;
			}
			else 
			{

			//////////IndexToCart/////////
	POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].x = 
		(CART_d [iy * N + ix].x + CART_d [iy * N + ix + 1 ].x + CART_d [(iy + 1 ) * N + ix].x + CART_d [(iy + 1 )* N + ix + 1].x)/4;
      
 
	POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].y = 
		(CART_d [iy * N + ix].y + CART_d [iy * N + ix + 1 ].y + CART_d [(iy + 1 ) * N + ix].y + CART_d [(iy + 1 )* N + ix + 1].y)/4;
			}

}
void callcomputeCart2Pol(DCX* POL,DCX* CART ,int N ,float R ,float Tetha , int NB, int WT)
{
	int CART_size = N  * N * sizeof(DCX);
	int POL_size = N  * N * sizeof(DCX);
	
	DCX* CART_d;
	DCX* POL_d;

	cudaMalloc((void**)&CART_d, CART_size);
	cudaMemcpy(CART_d, CART , CART_size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&POL_d, POL_size);

	dim3 dimGrid(1,NB);
	dim3 dimBlock( WT , 1);

	ComputeCart2Pol<<< dimGrid,dimBlock>>> ( POL_d ,CART_d ,N , R , Tetha , WT);

	cudaMemcpy(POL, POL_d, POL_size, cudaMemcpyDeviceToHost);
	
	
	cudaFree (CART_d); 
    cudaFree (POL_d); 
}

/////////////////////////////////////////////////////////////////////////////

void OUR_CALC(void * temp_pol,void * temp_cart,int N, float R,float Tetha)
{
     	int  NB, WT ;
				
		DCX* CART;
		DCX* POL;

		size_t size_dcx = N * N * sizeof(DCX);
		size_t size_int = N * N * sizeof(int);

		CART = (DCX*) malloc(size_dcx);
        POL  = (DCX*) malloc(size_dcx);
		
		temp_cart = (void *) malloc(size_int);
		temp_pol  = (void *) malloc(size_int);
		
		/////////////////////////

		for(int i = 0; i < N ; i++)
			for (int j = 0; j < N; j++)
			{		
				CART[i * N + j] .x = SHORT16ToDouble(getRCART((void *) temp_cart, i, j,  N));
				CART[i * N + j] .y = SHORT16ToDouble(getICART((void *) temp_cart, i, j,  N));
			}

			if ( N == 1000) { NB = 2506; WT = 384;}
			else if ( N == 700) {NB = 958; WT = 512;}
			else if ( N == 937) {NB = 2195; WT = 400;}
			else if ( N == 683) {NB = 7298; WT = 64;}		
			else if ( N == 13) {NB = 11; WT = 16;}
			else if ( N ==  10) {NB = 25; WT = 25;}
			callcomputeCart2Pol ( POL,CART ,N , R , Tetha ,  NB, WT);

			for(int i = 0; i < N ; i++)
				for (int j = 0; j < N; j++)
				{
					setRPOL  ((void *)temp_pol ,i , j, N , doubleToSHORT16(POL[i * N + j].x));
					setIPOL  ((void *)temp_pol ,i , j, N , doubleToSHORT16(POL[i * N + j].y));
				}		
		 free (CART);
		 free (POL);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Initialize .. */
TESTCASE test[12];

void initTestCases(void) {
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
			refSetRCART((void*)array0, i, j, gN, doubleToSHORT16(i*(double)(1/(double)(j+1))));
			refSetICART((void*)array0, i, j, gN, -doubleToSHORT16(i*(double)(1/(double)(j+1))));
			refSetRCART((void*)array1, i, j, gN, doubleToSHORT16(pow((double)1.05,(double)(i+j))));
			refSetICART((void*)array1, i, j, gN, -doubleToSHORT16(pow((double)1.05,(double)(i+j))));
		}
	}
	
	test[0].array=(void*)array0;test[1].array=(void*)array1;
	test[2].array=(void*)array0;test[3].array=(void*)array1;
	test[4].array=(void*)array0;test[5].array=(void*)array1;
	test[6].array=(void*)array0;test[7].array=(void*)array1;
	test[8].array=(void*)array0;test[9].array=(void*)array1;
	test[10].array=(void*)array0;test[11].array=(void*)array1;
}

///////////////////////////////////////////////////////////////////////////////
void testCart2pol() {

	int okay=1, testnum;

	initTestCases();

	for(testnum=0;testnum<12;testnum++) {
		unsigned int refsum=0, ressum=0;

		printf("****************************************\n");
		printf("Begin Test %d N=%d, R=%d/1000000, theta=%d/1000000\n",
			testnum, test[testnum].N, (int)(test[testnum].R*1000000), (int)(test[testnum].theta*1000000));
		{
			clock_t start;
			clock_t end;

			start = clock();

			/* compute "ground truth" using double precision reference implementation. */
			refComputeCart2Pol((void *)Refresult, (void *)test[testnum].array, test[testnum].N, test[testnum].R, test[testnum].theta);
			end = clock();
			printf("%f \n",(end-start)/((double)CLOCKS_PER_SEC));
		}

		
		{	/* run and time your implementation */	
			clock_t start;
			clock_t end;
			start = clock();
			 OUR_CALC((void *)result, (void *)test[testnum].array, test[testnum].N, test[testnum].R, test[testnum].theta);
			end = clock();
			printf("%f \n",(end-start)/((double)CLOCKS_PER_SEC));
		}

		///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	
		
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

int main ()
{
	testCart2pol();
	return 0;
}



#include <iostream>
#include <ostream>
#include <math.h>
#include <time.h>
#include <cublas.h>
#include <stdio.h>
#include "cuComplex.h"


using namespace std;

 typedef cuDoubleComplex  DCX;


///////////////////////////////////////////////////////////////////////////////
__global__ void Pol_To_Cart_Device (int N ,float R ,float Tetha ,DCX *CART_d ,DCX *POL_d , int NB , int HT , int WT)
{
			float Angle , Radius;
	
			int ID_BLK_Tetha = 0;
			int ID_BLK_Rad = 0;
			
			int INDEX = 0;
			float dy = 0;
			float dx = 0;

			

			dy = ((R + 1) * sinf (Tetha)) /(N - 1) ;
			dx = ((R + 1) - R *cosf (Tetha)) /(N - 1) ;

			INDEX = threadIdx.x +  WT * threadIdx.y + WT * HT * blockIdx.y ;
			
			ID_BLK_Tetha = INDEX / N ;
			ID_BLK_Rad = INDEX - (ID_BLK_Tetha * N); 
	
			/////////IndexToPolar//////////
			Radius = R + ID_BLK_Rad /(N - 1) ;
			Angle = Tetha * ID_BLK_Tetha /(N - 1) ;


			double xx , yy ;

	       ////////PolarToCart////////////
		
			xx = Radius * cosf(Angle) ;
			yy = Radius * sinf(Angle) ;

			int ix , iy ;

			//////////CartToindex////////

			iy =  floorf(yy / dy );
			ix =  floorf((xx - (R * __cosf (Tetha))) / dx) ;
			
            	
			//////////IndexToCart/////////
	POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].x = 
		(CART_d [iy * N + ix].x + CART_d [iy * N + ix + 1 ].x + CART_d [(iy + 1 ) * N + ix].x + CART_d [(iy + 1 )* N + ix + 1].x)/4;
      
 
	POL_d [ID_BLK_Rad * N + ID_BLK_Tetha].y = 
        (CART_d [iy * N + ix].y + CART_d [iy * N + ix + 1 ].y + CART_d [(iy + 1 ) * N + ix].y + CART_d [(iy + 1 )* N + ix + 1].y)/4;
}

////////////////////////////////////////////////////////////////////////////////

void Pol_To_CRT_Host(int N ,double R ,double Tetha ,DCX* CART ,DCX* POL, int NB, int HT, int WT)

{

	int CART_size = N  * N * sizeof(DCX);
	int POL_size = N  * N * sizeof(DCX);
	
	DCX* CART_d;
	DCX* POL_d;

	cudaMalloc((void**)&CART_d, CART_size);
	cudaMemcpy(CART_d, CART , CART_size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&POL_d, POL_size);

	dim3 dimGrid(1,NB);
	dim3 dimBlock( WT , HT);

	Pol_To_Cart_Device <<< dimGrid,dimBlock>>> (N , R , Tetha , CART_d , POL_d , NB , HT , WT);

	cudaMemcpy(POL, POL_d, POL_size, cudaMemcpyDeviceToHost);
	//
	for(int t=0 ; t < test[testnum].N; t++)
		for (r=0 ; r < test[testnum].N; r++)
		{
			setIPOL  (result, r , t, test[testnum].N , doubleToSHORT16(POL[t*test[testnum].N+r].y));
			setRPOL  (result, r , t, test[testnum].N , doubleToSHORT16(POL[t*test[testnum].N+r].x));
		}
	
	
	cudaFree (CART_d); 
    cudaFree (POL_d); 
}

/////////////////////////////////////////////////////////////////////////////

int main()
{
     	int N, NB, HT, WT ;
		float R ;

		static int cache=0;
		float Tetha ;/// 0.01227 < Tetha < 0.78539

		while (1)
		{
			
		cout << "Enter N :";
		cin >> N;

		cout << "Enter R between 10 <= R <= 100:";
		cin >> R;

		cout << "Enter Tetha between 0.01227 <= Tetha <= 0.78539:";
		cin >> Tetha;

		cout<<"Enter Block Number between 2605 and 65535 : ";
		cin>>NB;

		cout<<"Enter Width of Thread : ";
		cin>>WT;

		cout<<"Enter Higth oh Thread : ";
		cin>>HT;


		

		
		DCX* CART;
		DCX* POL;

     	clock_t start;
		clock_t end;
	
		size_t size_dcx = N * N * sizeof(DCX);

		CART = (DCX*) malloc(size_dcx);
        POL = (DCX*) malloc(size_dcx);

		//double remx , remy

		double MinX, MaxX, MaxY, stepX, stepY;
		MinX = R * cos (Tetha);
		MaxX = R+1;
		MaxY = (R+1) * sin (Tetha);
		stepX = ( MaxX - MinX ) / (N-1);
		stepY = MaxY / (N-1);
   
		/////Fill matrix randomly

		for(int i = 0; i < N ; i++)
			for (int j = 0; j < N; j++)
			{		
				CART[i * N + j] .x = MinX + stepX * j;
				CART[i * N + j] .y = stepY * i;
			}
		
		/*for(int i = 0; i<N * N; i++)/////chap
		{		
			//cout << CART [i].x << "+ i " << CART [i].y<< endl;	
		}*/

		start = clock();///////////START

			Pol_To_CRT_Host (N , R , Tetha , CART , POL, NB, HT, WT);
	
		end = clock();////////////END

		cout <<" TOTAL E_Time  is : " << (double)( end - start ) / CLOCKS_PER_SEC <<" seconds"<< endl;
		 cout <<"*********************************************************"<<endl;
		cin>>HT;
		////////////////////////////////////////////////Filing/////////////////////////////////////////

		FILE * pFile;
	    pFile = fopen ("DIC.txt","w");

		for(int i = 0; i<N * N; i++)/////chap
		{		
			//cout<< POL [i].x << "+ i "<<POL [i].y<<endl;
			 
				if (pFile!=NULL)
					 {
							fprintf ( pFile, "%f + i %f\n",POL [i].x,POL [i].y);
					 }
		}
			fclose (pFile); 
		//////////////////////////////////////////////////////////////////////////////////////////////

		 
		 free (CART);
		 free (POL);

		}
return 0;
}

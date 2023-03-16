#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

__global__ void MatrixMulKernel(float* Amat, float* Bmat, float* Cmat,int width,int BLK_Amount)
{
     	int ID_BLK1 = 0;
	int ID_BLK2 = 0;
	int Threads = 0;

	float A = 0;
	float B = 0;

	Threads = width / BLK_Amount;
     
	ID_BLK1 = threadIdx.x + blockIdx.x*(Threads);
     	ID_BLK2 = threadIdx.y + blockIdx.y*(Threads);

	for (int i = 0; i < width; i++) 
	{
		 A = Amat[ ID_BLK2 * width + i];
		 B = Bmat[ i * width + ID_BLK1];
		Cmat [ ID_BLK2 * width + ID_BLK1 ] += A * B;
	}
}
////////////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(float* Amat_D, float* Bmat_D, float* Cmat_D, int width,int BLK_Amount)

{
	int Threads = 0;
	int Mat_size = 0; 

	float* Amat;
	float* Bmat;
	float* Cmat;

	Threads = width / BLK_Amount;
	Mat_size = width * width * sizeof(float);	

	cudaMalloc((void**)&Amat, Mat_size);
	cudaMemcpy(Amat, Amat_D, Mat_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Bmat, Mat_size);
	cudaMemcpy(Bmat, Bmat_D, Mat_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Cmat, Mat_size);

	dim3 dimGrid(BLK_Amount, BLK_Amount);
	dim3 dimBlock(Threads, Threads);

	MatrixMulKernel<<<dimGrid, dimBlock>>>( Amat, Bmat, Cmat, width, BLK_Amount);

	cudaMemcpy(Cmat_D, Cmat, Mat_size, cudaMemcpyDeviceToHost);

	cudaFree (Amat); 
        cudaFree (Bmat); 
        cudaFree (Cmat);

}

///////////////////////////////////////////////////////////////////////////////////////
int main()
{

	int width = 512;
     	int BLK_Amount = 0;

	float* A;
	float* B;
	float* C;

	clock_t start;
	clock_t end;

	
	cout<<endl<<"Please Enter amount Of blocks that you want execute program on them: ";
	cin>>BLK_Amount;

	size_t size = width * width * sizeof(float);

	A = (float*)malloc(size);
	B = (float*)malloc(size);
	C = (float*)malloc(size);

        for(int i = 0; i<width*width; i++)/////Fill matrices randomly
	{
		A[i] = rand();
		B[i] = rand();	
	}	

	start = clock();

	MatrixMulOnDevice(A,B,C,width,BLK_Amount);	

	end = clock();

	cout<<"ETime is : "<<(double)(end - start) / CLOCKS_PER_SEC<<" seconds "<< endl;

return 0;
}

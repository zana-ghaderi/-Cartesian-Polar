/*#include <iostream>
#include <math.h>
#include <time.h>
#include <complex>
using namespace std;

///////////////////////////////////////////////////////////////////////////////
__global__ void Mat_Mul_Kernel(float* Amat, float* Bmat, float* Cmat, int width)

{
	float A = 0;
	float B = 0;
	
	for (int i = 0; i < width; i++) 
	{
		 A = Amat[threadIdx.y * width+i];
	         B = Bmat[i*width + threadIdx.x];
		Cmat[threadIdx.y*width + threadIdx.x] += A * B;
	}
}

////////////////////////////////////////////////////////////////////////////////

void Mat_Mul_Device(float* Amat_D, float* Bmat_D, float* Cmat_D, int width)

{

	int Mat_size = width * width * sizeof(float);

	float* Amat;
	float* Bmat;
	float* Cmat;

	cudaMalloc((void**)&Amat, Mat_size);
	cudaMemcpy(Amat, Amat_D, Mat_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Bmat, Mat_size);
	cudaMemcpy(Bmat, Bmat_D, Mat_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Cmat, Mat_size);

	dim3 dimGrid(1, 1);
	dim3 dimBlock(width, width);

	Mat_Mul_Kernel<<<dimGrid, dimBlock>>>(Amat, Bmat, Cmat, width);

	cudaMemcpy(Cmat_D, Cmat, Mat_size, cudaMemcpyDeviceToHost);

	cudaFree (Amat); 
    	cudaFree (Bmat); 
     	cudaFree (Cmat);
}

/////////////////////////////////////////////////////////////////////////////

int main()
{
	complex<double> c1(2.1,4.7); //define, r,i
	complex<double> c2(6.5,9);
	complex<double> c3(0,0);
	//c1=complex<double>(5,7); //meghdar dehi
	cout<<c1.imag()<<endl; //chap
	cout<<c1<<endl; //chap
	//c1._Add<double>(c2);
	c3=complex<double>(c1.real()+c2.real(),c1.imag()+c2.imag());
	cout<<c3.real()<<endl;
	c3 = c1 + c2;
	cout<<c3.real()<<endl;
	c3 = complex<double>(4,6.7567567675745);
	cout<< c3;
	c3=complex<double>(c3.real()/4,c3.imag()/4);
	cout<<c3.real()<<endl;
    int width = 16;
	float* A;
	float* B;
	float* C;

     	clock_t start;
	clock_t end;

	size_t size = width * width * sizeof(float);

	A = (float*)malloc(size);
	B = (float*)malloc(size);
	C = (float*)malloc(size);

        for(int i = 0; i<width*width; i++)/////Fill matrix randomly
	{
		A[i] = rand();
		B[i] = rand();	
	}

	start = clock();///////////START

		Mat_Mul_Device(A, B, C, width);

	end = clock();////////////END


	cout <<" TOTAL E_Time  is : " << (double)( end - start ) / CLOCKS_PER_SEC <<" seconds"<< endl;


return 0;
}*/
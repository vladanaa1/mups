#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include<fstream>
#include <omp.h>
#include <chrono>

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v)
{
  std::cerr << "Opening file:"<< fn << std::endl;
  std::fstream f(fn, std::fstream::in);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;

  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  while (f.good() ) {
    f >> data;
    v.push_back(data);
  }
  v.pop_back(); // remove the duplicated last element

  return true;

}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float>&v)
{
  std::cerr << "Opening file:"<< fn << " for write." << std::endl;
  std::fstream f(fn, std::fstream::out);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f << nr_row << " "<<nr_col<<" ";

  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  for (int i = 0; i < v.size(); ++i) {
    f << v[i] << ' ';
  }
  f << "\n";
  return true;

}

/* 
 * Base C implementation of MM
 */

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}

void parallelSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in parallelSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in parallelSgemm()" << std::endl;
    return;
  }

  // Get the number of threads
  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();

    // Divide the rows of matrix C among threads
    int start_row = (m * thread_id) / num_threads;
    int end_row = (m * (thread_id + 1)) / num_threads;

    // Each thread processes its assigned rows
    for (int mm = start_row; mm < end_row; ++mm) {
      for (int nn = 0; nn < n; ++nn) {
        float c = 0.0f;
        for (int i = 0; i < k; ++i) {
          float a = A[mm + i * lda]; 
          float b = B[nn + i * ldb];
          c += a * b;
        }
        C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
      }
    }
  }
}

void parallelSgemmWithWorksharing(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in parallelSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in parallelSgemm()" << std::endl;
    return;
  }

  #pragma omp parallel for collapse(2) schedule(static)
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda]; 
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
    }
  }
}

bool compare_mtrx(std::vector<float> m1, std::vector<float> m2){
  double epsilon = 0.01;
  for (int i = 0; i < m1.size(); ++i) {
    if(std::abs(m1[i] - m2[i]) > epsilon) return false;
  }
  return true;
}


int main (int argc, char *argv[]) {

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

   if (argc != 4)  
  {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
  }
 
  /* Read in data */
  // load A
  readColMajorMatrixFile(argv[1], matArow, matAcol, matA);

  // load B^T
  readColMajorMatrixFile(argv[2], matBcol, matBrow, matBT);

  // allocate space for C
  std::vector<float> matC(matArow*matBcol);
  std::vector<float> matCPar(matArow*matBcol);

  // Use standard sgemm interface
  auto start_seq = std::chrono::high_resolution_clock::now();
  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);
  auto end_seq = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_seq = end_seq - start_seq;
  std::cout << "Sequential SGEMM Time: " << duration_seq.count() << " seconds" << std::endl;

/*
  // Parallel implementation
  auto start_par = std::chrono::high_resolution_clock::now();
  parallelSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);
  auto end_par = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_par = end_par - start_par;
  std::cout << "Parallel SGEMM Time: " << duration_par.count() << " seconds" << std::endl;
*/

  // Parallel implementation (with worksharing)
  auto start_ws = std::chrono::high_resolution_clock::now();
  parallelSgemmWithWorksharing('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matCPar.front(), matArow);
  auto end_ws = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_ws = end_ws - start_ws;
  std::cout << "Parallel SGEMM Time with worksharing: " << duration_ws.count() << " seconds" << std::endl;

  if(compare_mtrx(matC,matCPar)) std::cout<<"Test PASSED\n";
  else std::cout<<"Test FAILED\n";

  writeColMajorMatrixFile(argv[3], matArow, matBcol, matC); 
  std::cout << std::endl;
  return 0;
}

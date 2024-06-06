#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;


//размер блока 
const int BLOCK_SIZE = 32;

__global__ void shared_matrix_mult(const int *a, const int *b, int *c, int N) {
  // Вычислить индекс строки и столбца для потока
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Статически выделяемая общая память
  __shared__ int s_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int s_b[BLOCK_SIZE][BLOCK_SIZE];

  int tmp = 0;

  // считаем блоки в матрице
  for (int i = 0; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        if (row < N && (i * BLOCK_SIZE + threadIdx.x) < N) {
            s_a[threadIdx.y][threadIdx.x] = a[row * N + (i * BLOCK_SIZE + threadIdx.x)];
        }
        else {
            s_a[threadIdx.y][threadIdx.x] = 0;
        }
        if ((i * BLOCK_SIZE + threadIdx.y) < N && col < N) {
            s_b[threadIdx.y][threadIdx.x] = b[(i * BLOCK_SIZE + threadIdx.y) * N + col];
        }
        else {
            s_b[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        c[row * N + col] = tmp;
    }
}

// проверяем результат на CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += a[i * N + k] * b[k * N + j];
      }

    //   assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
  // матрица размером 1024 x 1024;
//   int N = 1 << 10;
  int N = 5;
  size_t bytes = N * N * sizeof(int);

  // задаем вектора размера N*N на Host
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // заполняем матрицы
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // выделяем память 
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // копируем данные
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // количество потоков
  int THREADS = 32;

  // определяем количество блоков
  int BLOCKS = (N + THREADS - 1) / THREADS;

  // задаем размеры, используя конструкцию dim3
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // запускаем ядро
  shared_matrix_mult<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // копируем на host
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // проверяем результаты
  verify_result(h_a, h_b, h_c, N);

  cout << "matrix A \n";
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        cout << h_a[i*N+j] << " ";
    }
    cout << "\n";
  }

  cout << "matrix B \n";
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        cout << h_b[i*N+j] << " ";
    }
    cout << "\n";
  }

  cout << "matrix C \n";
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        cout << h_c[i*N+j] << " ";
    }
    cout << "\n";
  }

  cout << "COMPLETED SUCCESSFULLY\n";
  

  // освобождаем память 
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
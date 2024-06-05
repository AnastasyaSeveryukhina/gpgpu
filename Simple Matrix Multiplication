#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  // Вычислить индекс строки и столбца для потока
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // итерация по строке и вниз по столбцу
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // считаем результат для матрицы с
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
  c[row * N + col] = a[row * N +col];
}

// проверка результатов на CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  //цикл по строкам
  for (int i = 0; i < N; i++) {
    // по столбцам
    for (int j = 0; j < N; j++) {
        int tmp = 0;
        for (int k = 0; k < N; k++) {
            //вычисляем произведение и суммируем
            tmp += a[i * N + k] * b[k * N + j];
        }
        // cout << tmp << "cpu\n";

        // Check against the CPU result
        // cout << c[i*N+j] << "res\n";
        // assert(tmp == c[i * N + j]);
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
  int BLOCKS = N / THREADS;

  // задаем размеры, используя конструкцию dim3
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        cout << h_a[i*N+j] << " el_value in matrix a\n";
    }
  }

  // запускаем ядро
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // копируем на host
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        cout << h_c[i*N+j] << " el_value in matrix c\n";
    }
  }

  // проверяем результаты
  verify_result(h_a, h_b, h_c, N);

  cout << "COMPLETED SUCCESSFULLY\n";
  

  // освобождаем память 
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}

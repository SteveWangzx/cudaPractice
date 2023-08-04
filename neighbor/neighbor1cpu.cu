#include "../error.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

int N;
const int NUM_REPEATS = 10;
const int MN = 10;
const real cutoff = 1.9;                    // distance
const real cutoff_square = cutoff * cutoff; // distance_square

void read_xy(std::vector<real> &x, std::vector<real> &y);
void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y);
void print_neighbor(const int *NN, const int *NL);

int main()
{
    std::vector<real> x, y;
    read_xy(x, y);
    N = x.size();
    int *NN = (int *)malloc(N * sizeof(int));      // Neighbor count for each atom.  length = N
    int *NL = (int *)malloc(N * MN * sizeof(int)); // Neighbor List. length = N * MN

    timing(NN, NL, x, y);
    print_neighbor(NN, NL);
}

void read_xy(std::vector<real> &x, std::vector<real> &y)
{
    std::ifstream infile("xy.txt");
    std::string line, word;
    if (!infile)
    {
        std::cout << "Cannot open xy.txt" << std::endl;
        exit(1);
    }
    while (std::getline(infile, line))
    {
        std::istringstream words(line);
        if (line.length() == 0)
        {
            continue;
        }
        for (int i = 0; i < 2; i++)
        {
            if (words >> word)
            {
                if (i == 0)
                {
                    x.push_back(std::stod(word));
                }
                if (i == 1)
                {
                    y.push_back(std::stod(word));
                }
            }
            else
            {
                std::cout << "Error for reading .txt" << std::endl;
                exit(1);
            }
        }
    }
    infile.close();
}

void find_neighbor(int *NN, int *NL, const real *x, const real *y)
{
    for (int n = 0; n < N; ++n)
    {
        NN[n] = 0;
    }

    for (int i = 0; i < N; ++i)
    {
        real x1 = x[i];
        real y1 = y[i];

        for (int j = i + 1; j < N; ++j)
        {
            real x_dis = x[j] - x1;
            real y_dis = y[j] - y1;
            real distance_square = x_dis * x_dis + y_dis * y_dis;
            if (distance_square < cutoff_square)
            {
                NL[i * MN + NN[i]++] = j;
                NL[j * MN + NN[j]++] = i;
            }
        }
    }
}

void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y)
{
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        while (cudaEventQuery(start) != cudaSuccess)
        {
        }
        find_neighbor(NN, NL, x.data(), y.data());

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
}

void print_neighbor(const int *NN, const int *NL)
{
    std::ofstream outfile("neighbor.txt");
    if (!outfile)
    {
        std::cout << "Cannot open neighbor.txt" << std::endl;
    }
    for (int n = 0; n < N; ++n)
    {
        if (NN[n] > MN)
        {
            std::cout << "Error: MN is too small" << std::endl;
            exit(1);
        }
        outfile << NN[n];
        for (int k = 0; k < MN; ++k)
        {
            if (k < NN[n])
            {
                outfile << " " << NL[n * MN + k];
            }
            else
            {
                outfile << " NaN";
            }
        }
        outfile << std::endl;
    }
    outfile.close();
}

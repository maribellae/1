#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <fstream>
#include<ctime>
#include <chrono>
using namespace std;
int ROWS, COLS;


int** readMatrix(string name) {
    size_t row, col, i, j;
    std::ifstream fin(name);
    if (!fin.is_open()) {
        std::cerr << "Can't open input file!" << std::endl;
        exit(1);
    }

    fin >> row >> col;
    ROWS = row;
    COLS = col;
    if (fin.bad()) {
        std::cerr << "Error while reading file!" << std::endl;
        exit(1);
    }

    if (!col || !row) {
        std::cerr << "Wrong data!" << std::endl;
        exit(1);
    }

    int** matrix;
    matrix = new int* [row];
    for (i = 0; i < row; i++)
        matrix[i] = new int[col];

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            fin >> matrix[i][j];

            if (fin.bad()) {
                std::cerr << "Error while reading file!" << std::endl;
                exit(1);
            }
        }
    }

    fin.close();

    return(matrix);
}


void WriteMatrix(int **matrix , string name) {
    int row = ROWS;
    int col = COLS;
    int i,j;
    std::ofstream fout(name);
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
          
            fout << matrix[i][j] << ((j < col - 1) ? ' ' : '\n');
            if (fout.bad()) {
                std::cerr << "Error while writing data!" << std::endl;
                exit(1);
            }
        }
    }

}

int** NewMatrix(int n)
{
    int* data = (int*)malloc(n * n * sizeof(int));
    int** array = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++)
    {
        array[i] = &(data[n * i]);
    }
    return array;
}

void fillMatrix(int n, int**& mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[i][j] = rand() % 8;
        }
    }
}

void freeMatrix(int n, int** mat)
{
    free(mat[0]);
    free(mat);
}

int** naive(int n, int** mat1, int** mat2)
{
    int** prod = NewMatrix(n);

    int i, j;

#pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            prod[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                prod[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return prod;
}
int** getSlice(int n, int** mat, int offseti, int offsetj)
{
    int m = n / 2;
    int** slice = NewMatrix(m);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            slice[i][j] = mat[offseti + i][offsetj + j];
        }
    }
    return slice;
}

int** addMatrices(int n, int** mat1, int** mat2, bool add)
{
    int** result = NewMatrix(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (add)
                result[i][j] = mat1[i][j] + mat2[i][j];
            else
                result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    return result;
}

int** combineMatrices(int m, int** c11, int** c12, int** c21, int** c22)
{
    int n = 2 * m;
    int** result = NewMatrix(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < m && j < m)
                result[i][j] = c11[i][j];
            else if (i < m)
                result[i][j] = c12[i][j - m];
            else if (j < m)
                result[i][j] = c21[i - m][j];
            else
                result[i][j] = c22[i - m][j - m];
        }
    }

    return result;
}

int** strassen(int n, int** mat1, int** mat2)
{
    if (n <= 8)
    {    

        return naive(n, mat1, mat2);
    }


    int m = n / 2;


    //РАЗБИВАЮ НА БЛОКИ МАТРИЦЫ (ПО ЧЕТЫРЕ РАВНЫХ)
    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);

    int** s1;
#pragma omp task shared(s1)
    {
        int** bds = addMatrices(m, b, d, false); //A12 -A22
        int** gha = addMatrices(m, g, h, true);  //B21+B22
        s1 = strassen(m, bds, gha);
        freeMatrix(m, bds);
        freeMatrix(m, gha);
    }

    int** s2;
#pragma omp task shared(s2)
    {
        int** ada = addMatrices(m, a, d, true); //A11+A22
        int** eha = addMatrices(m, e, h, true); //B11+B22
        s2 = strassen(m, ada, eha);
        freeMatrix(m, ada);
        freeMatrix(m, eha);
    }

    int** s3;
#pragma omp task shared(s3)
    {
        int** acs = addMatrices(m, a, c, false); //A11 - A21
        int** efa = addMatrices(m, e, f, true); //B11+B12
        s3 = strassen(m, acs, efa);
        freeMatrix(m, acs);
        freeMatrix(m, efa);
    }

    int** s4;
#pragma omp task shared(s4)
    {
        int** aba = addMatrices(m, a, b, true);//A11+A12
        s4 = strassen(m, aba, h); //WITH B22
        freeMatrix(m, aba);
    }

    int** s5;
#pragma omp task shared(s5)
    {
        int** fhs = addMatrices(m, f, h, false);//B12-B22
        s5 = strassen(m, a, fhs); //WITH A11
        freeMatrix(m, fhs);
    }

    int** s6;
#pragma omp task shared(s6)
    {
        int** ges = addMatrices(m, g, e, false);//B21-B11
        s6 = strassen(m, d, ges);//WITH A22
        freeMatrix(m, ges);
    }

    int** s7;
#pragma omp task shared(s7)
    {
        int** cda = addMatrices(m, c, d, true);//A21+A22
        s7 = strassen(m, cda, e);//WITH B11
        freeMatrix(m, cda);
    }

#pragma omp taskwait

    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);

    int** c11;
#pragma omp task shared(c11)
    {
        int** s1s2a = addMatrices(m, s1, s2, true);
        int** s6s4s = addMatrices(m, s6, s4, false);
        c11 = addMatrices(m, s1s2a, s6s4s, true);
        freeMatrix(m, s1s2a);
        freeMatrix(m, s6s4s);
    }

    int** c12;
#pragma omp task shared(c12)
    {
        c12 = addMatrices(m, s4, s5, true);
    }

    int** c21;
#pragma omp task shared(c21)
    {
        c21 = addMatrices(m, s6, s7, true);
    }

    int** c22;
#pragma omp task shared(c22)
    {
        int** s2s3s = addMatrices(m, s2, s3, false);
        int** s5s7s = addMatrices(m, s5, s7, false);
        c22 = addMatrices(m, s2s3s, s5s7s, true);
        freeMatrix(m, s2s3s);
        freeMatrix(m, s5s7s);
    }

#pragma omp taskwait

    freeMatrix(m, s1);
    freeMatrix(m, s2);
    freeMatrix(m, s3);
    freeMatrix(m, s4);
    freeMatrix(m, s5);
    freeMatrix(m, s6);
    freeMatrix(m, s7);

    int** prod = combineMatrices(m, c11, c12, c21, c22);

    freeMatrix(m, c11);
    freeMatrix(m, c12);
    freeMatrix(m, c21);
    freeMatrix(m, c22);
    return prod;
}


void PrintMatrix(int n, int** mat)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}





int main(int argc, char** argv) {

   
    int MODE;
    cout << "ENTER MODE : 0,1,2,3,4,5,6\n";
    cin >> MODE;
    int** matrix1;
    int** matrix2;
    int** result;

    if (MODE < 3) {
        int n, m, t, g;
        int l = 0;

        matrix1 = readMatrix("first.txt");
        n = ROWS;
        g = COLS;

        WriteMatrix(matrix1, "first_nice.txt");
        matrix2 = readMatrix("second.txt");
        t = ROWS;
        m = COLS;

        if (g != t) { cout << "Wrong sizes of matrices!"; }
        WriteMatrix(matrix2, "second_nice.txt");

        int** result;
        result = new int* [n];
        for (l = 0; l < n; l++)
            result[l] = new int[m];

        ROWS = n;
        COLS = m;


        std::ofstream fout("output.txt");
        if (!fout.is_open()) { std::cerr << "Can't open output file!" << std::endl;    exit(1); }
        fout << n << " " << m << " ";

        int i, j, k;

        switch (MODE) {


        case 0: {


         //   auto start = std::chrono::steady_clock::now();
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    result[i][j] = 0;



                    for (k = 0; k < g; k++) {
                        result[i][j] += (matrix1[i][k] * matrix2[k][j]);

                    }
                    fout << result[i][j] << " ";
                }
            }
         //   auto end = std::chrono::steady_clock::now();
            fout.close();

          //  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
          //  std::cout << " 0 It tooks " << elapsed.count() << " microseconds." << std::endl;

            WriteMatrix(result, "output_nice.txt");
            break; }

        case 1: {

            int threadsNum = 30;
            omp_set_num_threads(threadsNum);


            auto start = std::chrono::steady_clock::now();


#pragma omp parallel for shared(matrix1, matrix2, result) private(i, j, k)                      //NATIVE 

            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    result[i][j] = 0;



                    for (k = 0; k < g; k++) {
                        result[i][j] += (matrix1[i][k] * matrix2[k][j]);

                    }
                    fout << result[i][j] << " ";
                }
            }
            auto end = std::chrono::steady_clock::now();
            fout.close();

            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "1 It tooks " << elapsed.count() << " microseconds." << std::endl;

            WriteMatrix(result, "output_nice.txt");
            break; }


        case 2: {
            int threadsNum = 30;
            omp_set_num_threads(threadsNum);
            int i, j, k;
            auto start = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic) shared(matrix1, matrix2, result) private(i, j, k) //OPTIMISED

            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    result[i][j] = 0;

                    for (k = 0; k < g; k++) {
                        result[i][j] += (matrix1[i][k] * matrix2[k][j]);

                    }
                    fout << result[i][j] << " ";
                }
            }
            auto end = std::chrono::steady_clock::now();
            fout.close();

            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "2 It tooks " << elapsed.count() << " microseconds." << std::endl;

            WriteMatrix(result, "output_nice.txt");
            break; }

        }
    }





        ///int N;
        if ((MODE > 2) && (MODE < 7)) {
            int N;
            cout << "Put the size of matrices ( one number )   " << endl;
            cin >> N;
            matrix1 = NewMatrix(N);
            matrix2 = NewMatrix(N);
            fillMatrix(N, matrix1);
            fillMatrix(N, matrix2);
            int i, j, k;
            result = NewMatrix(N);
            int n, m, g;
            n = m = g = N;

            switch (MODE) {
            case 3: {


               // auto start = std::chrono::steady_clock::now();
                for (i = 0; i < n; i++) {
                    for (j = 0; j < m; j++) {
                        result[i][j] = 0;

                        for (k = 0; k < g; k++) {
                            result[i][j] += (matrix1[i][k] * matrix2[k][j]);

                        }

                    }
                }
            //    auto end = std::chrono::steady_clock::now();


              //  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
             //   std::cout << "3 It tooks " << elapsed.count() << " microseconds." << std::endl;
             //


                cout << "MATRIX1 =   " << endl;
                PrintMatrix(N, matrix1);
                cout << "MATRIX2 =   " << endl;
                PrintMatrix(N, matrix2);
                cout << "RESULT =   " << endl;
                PrintMatrix(N, result);
                break; }


            case 4: {

                int threadsNum = 30;
                omp_set_num_threads(threadsNum);


                auto start = std::chrono::steady_clock::now();


#pragma omp parallel for shared(matrix1, matrix2, result) private(i, j, k)                      //NATIVE 

                for (i = 0; i < n; i++) {
                    for (j = 0; j < m; j++) {
                        result[i][j] = 0;



                        for (k = 0; k < g; k++) {
                            result[i][j] += (matrix1[i][k] * matrix2[k][j]);

                        }

                    }
                }
                auto end = std::chrono::steady_clock::now();


                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "It tooks " << elapsed.count() << " microseconds." << std::endl;

                cout << "MATRIX1 =   "<<"endl";
                PrintMatrix(N, matrix1);
                cout << "MATRIX2 =   " << "endl";
                PrintMatrix(N, matrix2);
                cout << "RESULT =   " << "endl";
                PrintMatrix(N, result);
                break;

            }



            case 5: {
                int threadsNum = 30;
                omp_set_num_threads(threadsNum);
                int i, j, k;
                auto start = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic) shared(matrix1, matrix2, result) private(i, j, k) //OPTIMISED

                for (i = 0; i < n; i++) {
                    for (j = 0; j < m; j++) {
                        result[i][j] = 0;

                        for (k = 0; k < g; k++) {
                            result[i][j] += (matrix1[i][k] * matrix2[k][j]);

                        }

                    }
                }
                auto end = std::chrono::steady_clock::now();


                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "It tooks " << elapsed.count() << " microseconds." << std::endl;


                cout << "MATRIX1 =   " << "endl";
                PrintMatrix(N, matrix1);
                cout << "MATRIX2 =   " << "endl";
                PrintMatrix(N, matrix2);
                cout << "RESULT =   " << "endl";
                PrintMatrix(N, result);
                break;

            }

            case 6: {
                if (N % 2 != 0) 
                    printf("Please enter N%2 = 0 size of matrice \n");
                int** mat1 = NewMatrix(N);
                fillMatrix(N, mat1);

                int** mat2 = NewMatrix(N);
                fillMatrix(N, mat2);
                int** prod;
                int threadsNum = 8;
                omp_set_num_threads(threadsNum);

                auto start = std::chrono::steady_clock::now();


#pragma omp parallel
                { 
#pragma omp single
                    {
 
                        prod = strassen(N, mat1, mat2);
                    }
                }
                auto end = std::chrono::steady_clock::now();


                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "It tooks " << elapsed.count() << " microseconds." << std::endl;

                cout << "MATRIX1 =   "<<endl;
                PrintMatrix(N, mat1);
                cout << "MATRIX2 =   "<<endl;
                PrintMatrix(N, mat2);
                cout << "RESULT =   "<<endl;
                PrintMatrix(N, prod);



                break;
            }
            }
        }
    return 0;
}
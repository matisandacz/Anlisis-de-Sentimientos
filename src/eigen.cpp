#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    double eigenvalue;
    
    for(unsigned i = 0; i < num_iter ; i++){
      Vector Xb = X*b;
      b = Xb/Xb.norm();
    }
    eigenvalue = (X*b)[0]/(b[0]);
    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for(unsigned i = 0; i < num; i++){
      pair<double, Vector> primeros = power_iteration(A, num_iter, epsilon);
      eigvalues[i] = primeros.first;
      for(unsigned j = 0; j < A.rows(); j++)
        eigvectors(j,i) = primeros.second[j];
      A -= primeros.second.transpose()*(primeros.second);
    }

    return make_pair(eigvalues, eigvectors);
}

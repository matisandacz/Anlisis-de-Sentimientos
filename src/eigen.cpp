#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    double eigenvalue = 0;
    Vector b = Vector::Random(X.cols());
    for(unsigned i = 0; i < num_iter ; i++){
      Vector Xb = X*b;
      b = Xb/(Xb.norm());
      double new_eigenvalue = (b.transpose()*(X*b));
      new_eigenvalue /= b.transpose()*b;
      double dif = new_eigenvalue - eigenvalue;
      eigenvalue = new_eigenvalue;
      if((dif < 0 ? -dif : dif) < eps)
        break;
    }
    return make_pair(eigenvalue, b);
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
        eigvectors(j,i) = (primeros.second)[j];
      A -= primeros.first*(primeros.second*(primeros.second.transpose()));
    }

    return make_pair(eigvalues, eigvectors);
}

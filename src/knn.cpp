#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
  _n_neighbors = n_neighbors;
}

void KNNClassifier::fit(SparseMatrix X, Matrix y)
{
  cout << y;
  _data = X;
  _labels = y;
}


Vector KNNClassifier::predict(SparseMatrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows()); //va a tener un 0 o un 1 segun la prediccion para cada fila de X

    //para cada fila de X calculo la distancia con todas las filas de mi dataset
    cout << X.rows() << endl;
    for (unsigned k = 0; k < X.rows(); ++k)
    {
        if(k % 10 == 0){
          cout << k << endl;
        }
        Vector dist(_data.rows());
        for(unsigned i = 0; i < _data.rows(); i++)
          dist[i] = (_data.row(i) - X.row(i)).norm();
        intVector indices(_n_neighbors);
        for(unsigned i = 0; i < _n_neighbors; i++){
          double min = dist[0];
          unsigned arg_min = 0;
          for(unsigned j = 0; j < _data.rows()-i; j++){
            if(dist[j] < min){
              min = dist[j];
              arg_min = j;
            }
          }
          dist[arg_min] = dist[_data.rows()-1-i];
          dist[_data.rows()-1-i] = min;
          indices[i] = arg_min;
        }
        unsigned cantidad_positivas = 0;
        for(unsigned i = 0; i < _n_neighbors; i++){
          if(_labels(indices[i]) == 1.0)
            cantidad_positivas++;
        }
        if(cantidad_positivas > _n_neighbors/2)
          ret(k) = 1.0;
        else
          ret(k) = 0.0;
    }

    return ret;
}

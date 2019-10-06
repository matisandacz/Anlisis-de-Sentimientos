#include <algorithm>
//#include <chrono>
#include <iostream>
#include <queue>
#include "wdknn.h"

using namespace std;


double WDKNNClassifier::peso_vecino(double distancia, int vecino) {
  return 1.0;
}

WDKNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
  _n_neighbors = n_neighbors;
}

void WDKNNClassifier::fit(SparseMatrix X, Matrix y)
{
  _data = X;
  _labels = y;
}


Vector WDKNNClassifier::predict(SparseMatrix X)
{
  
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows()); //va a tener un 0 o un 1 segun la prediccion para cada fila de X

    //para cada fila de X calculo la distancia con todas las filas de mi dataset
    for (unsigned k = 0; k < X.rows(); ++k)
    {

      priority_queue <double> vecinosLabelPos;
      priority_queue <double> vecinosLabelNeg;

      for(unsigned i = 0; i < _data.rows(); i++) {
        double d;
        d = (_data.row(i) - X.row(k)).norm();

        if (_labels(0,i) == 1)
        {
          vecinosLabelPos.push(d);
        }
        else {
          vecinosLabelNeg.push(d);
        }

        if(vecinosLabelPos.size() > _n_neighbors) {
          vecinosLabelPos.pop();
        }

        if(vecinosLabelNeg.size() > _n_neighbors) {
          vecinosLabelNeg.pop();
        }
      }

      unsigned double pesos_positivo = 0;
      unsigned double pesos_negativo = 0;


      for(unsigned i = _n_neighbors-1; i >= 0; i--) {
        double d = vecinosLabelPos.top();
        pesos_positivo += peso_vecino(d,i);
        vecinosLabelPos.pop();

        d = vecinosLabelNeg.top();
        pesos_negativo += peso_vecino(d,i);
        vecinosLabelNeg.pop();

      }

      if(pesos_positivo < pesos_negativo)
        ret(k) = 1.0;
      else
        ret(k) = 0.0;
    }

    return ret;
}

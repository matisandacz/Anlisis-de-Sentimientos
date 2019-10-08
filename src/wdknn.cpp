#include <algorithm>
//#include <chrono>
#include <iostream>
#include <queue>
#include "wdknn.h"

using namespace std;


double WDKNNClassifier::peso_vecino(double distancia, int vecino) {
  return 1.0*distancia*distancia;
}

WDKNNClassifier::WDKNNClassifier(unsigned int n_neighbors)
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
    Vector ret = Vector(X.rows()); //va a tener un 0 o un 1 segun la prediccion para cada fila de X

    //para cada fila de X calculo la distancia con todas las filas de mi dataset
    for (unsigned k = 0; k < X.rows(); ++k)
    {
      priority_queue <double> vecinosLabelPos;
      priority_queue <double> vecinosLabelNeg;  

      // recorremos la data y vamos modificando las queue para que
      // tengan las menores _n_neighbors distancias de cada clase
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
        };
      }


      double pesos_positivo = 0.0;
      double pesos_negativo = 0.0;


      // sacamos los mas alejados
      while(vecinosLabelPos.size() + vecinosLabelNeg.size() >  _n_neighbors ) {
        if (vecinosLabelPos.top() > vecinosLabelNeg.top())
        {
          vecinosLabelPos.pop();
          
        }
        else {
          vecinosLabelNeg.pop();          
        }

      }

      // se van sacando hasta que uno se vacia
      while(vecinosLabelPos.size()*vecinosLabelNeg.size() >  0) {
        if (vecinosLabelPos.top() > vecinosLabelNeg.top())
        {
          pesos_positivo += peso_vecino(vecinosLabelPos.top(),vecinosLabelPos.size() + vecinosLabelNeg.size()-1);
          vecinosLabelPos.pop();
          
        }
        else {
          pesos_negativo += peso_vecino(vecinosLabelNeg.top(),vecinosLabelPos.size() + vecinosLabelNeg.size()-1);
          vecinosLabelNeg.pop();          
        }

      }


      // sacamos todos los vecinos
      while(vecinosLabelPos.size() >  0) {
        pesos_positivo += peso_vecino(vecinosLabelPos.top(),vecinosLabelPos.size()-1);
        vecinosLabelPos.pop();
      }

      // sacamos todos los vecinos
      while(vecinosLabelNeg.size() >  0) {
        pesos_negativo += peso_vecino(vecinosLabelNeg.top(),vecinosLabelNeg.size()-1);
        vecinosLabelNeg.pop();
      }


      if(pesos_positivo > pesos_negativo) {
        cout << " predigo positivo" << endl;
        ret(k) = 1.0;
      }
      else {
        cout << " predigo negativo" << endl;
        ret(k) = 0.0;
      }
    }

    return ret;
}

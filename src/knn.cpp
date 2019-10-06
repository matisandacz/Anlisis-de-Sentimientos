#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <math.h>
#include <queue> 


using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
  _n_neighbors = n_neighbors;
}

void KNNClassifier::fit(SparseMatrix X, Matrix y)
{
  _data = X;
  _labels = y;
}


Vector KNNClassifier::predict(SparseMatrix X)
{
    // Creamos vector columna a devolver
  auto ret = Vector(X.rows()); //va a tener un 0 o un 1 segun la prediccion para cada fila de X

  //para cada fila de X calculo la distancia con todas las filas de mi dataset
  for (unsigned k = 0; k < X.rows(); ++k) {
    if(k % 10 == 0){
      cout << "%" << ((double)k / (double)X.rows())*100 << endl;
    }

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

    // sacamos el mas lejano hasta que se tenga _n_neighbors vecinos
    while(vecinosLabelPos.size() + vecinosLabelNeg.size() >  _n_neighbors) {
      if (vecinosLabelPos.top() > vecinosLabelNeg.top())
      {
        vecinosLabelPos.pop()
      }
      else {
        vecinosLabelNeg.pop()
      }
    }

    // gana el que tenga mas de los primeros _n_neighbors vecinos
    if(vecinosLabelPos.size() > vecinosLabelNeg.size())
      ret(k) = 1.0;
    else
      ret(k) = 0.0;
  }

    return ret;
}

Vector KNNClassifier::predict_weighted(SparseMatrix X,const Vector& covarianzas)
{

  // Creamos vector columna a devolver
  auto ret = Vector(X.rows()); //va a tener un 0 o un 1 segun la prediccion para cada fila de X

  //para cada fila de X calculo la distancia con todas las filas de mi dataset
  for (unsigned k = 0; k < X.rows(); ++k) {
    if(k % 10 == 0){
      cout << "%" << ((double)k / (double)X.rows())*100 << endl;
    }

    priority_queue <double> vecinosLabelPos;
    priority_queue <double> vecinosLabelNeg;

    // recorremos la data y vamos modificando las queue para que
    // tengan las menores _n_neighbors distancias de cada clase
    for(unsigned i = 0; i < _data.rows(); i++) {
      double d;
      d = weighted_norm(_data.row(i) - X.row(k),covarianzas);

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

    // sacamos el mas lejano hasta que se tenga k vecinos
    while(vecinosLabelPos.size() + vecinosLabelNeg.size() >  _n_neighbors) {
      if (vecinosLabelPos.top() > vecinosLabelNeg.top())
      {
        vecinosLabelPos.pop()
      }
      else {
        vecinosLabelNeg.pop()
      }
    }

    // gana el que tenga mas de los primeros _n_neighbors vecinos
    if(vecinosLabelPos.size() > vecinosLabelNeg.size())
      ret(k) = 1.0;
    else
      ret(k) = 0.0;
  }

  return ret;
}

double KNNClassifier::weighted_norm(const Vector& v,const Vector& pesos){
  double res = 0;
  for(int i = 0; i < v.size(); i++){
    res += v[i]*v[i]*pesos[i];
  }
  return sqrt(res);
}
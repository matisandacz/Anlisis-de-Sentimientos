#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <math.h>
#include <queue>


using namespace std;

class CompareDist
{
public:
    bool operator()(pair<double, double> n1,pair<double,double> n2) {
        return n1.first > n2.first; //hay que usar mayor por cosas de priority queue
    }
};

KNNClassifier::KNNClassifier(unsigned int n_neighbors) : _n_neighbors(n_neighbors){
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
      cout << "%" << (double)k/(double)X.rows() * 100 << "\r";
      cout.flush();
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
        vecinosLabelPos.pop();
      }
      else {
        vecinosLabelNeg.pop();
      }
    }

    // gana el que tenga mas de los primeros _n_neighbors vecinos
    if(vecinosLabelPos.size() > vecinosLabelNeg.size())
      ret(k) = 1.0;
    else
      ret(k) = 0.0;
  }
  cout << "\n";

  return ret;
}

Matrix KNNClassifier::testearK(SparseMatrix X)
{
  // Creamos vector columna a devolver
  Matrix res(X.rows(), _data.rows()); //va a tener un 0 o un 1 segun la prediccion para cada fila de X por cada k

  //para cada fila de X calculo la distancia con todas las filas de mi dataset
	for (unsigned k = 0; k < X.rows(); ++k) {
		if(k % 10 == 0){
			cout << "%" << (double)k/(double)X.rows() * 100 << "\r";
			cout.flush();
		}

		priority_queue<pair<double,double>,vector<pair<double,double>>,CompareDist> vecinos;

		// recorremos la data y vamos modificando las queue para que
		// tengan las menores _n_neighbors distancias de cada clase
		for(unsigned i = 0; i < _data.rows(); i++) {
			double d;
			d = (_data.row(i) - X.row(k)).norm();

			pair<double, double> distEt;
			distEt = make_pair(d, _labels(0, i));
			vecinos.push(distEt);
		}

		unsigned int positivos = 0;
		unsigned int negativos = 0;

		for(unsigned int i = 0; i < _data.rows() ; i++) {
			if((vecinos.top()).second == 1)
				positivos++;
			else
				negativos++;
			vecinos.pop();
			res(k, i) = (positivos > negativos)? 1.0 : 0.0;
		}
	}
	cout << "\n";

	return res;
}
Matrix KNNClassifier::testearK_weighted(SparseMatrix X,const Vector& correlaciones)
{
    // Creamos vector columna a devolver
    Matrix res(X.rows(), _data.rows()); //va a tener un 0 o un 1 segun la prediccion para cada fila de X por cada k

  //para cada fila de X calculo la distancia con todas las filas de mi dataset
	for (unsigned k = 0; k < X.rows(); ++k) {
		if(k % 10 == 0){
			cout << "%" << (double)k/(double)X.rows() * 100 << "\r";
			cout.flush();
		}

		priority_queue<pair<double,double>,vector<pair<double,double>>,CompareDist> vecinos;

		// recorremos la data y vamos modificando las queue para que
		// tengan las menores _n_neighbors distancias de cada clase
		for(unsigned i = 0; i < _data.rows(); i++) {
			double d;
			d = weighted_norm(_data.row(i) - X.row(k),correlaciones);

			pair<double, double> distEt;
			distEt = make_pair(d, _labels(0, i));
			vecinos.push(distEt);
		}

		unsigned int positivos = 0;
		unsigned int negativos = 0;

		for(unsigned int i = 0; i < _data.rows() ; i++) {
			if((vecinos.top()).second == 1)
				positivos++;
			else
				negativos++;
			vecinos.pop();

			res(k, i) = (positivos > negativos)? 1.0 : 0.0;
		}
	}
	cout << "\n";

	return res;
}

Vector KNNClassifier::predict_weighted(SparseMatrix X,const Vector& correlaciones)
{

  // Creamos vector columna a devolver
  auto ret = Vector(X.rows()); //va a tener un 0 o un 1 segun la prediccion para cada fila de X

  //para cada fila de X calculo la distancia con todas las filas de mi dataset
  for (unsigned k = 0; k < X.rows(); ++k) {
    if(k % 10 == 0){
      cout << "%" << (double)k/(double)X.rows() * 100 << "\r";
      cout.flush();
    }
    priority_queue <double> vecinosLabelPos;
    priority_queue <double> vecinosLabelNeg;

    // recorremos la data y vamos modificando las queue para que
    // tengan las menores _n_neighbors distancias de cada clase
    for(unsigned i = 0; i < _data.rows(); i++) {
      double d;
      d = weighted_norm(_data.row(i) - X.row(k),correlaciones);

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
        vecinosLabelPos.pop();
      }
      else {
        vecinosLabelNeg.pop();
      }
    }

    // gana el que tenga mas de los primeros _n_neighbors vecinos
    if(vecinosLabelPos.size() > vecinosLabelNeg.size())
      ret(k) = 1.0;
    else
      ret(k) = 0.0;
  }

  cout << "\n";
  return ret;
}

double KNNClassifier::weighted_norm(const Vector& v,const Vector& pesos){
  double res = 0;
  for(int i = 0; i < v.size(); i++){
    res += v[i]*v[i]*pesos[i];
  }
  return sqrt(res);
}

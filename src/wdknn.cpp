#include <algorithm>
//#include <chrono>
#include <iostream>
#include "wdknn.h"
#include <queue>

using namespace std;

#define MAYORITY 0
#define INVERSE 1
#define ZABREL 2
#define FIBONACCI 3
#define DUDANI 4




void WDKNNClassifier::mayority_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo) {
  pesos_positivo = vecinosLabelPos.size();
  pesos_negativo = vecinosLabelNeg.size();
}


void WDKNNClassifier::inverse_distance_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo) {
  while (vecinosLabelPos.size() > 0) {
    pesos_positivo += 1.0/vecinosLabelPos.top();
    vecinosLabelPos.pop();
  }

  while (vecinosLabelNeg.size() > 0) {
    pesos_negativo += 1.0/vecinosLabelNeg.top();
    vecinosLabelNeg.pop();
  }
}

void WDKNNClassifier::zabrel_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo){

  while (vecinosLabelPos.size() > 0) {
    pesos_positivo += exp(- vecinosLabelPos.top());
    vecinosLabelPos.pop();
  }

  while (vecinosLabelNeg.size() > 0) {
    pesos_negativo += exp(- vecinosLabelNeg.top());
    vecinosLabelNeg.pop();
  }
}


void WDKNNClassifier::fibonacci_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo) {  
      int contador = 0;

      // se van sacando hasta que uno se vacia
      while(vecinosLabelPos.size()*vecinosLabelNeg.size() >  0) {
        if (vecinosLabelPos.top() > vecinosLabelNeg.top())
        {
          pesos_positivo += num_fibo[contador];
          vecinosLabelPos.pop();
          
        }
        else {
          pesos_negativo += num_fibo[contador];
          vecinosLabelNeg.pop();          
        }
        contador++;

      }


      // sacamos todos los vecinos
      while(vecinosLabelPos.size() >  0) {
        pesos_positivo += num_fibo[contador];
        vecinosLabelPos.pop();
        contador++;
      }

      // sacamos todos los vecinos
      while(vecinosLabelNeg.size() >  0) {
        pesos_negativo += num_fibo[contador];
        vecinosLabelNeg.pop();
        contador++;
      }
}

void WDKNNClassifier::dudani_vote(priority_queue<double>& vecinosLabelPos, priority_queue<double>& vecinosLabelNeg, double& pesos_positivo, double& pesos_negativo) {
  int total_pos = vecinosLabelPos.size();
  int total_neg = vecinosLabelNeg.size();


  while (vecinosLabelPos.size() > 1) {
    pesos_positivo +=  - vecinosLabelPos.top();
    vecinosLabelPos.pop();
  }

  while (vecinosLabelNeg.size() > 1) {
    pesos_negativo += -vecinosLabelNeg.top();
    vecinosLabelNeg.pop();
  }

  double d_k = 0;
  if (vecinosLabelPos.size() > 0) {
    d_k = vecinosLabelPos.top();
    pesos_positivo +=  - vecinosLabelPos.top();
    vecinosLabelPos.pop();  
  }

  if (vecinosLabelNeg.size() > 0) {
    if (d_k < vecinosLabelNeg.top())
    {
      d_k = vecinosLabelNeg.top();
    }

    pesos_negativo +=  -vecinosLabelNeg.top();
    vecinosLabelNeg.pop();  
  }

  pesos_positivo += total_pos*d_k;
  pesos_negativo += total_neg*d_k;

  // no es necesario dividir ya que eso no afecta el ganador.
}



// setear funcion de votacion
void WDKNNClassifier::set_mayority() {
    vote_rule = MAYORITY;
}
void WDKNNClassifier::set_inverse_distance() {
    vote_rule = INVERSE;
}
void WDKNNClassifier::set_dudani() {
    vote_rule = DUDANI;
}
void WDKNNClassifier::set_zabrel() {
    vote_rule = ZABREL;
}
void WDKNNClassifier::set_fibonacci() {
    vote_rule = FIBONACCI;
}


WDKNNClassifier::WDKNNClassifier(unsigned int n_neighbors)
{
  _n_neighbors = n_neighbors;
  vote_rule = MAYORITY;

  num_fibo.resize(n_neighbors, 1.0);

  for (unsigned int i = 2; i < n_neighbors; ++i)
  {
    num_fibo[i] = num_fibo[i-1] + num_fibo[i-2];
  }
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

      // se somete a votacion segun la regla que se tenga
      switch (vote_rule) {
        case MAYORITY: mayority_vote(vecinosLabelPos, vecinosLabelNeg, pesos_positivo, pesos_negativo); break;
        case INVERSE: inverse_distance_vote(vecinosLabelPos, vecinosLabelNeg, pesos_positivo, pesos_negativo); break;     
        case ZABREL: zabrel_vote(vecinosLabelPos, vecinosLabelNeg, pesos_positivo, pesos_negativo); break;
        case FIBONACCI: fibonacci_vote(vecinosLabelPos, vecinosLabelNeg, pesos_positivo, pesos_negativo); break;
        case DUDANI: dudani_vote(vecinosLabelPos, vecinosLabelNeg, pesos_positivo, pesos_negativo); break;      
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

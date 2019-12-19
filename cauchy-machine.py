#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

class CM:
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True
        """
        Inicialice una matriz de peso, de dimensiones (num_visible x num_hidden), utilizando
        una distribución uniforme entre -sqrt(6.0/(num_hidden+num_visible))
        y sqrt(6.0/(num_hidden+num_visible)). Se podría variar la desviación 
        estándar multiplicando el intervalo con el valor apropiado. 
        Aquí inicializamos los pesos con media 0 y desviación estándar 0.1. 
        """
        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

        # Inserte pesos para las unidades de sesgo en la primera fila y la primera columna.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)


        # Entrenamiento de la máquina.
    def train(self, data, max_epochs=1000, learning_rate=0.1):
        # Parametros de entrada: data: Una matriz donde cada fila es un ejemplo de entrenamiento
        # que consiste en los estados de unidades visibles.
        num_examples = data.shape[0]

        # Inserte unidades de sesgo de 1 en la primera columna.
        data = np.insert(data, 0, 1, axis=1)

        for epoch in range(max_epochs):
            # Sujete los datos y la muestra de las unidades ocultas.
            # (Esta es la "fase CD positiva", también conocida como la fase reality).
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:, 0] = 1  # Arregle la unidad de sesgo
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            """
            Tener en cuenta que estamos usando las *probabilidades* de activación de los estados ocultos,
            no los estados ocultos en sí mismos, al calcular las asociaciones. También podríamos usar los estados
            pero en este caso no lo haremos de esta forma.
            """
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # Reconstruimos las unidades visibles y muestrear nuevamente a partir de las unidades ocultas.
            # (Esta es la "fase de negative CD", también conocida como la fase de daydream).
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Arregle la unidad de sesgo.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Actualizamos los pesos
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - neg_visible_probs) ** 2)
            if self.debug_print:
                print("Epoch %s: error es %s" % (epoch, error))

        # Asumiendo que el CM ha sido entrenado (para que se hayan aprendido los pesos de la red),
        # ejecute la red en un conjunto de unidades visibles, para obtener una muestra de las unidades ocultas.
    def run_visible(self, data):
        # Parametros de entrada: data: Una matriz donde cada fila consiste en los estados de las unidades visibles.
        num_examples = data.shape[0]

        # Creamos una matriz, donde cada fila debe ser las unidades ocultas (más una unidad de sesgo)
        # muestreado de un ejemplo de entrenamiento.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        # Insertamos unidades de sesgo de 1 en la primera columna de datos.
        data = np.insert(data, 0, 1, axis=1)

        # Calculamos las activaciones de las unidades ocultas.
        hidden_activations = np.dot(data, self.weights)
        # Calculamos las probabilidades de encender las unidades ocultas.
        hidden_probs = self._logistic(hidden_activations)
        # Encendemos las unidades ocultas con sus probabilidades específicas.
        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Siempre arregle la unidad de polarización a 1.
        # hidden_states[:,0] = 1

        # Ignora las unidades de sesgo.
        hidden_states = hidden_states[:, 1:]
        return hidden_states
        # Retorna: hidden_states: Una matriz donde cada fila consta de las unidades ocultas activadas a partir
        # de las unidades visibles en la matriz de datos que se pasa.


        # Suponiendo que el CM ha sido entrenado (para que se hayan aprendido los pesos de la red),
        # ejecute la red en un conjunto de unidades ocultas, para obtener una muestra de las unidades visibles.
    def run_hidden(self, data):
        # Parametros de entrada: data: Una matriz donde cada fila consiste en los estados de las unidades ocultas.

        num_examples = data.shape[0]

        # Creamos una matriz, donde cada fila debe ser las unidades visibles (más una unidad de sesgo)
        # muestreado de un ejemplo de entrenamiento.
        visible_states = np.ones((num_examples, self.num_visible + 1))

        # Insertamos unidades de sesgo de 1 en la primera columna de datos.
        data = np.insert(data, 0, 1, axis=1)

        # Calculamos la activación de las unidades visibles.
        visible_activations = np.dot(data, self.weights.T)
        # Calculamos las probabilidades de encender las unidades visibles.
        visible_probs = self._logistic(visible_activations)
        # Encendemos las unidades visibles con sus probabilidades específicas.
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        # Fijamos siempre la unidad de polarización a 1.
        # visible_states[:,0] = 1

        # Ignora las unidades de sesgo.
        visible_states = visible_states[:, 1:]
        return visible_states
        #  Retorna: visible_states: Una matriz donde cada fila consiste en las unidades visibles activadas
        #  a partir de las unidades ocultas en la matriz de datos que se pasa.



        # Inicializamos aleatoriamente las unidades visibles una vez y comience a ejecutar pasos
        # de muestreo de Gibbs alternativos (donde cada paso consiste en actualizar todas las
        # unidades ocultas y luego actualizar todas las unidades visibles), tomando una muestra
        # de las unidades visibles en cada paso.

        # Tenga en cuenta que solo inicializamos la red una vez, por lo que estas muestras están
        # correlacionadas.
    def daydream(self, num_samples):
        # Cree una matriz, donde cada fila debe ser una muestra de las unidades visibles
        # (con una unidad de sesgo adicional), inicializado a todos.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Tome la primera muestra de una distribución uniforme.
        samples[0, 1:] = np.random.rand(self.num_visible)

        # Comience el muestreo alternativo de Gibbs.
        # Tenga en cuenta que mantenemos los estados binarios de las unidades
        # ocultas, pero dejamos las unidades visibles como probabilidades reales.
        for i in range(1, num_samples):
            visible = samples[i - 1, :]

            # Calcula la activación de las unidades ocultas.
            hidden_activations = np.dot(visible, self.weights)
            # Calcule las probabilidades de encender las unidades ocultas.
            hidden_probs = self._logistic(hidden_activations)
            # Encienda las unidades ocultas con sus probabilidades específicas.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            #Fije siempre la unidad de polarización a 1.
            hidden_states[0] = 1

            # Calcule las probabilidades de que estén las unidades visibles.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states

        # Ignora las unidades de sesgo (la primera columna), ya que siempre se establecen en 1.
        return samples[:, 1:]
        #  Devuelve: una matriz, donde cada fila es una muestra de las unidades visibles producidas mientras
        #  la red estaba daydream.

    def _logistic(self, x):
    # Al poner la función de distribución de Cauchy, se hace uso de la máquina de Cauchy
        return (1.0/2.0)+((1.0/np.pi)*np.arctan(x)) 


if __name__ == '__main__':
    r = CM(num_visible=6, num_hidden=2)
    training_data = np.array(
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0]])
    r.train(training_data, max_epochs=5000)
    print(r.weights)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print(r.run_visible(user))
# ME02 : Proyecto Final NS3.

## Propósito

Análisis, diseño e implementación de las máquinas de Boltzmann y de Cauchy.

## Integrantes

|       Integrante      |                 Correo                       |
|-----------------------|-----------------------------------------------|
| Dave Sebastian Valencia Salazar      |    <dsvalencias@unal.edu.co>    |
| Yesid Alberto Ochoa Luque      |    <yaochoal@unal.edu.co>     |

## Entregables

### 1. Código fuente la implementación (Completa).


``` fichero: ./boltzmann-machine.py```

```

class BM:

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

        # Asumiendo que el RM ha sido entrenado (para que se hayan aprendido los pesos de la red),
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


        # Suponiendo que el BM ha sido entrenado (para que se hayan aprendido los pesos de la red),
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
    # Al poner la función de distribución de Boltzmann, se hace uso de la máquina de Boltzmann
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    r = BM(num_visible=6, num_hidden=2)
    training_data = np.array(
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0]])
    r.train(training_data, max_epochs=5000)
    print(r.weights)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print(r.run_visible(user))
```

``` fichero: ./cauchy-machine.py```

```

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
        return (1.0/2.0)+((1.0/pi)*np.arctan(x)) 


if __name__ == '__main__':
    r = CM(num_visible=6, num_hidden=2)
    training_data = np.array(
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0]])
    r.train(training_data, max_epochs=5000)
    print(r.weights)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print(r.run_visible(user))
```
### 2. Manuales:

### a) Usuario

### b) Técnico

Las máquinas restringidas de Boltzmann o máquina de Cauchy son redes neuronales que pertenecen a los modelos basados en energía, no son tan conocidas como las redes neuronales convolucionales, sin embargo, este tipo de redes han ganado popularidad recientemente en el contexto del premio Netflix en donde las máquinas de Cauchy han alcanzado un rendimiento impecable en filtrado colaborativo y han vencido a la competencia.

![Figura1](/img/figura1.png)

Figura 1. Ejemplo máquina basada en energía

### Arquitectura

Las máquinas basadas en energía tienen una arquitectura bastante sencilla comparada con otros tipos de redes neuronales, como se ve en la figura 1, una máquina de estas consiste de una capa visible o de entrada (v1,...,v6), una capa oculta (h1, h2) y sus correspondientes sesgos (a,b). Por otra parte, no se requiere de una capa de salida ya que las predicciones de este modelo se realizan de una forma distinta a otros tipos de redes neuronales.

### i. Descripción detallada de modelo matemático que rige la implementación o las funciones en especial de los modelos estocásticos.

### Un modelo basado en energía

La energía no es un concepto que se asocie al aprendizaje de máquina de forma intuitiva al ser un concepto de la física; sin embargo, algunas arquitecturas de aprendizaje de máquina la noción de energía como métrica para medir la calidad de los modelos. Como uno de los principales objetivos del aprendizaje de máquina es codificar dependencias entre variables, la captura de estas dependencias se realiza al asociar una energía escalar a cada una de las configuraciones de las variables, las cuales sirven como una medida de compatibilidad. Una alta escala energética significa una baja compatibilidad, y un modelo basado en energía reduce el problema a la minimización de una función de energía predefinida y para una máquina basada en energía es la siguiente.

![Figura2](/img/figura2.png)

Figura 2. Función de energía para una máquina basada en energía

### Un modelo probabilístico

Este tipo de red neuronal se basa en un modelo probabilístico, en vez de asignar valores discretos asigna probabilidades, así la máquina estará en cierto estado en cierto instante temporal, los estados se refieren a los valores de las neuronas en las capa visible V y oculta H; y la probabilidad de ver un estado en específico de V y H, se da a través de la siguiente distribución conjunta. Además Z, indica la función de partición que no es más que la suma de todas las parejas de neuronas entre la capa visible V y la oculta H.

![Figura3](/img/figura3.png)

Figura 3. Distribución de unión en una máquina restringida de Boltzmann.

Esta distribución en física se conoce como la distribución de Boltzmann la cual le asigna a una partícula la probabilidad de ser observada en un estado con una energía E asociada, y en este caso se asocia la probabilidad de observar un estado de V y H, que depende de la energía total del modelo. Desafortunadamente es muy complejo el cálculo de estas probabilidades en la función de partición Z. Es más fácil aplicar calcular las probabilidades condicionadas al estado H dado el estado V y las probabilidades condicionadas de V dado el estado H.

![Figura4](/img/figura4.png)

Figura 4. Probabilidades condicionadas para V y H.

Ahora, dadas la probabilidades condicionadas se deben tratar de cierta forma que cumplan con un concepto básico de una neurona, que esté en un estado binario de activación, ahora dada una capa visible V, la probabilidad de que se active una neurona de la capa H se rige bajo esta función dependiendo del tipo de máquina:

![Figura5](/img/figura5.png)

Figura 5. Probabilidades de activar una neurona.

### Entrenamiento

El entrenamiento de este tipo de máquinas se puede sintetizar a dos pasos.

### Muestreo de Gibbs

Este paso consiste en tener un vector V para poder predecir los valores ocultos H, luego usamos la probabilidad para obtener nuevos valores de entrada V, se repite el proceso k veces dando como resultado un vector de entrada V_k el cual es una recreación del vector original V_0

![Figura6](/img/figura6.png)

Figura 6. Proceso de recreación de V_0 a V_k usando muestreo de Gibbs.

### Divergencia por contraste

Por último, la actualización de la matriz de pesos ocurre durante el paso de divergencia por contraste.
Los vectores V_0 y V_k son usados para calcular las probabilidades de activación de los valores ocultos H_0 y H_k. La diferencia entre el producto cruz de estas probabilidades con entradas V_0 y V_k dan como resultado el siguiente gradiente:

![Figura7](/img/figura7.png)

Figura 7. Actualización de la gradiente de la matriz de pesos.

Usando este gradiente se puede calcular el peso actualizado usando esta ecuación.

![Figura8](/img/figura8.png).

Figura 8. Actualización matriz de pesos.

### ii. Diseños completos de la implementación.

Para el desarrollo de la aplicación se utilizó la programación orientada a objetos con el fin de reproducir fielmente los conceptos teóricos introducidos en la sección previa, a nivel código ambos tipos de máquinas tienen una estructura similar, el único cambio que tuvo fue la función de distribución de activación de una neurona, usando la distribución de Boltzmann o Cauchy para diferenciarlo. Durante el desarrollo de la aplicación se aprovechó de la existencia de la metodología Extreme Programming (XP), y herramientas de videoconferencia como Hangouts y Discord.

### iii. Escenarios de pruebas 

### Máquina de Cauchy y Máquina de Boltzmann.

Primero inicializamos una de las maquinas 

```bm = BM(num_visible = 6, num_hidden = 2)```

o

```cm = CM(num_visible = 6, num_hidden = 2)```

posteriormente entrenamos la máquina.
```training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])```
Donde una matriz de 6x6 es donde cada fila es un ejemplo de entrenamiento y la columna es una unidad visible

```cm.train (training_data, max_epochs = 5000)```
No ejecutar el entrenamiento durante más de 5000 épocas o se saturara por memoria.
y luego ejecutamos el compilador de Python.
```./cauchy-machine.py``` o ```./boltzmann-machine.py```
![prueba](/img/prueba.jpg)
### iv. Estudio comparativo del desempeño de los modelos.

Desafortunadamente, no se pudo presentar un caso concreto por limitantes computacionales para la ejecución de este programa, ya que este tipo de código requiere tener una GPU de alta gama y RAM de bastante capacidad (>16 GB), como contramedida se opta por una pequeña prueba con pocas variables.
Después de varias iteraciones, se pudo concluir que la máquina de Cauchy efectivamente es una versión mejorada de la máquina de Boltzmann ya que la diferencia radica en las épocas utilizadas para poder empezar a inferir patrones en el comportamiento, mientras que aplicando Boltzmann se demora 1500~1600 épocas para asegurar un aprendizaje e infiera comportamientos; en cambio, aplicando Cauchy este toma entre 700 y 750 épocas para asegurar el aprendizaje.

### 3. Presentación.
![slide1](/img/pp1.jpg)
![slide2](/img/pp2.jpg)
![slide3](/img/pp3.jpg)
![slide4](/img/pp4.jpg)
![slide5](/img/pp5.jpg)
![slide6](/img/pp6.jpg)
![slide7](/img/pp7.jpg)
![slide8](/img/pp8.jpg)
![slide9](/img/pp9.jpg)
![slide10](/img/pp10.jpg)
![slide11](/img/pp11.jpg)
![slide12](/img/pp12.jpg)
![slide13](/img/pp13.jpg)
![slide14](/img/pp14.jpg)

### 4. Referencias.
- [C1.4 Stochastic neural networks](https://pdfs.semanticscholar.org/3c03/7ba6a431ed86d664af410b5fcfa64fbeaf21.pdf) 
- [Introduction to Restricted Boltzmann Machines](http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/)
- [Deep Learning meets Physics: Restricted Boltzmann Machines Part I](https://towardsdatascience.com/deep-learning-meets-physics-restricted-boltzmann-machines-part-i-6df5c4918c15)
- [Deep Learning meets Physics: Restricted Boltzmann Machines Part II](https://towardsdatascience.com/deep-learning-meets-physics-restricted-boltzmann-machines-part-ii-4b159dce1ffb)
- [REDES NEURONALES ARTIFICIALES - Fundamentos, Modelos y aplicaciones pag.285-302](#)
- [Restricted Boltzmann Machines for Collaborative Filtering](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)

## Developing a mental model for how GPTs work

#### Traditional Software vs Machine Learning
:books: In traditional software, the rules are explicitly programmed by humans whereas in machine learning, the model learns patterns from data
<br/><br/>
![trad_sw_vs_ml.png](images/trad_sw_vs_ml.png)
<br/><br/>

#### An ML Model is a Mathematical Function
:books: A Machine Learning Model is a mathematical function that maps its inputs to outputs based on patterns learned during training. For instance below is the mathematical function for early GPT models from 2019
![GPT2_equation.png](images/GPT2_equation.png)
:books: The size of this mathematical function ( i.e number of parameters ) is a key factor in the model's capacity to learn complex patterns from data. Below graph illustrates the number of parameters of these mathematical functions over time. 
![model_size_growth.png](images/model_size_growth.png)


## How do we learn the parameters of these mathematical functions?
:books: Machine Learning Development includes 2 main phases: Training and Inference, during training the model learns the parameters of the mathematical function ( such as the one above ), during inference the model uses the learned parameters to make predictions on new data
:books: We first define a loss function which is a mathematical function which sets a target for the model![loss_func.png](images/loss_func.png)

:books: Let's work through an example to illustrate an ML training process and the loss function. Our aim is to develop a model to distinguish between 3 types of Iris Species - https://en.wikipedia.org/wiki/Iris_(plant)
<br/><br/>
:books:  [iris_mlp.ipynb](notebooks/iris_mlp.ipynb)
:books: Here is markdown version of the same if you dont intend to run the code: [iris_mlp.md](notebooks/iris_mlp.md) 
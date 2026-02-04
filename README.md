## Developing a mental model for how GPTs work

#### Traditional Software vs Machine Learning
:books: In traditional software, rules are explicitly programmed by humans. In machine learning, the model learns patterns from data.
<br/><br/>
![trad_sw_vs_ml.png](images/trad_sw_vs_ml.png)
<br/><br/>

#### An ML Model is a Mathematical Function
:books: A machine learning model is a mathematical function that maps inputs to outputs based on patterns learned during training. Shown below is the mathematical function for early GPT models (2019).
![GPT2_equation.png](images/GPT2_equation.png)
:books: The size of this function—i.e., the number of parameters—is a key factor in the model’s capacity to learn complex patterns. The graph below shows how the number of parameters in these models has grown over time.
![model_size_growth.png](images/model_size_growth.png)


## How do we learn the parameters of these mathematical functions?
:books: ML development has two main phases: **training** and **inference**. During training, the model learns the parameters of the mathematical function (such as the one above). During inference, it uses those learned parameters to make predictions on new data.
:books: We first define a **loss function**—a mathematical function that sets a target for the model.
![loss_func.png](images/loss_func.png)

:books: Let’s work through an example to illustrate the ML training process and the role of the loss function. The goal is to build a model that distinguishes among three Iris species—[Setosa, Versicolor, and Virginica](https://en.wikipedia.org/wiki/Iris_(plant)).
![iris_io.png](images/iris_io.png)![iris.png](images/iris.png)
<br/><br/>
- To run the code yourself, open [iris_mlp.ipynb](notebooks/iris_mlp.ipynb). Or expand the section below for a markdown version of the same content. 

<details>
  <summary>Predicting Iris Species</summary>

**Problem:** Predict Iris species (Setosa, Versicolor, or Virginica) from sepal length/width and petal length/width.




```python
from IPython.display import Image, display

display(Image(filename="../images/iris_io.png"))
display(Image(filename="../images/iris.png"))
```


    
![png](images/iris_mlp_1_0.png)
    



    
![png](images/iris_mlp_1_1.png)
    


Let's read the dataset.


```python
import pandas as pd
df = pd.read_csv('../dataset/iris.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f'we have {len(df)} data points')
```

    we have 150 data points


#### Let's define our model


```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)
```

Prepare the dataset.


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Extract features and target
features = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
labels = df['Species'].values

# Encode labels to integers
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split dataset into train and test (let's use 80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Convert to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
```


```python
display(Image(filename="../images/iris_io.png"))

# each row of x_train contains 4 numbers which correspond to Sepal Length/Width and Petal Length/Width
# each row y_train contains a single number where 0=setosa, 1=versicolor and 2=virginica
next(zip(x_train, y_train))
```


    
![png](images/iris_mlp_9_0.png)
    





    (tensor([4.4000, 2.9000, 1.4000, 0.2000]), tensor(0))




```python
model = SimpleMLP()
logits = model(x_train[0])
f_loss = nn.CrossEntropyLoss()
f_loss(logits, y_train[0])
```




    tensor(0.9625, grad_fn=<NllLossBackward0>)



## Training loop


```python
model = SimpleMLP()
f_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()

losses = []
acc = []
import time

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # forward ------------>
    logits = model(x_train)        # shape: (N, 3)
    loss = f_loss(logits, y_train) # y shape: (N,)

    # <----------- backward
    loss.backward()
    optimizer.step()


    if epoch % 10 == 0:
        acc.append(accuracy(model, x_test, y_test))
        losses.append(loss.item())
        time.sleep(0.5)

```


```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", rc={"axes.facecolor": (0, 0, 0, 0)})

# Plot Loss
plt.figure(figsize=(8, 4), facecolor="none")  # transparent figure background
sns.lineplot(x=range(len(losses)), y=losses)
plt.xlabel('Epoch (per 10)')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.gcf().patch.set_alpha(0.0)   # transparent figure background
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 4), facecolor="none")  # transparent figure background
sns.lineplot(x=range(len(acc)), y=acc)
plt.xlabel('Epoch (per 10)')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Over Epochs')
plt.gcf().patch.set_alpha(0.0)   # transparent figure background
plt.show()
```


    
![png](images/iris_mlp_13_0.png)
    



    
![png](images/iris_mlp_13_1.png)
    


## Result: a model with over 98% accuracy at predicting Iris species

</details>

So far we have built a simple classification model to demonstrate the loss function, the forward and backward pass, and how optimization reduces loss.
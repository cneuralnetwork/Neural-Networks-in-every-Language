import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data.head()

data=np.array(data)
m,n=data.shape
np.random.shuffle(data)

data_dev=data[0:1000].T
Y_dev=data_dev[0]
X_dev=data_dev[1:n]
X_dev=X_dev/255

data_train=data[1000:m].T
Y_train=data_train[0]
X_train=data_train[1:n]
X_train=X_train/255
_,m_train=X_train.shape

def init_params():
    w1=np.random.rand(10,784)-0.5
    b1=np.random.rand(10,1)-0.5
    w2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return w1,b1,w2,b2

def ReLU(z):
    return np.maximum(0,z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def one_hot(z):
    one_hot_z=np.zeros((z.size,z.max()+1))
    one_hot_z[np.arange(z.size),z]=1
    one_hot_z=one_hot_z.T
    return one_hot_z

def derivative_ReLU(z):
    return z>0

def forward_prop(w1,b1,w2,b2,x):
    z1=w1.dot(x)+b1
    a1=ReLU(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(z2)
    return z1,a1,z2,a2

def back_prop(z1,a1,z2,a2,w2,x,y):
    m=y.size
    one_hot_Y=one_hot(y)
    dz2=a2-one_hot_Y
    dw2=1/m * dz2.dot(a1.T)
    db2=1/m * np.sum(dz2)
    dz1 = w2.T.dot(dz2)*derivative_ReLU(z1)
    dw1=1/m * dz1.dot(x.T)
    db1=1/m * np.sum(dz1)
    return dw1,db1,dw2,db2

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1=w1-alpha*dw1
    w2=w2-alpha*dw2
    b1=b1-alpha*db1
    b2=b2-alpha*db2
    return w1,b1,w2,b2

def get_predictions(a2):
    return np.argmax(a2,0)

def get_accuracy(predictions,y):
    print(predictions,y)
    return np.sum(predictions==y)/y.size

def gradient_descent(x,y,it,alpha):
    w1,b1,w2,b2=init_params()
    for i in range(it):
        z1,a1,z2,a2=forward_prop(w1,b1,w2,b2,x)
        dw1,db1,dw2,db2=back_prop(z1,a1,z2,a2,w2,x,y)
        w1,b1,w2,b2=update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
        if i%10==0:
            print("Iteration",i)
            print("Accuracy", get_accuracy(get_predictions(a2),y))
    return w1,b1,w2,b2

w1,b1,w2,b2=gradient_descent(X_train,Y_train,500,0.1)

def make_predictions(x,w1,b1,w2,b2):
    _,_,_,a2=forward_prop(w1,b1,w2,b2,x)
    predictions=get_predictions(a2)
    return predictions

def test_predictions(index,w1,b1,w2,b2):
    current_image=X_train[:,index,None]
    prediction=make_predictions(X_train[:,index,None],w1,b1,w2,b2)
    label=Y_train[index]
    print("Prediction : ",prediction)
    print("Label : ",label)
    current_image=current_image.reshape((28,28))*255
    plt.gray()
    plt.imshow(current_image,interpolation='nearest')
    plt.show()

dev_predictions=make_predictions(X_dev,w1,b1,w2,b2)
print("Accuracy - ",get_accuracy(dev_predictions,Y_dev))

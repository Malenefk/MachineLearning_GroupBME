#How to run our neural network script:
"""
Creating data for the regression problems:
"""
#Importing packages

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np 

#Random seed to keep the results consistent 
np.random.seed(2030)

def Runge():
    """
    Description:
    Creating the Runge function with 500 datapoints. 

    Returns:
    The runge function - x and y. 
    """
    n = 500
    x = np.linspace(-1,1,n).reshape(-1,1) #interval between -1 and 1 formatted to fit in shape with the neural network
    denominator = 1 + (25*x**2) 
    y = 1 / denominator #will not add noise right now
    return x, y  

x, y = Runge()

#Scale and split the data into test and training sets. 80% as training and 20% as test. 
Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42)
Scaling = StandardScaler(with_std = True)
Scaling.fit(Xtrain)
Xtrain = Scaling.transform(Xtrain)
Xtest = Scaling.transform(Xtest)
ymin, ymax = ytrain.min(), ytrain.max()
ytrain = (ytrain - ymin) / (ymax - ymin)
ytest = (ytest - ymin) / (ymax - ymin)

"""
Running the script and train example:
"""

#Import necessary functions from the neural network
from neuralnetwork import FFNN 
from neuralnetwork import Sigmoid
from neuralnetwork import MSE
from neuralnetwork import Adam

#Run the script
if __name__ == '__main__':
    
    b_1h_50n = FFNN(
        nodes = (1,50,1), #Dimensions (how many hidden layers (shape) + number of nodes)
        hidden_activation=Sigmoid(), #Hidden layer activation function
        output_activation = lambda x: x, #Outout layer activation function
        cost_func = MSE() #Chosen costfunction 
    )

    results = b_1h_50n.fit(
        X = Xtrain, 
        t = ytrain,
        scheduler = Adam(eta = 0.01, rho=0.9, rho2=0.99), #Scheduler - chosen gradient optimizer
        batches = 4, #batchsize 
        epochs = 300, #Number of epochs
        lam = 0.01 #regularization parameter
    )


#Code for the neural network 

#Importing necessary libraries 
import math
import autograd.numpy as np
import sys
import warnings
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy, copy
from typing import Tuple, Callable
from sklearn.utils import resample
from sklearn.utils import shuffle 

# TASK A 
class cost_functions():

    """
    Description:
    The class cost functions is a parent class for calculating the cost functions used in the Neural Network.
    The parent class itself contains four functions defining the L1 and the L2 norm. The L1 and L2 norms can be entered
    when creating a class object when choosing the cost function. 
    
    All subclasses will use cost_functions as a parent class, and contains the equations for calculating the cost functions as call methods.
    There are three different cost functions and all contains the same arugments which are defined below. 
    All cost functions have derived = False and norm = None as default. If derived = True, the functions returns a differentiated
    version of the cost function, which is done in the backpropogation of the neural network. If the norm is None, the neural 
    network only trains with a small regularization parameter lambda with default value: 0.01. If Norm is sat to L1 or L2,
    the functions will return a cost function with the norms added which applies a stronger regularization/penalty to the cost.  

    Arguments:
    arg[1] : y_pred (np.ndarray) - The predicted function y from the neural network training
    arg[2] : y_target (np.ndarray) - The target
    arg[3] : Weights - weights used in the neural network. These are updated in the backpropogation function
    arg[4] : lmd (float )- Regularization parameter. Default value is 0.1

    Returns:
    The cost (float)
    """

    def L1(self, weights, lmd):
        return lmd * np.abs(weights)
        
    def L1_der(self, weights, lmd): 
        return lmd*np.sign(weights)
    
    def L2(self, weights, lmd):
        return lmd * (weights)**2
        
    def L2_der(self, weights, lmd):
        return lmd*2*weights  
    
class MSE(cost_functions):
    
    """
    Description:
    Subclass for the Mean Squared Error (MSE) cost function.
    """

    def __init__(self):
        super().__init__()
        self.norm = None 
        self.derive = False 

    def __call__(self, y_pred, y_target, weights, lmd):
        if not self.derive:
            cost = np.mean((y_target - y_pred)**2) #MSE 

            if self.norm == 'L1':
                if isinstance(weights, list): #iterates through w in weights if cost is a list to support training with regularization without impacting the dimensions 
                    regularization = sum(np.sum(np.abs(w)) for w in weights)
                else:
                    regularization = np.sum(np.abs(weights))
                cost += lmd * regularization

            elif self.norm == 'L2':
                if isinstance(weights, list):
                    regularization = sum(np.sum(w**2) for w in weights)
                else:
                    regularization = np.sum(weights**2)
                cost += lmd * regularization
            return float(cost) 
        
        else:
            cost = -2 / len(y_pred[:,0]) * (y_target - y_pred)
            return cost 
        
class BCE_logloss(cost_functions):

    """
    Description:
    Subclass with parentclass cost_functions for the Binary Cross entropy cost function (otherwise known as Log loss) used for binary classification.
    """
    def __init__(self):
        super().__init__()
        self.norm = None 
        self.derive = False 

    def __call__(self, y_pred, y_target, weights, lmd):
        if not self.derive:
            cost = -(1.0 / y_target.shape[0]) * np.sum(y_target * np.log(y_pred + 10e-10) + (1 - y_target) * np.log(1-y_pred + 10e-10))
            if self.norm == 'L1':
                cost += self.L1(weights, lmd)
            elif self.norm == 'L2':
                cost += self.L2(weights, lmd) 
        else:
            cost = - 1/len(y_pred[:,0]) * ((y_target - y_pred) / (y_pred * (1-y_pred)))
        return cost
     
#Multiclass entropy
class MCE_multiclass(cost_functions):

    """
    The Multiclass Cross Entropy / loss function. 
    """

    def __init__(self):
        super().__init__()
        self.norm = None 
        self.derive = False

    def __call__(self, y_pred, y_target, weights, lmd):
        if not self.derive:
            cost = -(1.0 / y_target.size) * np.sum(y_target * np.log(y_pred + 10e-10))
            if self.norm == 'L1':
                cost += self.L1(weights, lmd)
            elif self.norm == 'L2':
                cost += self.L2(weights, lmd) 
        
        else:
            cost = - 1/len(y_pred[:,0])*(y_target/y_pred) 
        return cost 
    

class Activations():

    """
    Description: 
    This class contains activation functions used in the neural network. Activations is a parent class, and the subclasses contains the
    equations for the activations as call functions. All functions can be differentiated if self.derive = True which is done in the backpropogation
    function of the neural network, but the default is sat to False.

    Returns:
    All subclasses return the activation function.
    """

    def __init__(self):
        self.derive = False  

    def __call__(self,X):
        return X 

class Sigmoid(Activations):

    """
    Description:
    Calculates the Sigmoid function, also knwon as the logit function.
    The call function includes a Floatingpointerror, in cases of saturation to inform the user. 

    Arguments:
    arg[1] : X (np.ndarray) - The training data.
    """

    def __init__(self):
        super().__init__()
        
    def __call__(self, X): 
        if not self.derive:
            try:
                activation = 1.0 / (1+np.exp(-X)) 
            except FloatingPointError:
                activation = np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))
        
        else: 
            try:
                sig = 1.0 / (1 + np.exp(-X))
                activation = sig * (1-sig)
            except FloatingPointError:
                activation = np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))
        
        return activation

class RELU(Activations):
    """
    Description:
    Calculates the RELU function.

    Arguments:
    arg[1] : X (np.ndarray) - The training data.

    Returns:
    If the value < 0, it returns zero. If the value is > 0, it returns the input value. 
    """
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        if not self.derive:
            activation = np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))
        else:
            activation = np.where(X > 0, 1, 0)

        return activation
    
class LRELU(Activations):
    """
    Desciption:
    Calculates the Leaky RELU (LRELU). 

    Arguments:
    arg[1] : X (np.ndarray) - The training data.
    arg[2] : Delta (float) - Default value is 10e-10. Optional input. 
    
    Returns: 
    If the value < 0, the value is multiplied with delta, which is a small default value. 
    If the value > 0, it returns the input value. 
    """
    def __init__(self):
        super().__init__()
    
    def __call__(self, X, *, delta = 10e-10):
        if not self.derive:
            activation = np.where(X > np.zeros(X.shape), X, delta * X)
        else: 
            activation = np.where(X > 0, 1, delta)
        return activation


class Softmax(Activations):
    """
    Description:
    Calculates the Leaky RELU (LRELU). 

    Arguments:
    arg[1] : X (np.ndarray) - The training data.
    arg[2] : Target - Target function. Default is None, but to differentiate the equation it needs to be an input. 
    arg[2] : Delta (float) - Default value is 10e-10. Optional input.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, X, target=None, *, delta = 10e-10):
        X = X - np.max(X, axis=-1, keepdims=True)
        #Differentiated      
        if target is not None:
            soft = np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)
            activation = soft - target
        #Not differentiated
        else:
            activation = np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)
        return activation



#Learningrates from week42 code
class Scheduler:
    """
    Abstract class for Schedulers. Code imported from:

    Hjort-Jensen, Morten (2025). Computational Physics lecture notes 2025. 
    https://github.com/Malenefk/MachineLearning_teacher/blob/master/doc/LectureNotes/week42.ipynb
    
    Description: 
    The Scheduler class is a parentclass containing subclasses with gradient optimizer functions. 
    All subclasses call on Scheduler class and optimize the gradients as their updated during training.
    This code is imported from the lecture notes as cited above, and are therefore not changed. 

    Returns:
    All functions in the subclasses returns the updated gradient.
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass


class Constant(Scheduler):
    """
    Description:
    Plain gradient descent, where the gradient is multiplied with eta.

    Arguments:
    arg[1] : eta (float) - Learning rate
    arg[2] : Gradient 
    """
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient
    
    def reset(self):
        pass


class Momentum(Scheduler):
    """
    Description:
    Momentum optimizer, which includes a momentum that weights the previous steps.

    Arguments:
    arg[1] : eta (float) - Learning rate 
    arg[2] : Gradient 
    """
    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        pass


class Adagrad(Scheduler):
    """
    Description:
    AdaGrad optimizer which updates the learning rate. 

    Arguments:
    arg[1] : eta (float) - Learning rate 
    arg[2] : Gradient 
    """
    def __init__(self, eta):
        super().__init__(eta)
        self.G_t = None

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        return self.eta * gradient * G_t_inverse

    def reset(self):
        self.G_t = None


class AdagradMomentum(Scheduler):
    """
    Description:
    AdaGrad optimizer which updates the learning rate including momentum that weights the previous steps.

    Arguments:
    arg[1] : eta (float) - Learning rate 
    arg[2] : Gradient 
    """
    def __init__(self, eta, momentum):
        super().__init__(eta)
        self.G_t = None
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        self.G_t = None


class RMS_prop(Scheduler):
    """
    Description:
    RMSprop optimizer which updates the parameters by taking the running average of the square of the gradient.

    Arguments:
    arg[1] : eta (float) - Learning rate 
    arg[2] : Gradient 
    """

    def __init__(self, eta, rho):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self):
        self.second = 0.0


class Adam(Scheduler):
    """
    Description:
    ADAM optimizer which updates the parameters by taking the running average of the gradients (first momentum) and of the square of the gradient (second momentum). 

    Arguments:
    arg[1] : eta (float) - Learning rate 
    arg[2] : Gradient 
    arg[3] : Rho - Momentum parameter 1. Default value is 0.9 in the fit function. 
    arg[4] : Rho2 - Momentum parameter 2. Default value is 0.999 in the fit function. 
    """
    def __init__(self, eta, rho, rho2):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self):
        self.n_epochs += 1
        self.moment = 0
        self.second = 0


# TASK B 
class FFNN:
    """
    Description:
    A Feed Forward Neural Network class that can be used for both regression and classification problems.
    The following function specifies the activation and cost functions used for the code, but are callable
    to be changed if needed. 

    Arguments:
    arg[1]: Nodes (tuple[int]) - A tuple of positive integers, where the numbers of the array defines
        the number of nodes in each layer. The first number represents the number of nodes in the input layer, 
        the second number represents the number of nodes in the first hidden layer and so on, until the last layer.
    arg[2]: Hidden_activation (Callable) - The activation function for the hidden layers.
    arg[3]: Output_activation (Callable) - The activation function for the output layer.
    arg[4]: Cost_func (Callable) - The cost function
    arg[5]: Seed (int) - Set to 2030, which will produce the same random numbers each time we run the code
    
    """

    def __init__(
        self, 
        nodes: tuple[int],
        hidden_activation: Callable = Sigmoid(),
        output_activation: Callable = lambda x: x,
        cost_func: Callable = MSE(),
        seed: int = 2030,
    ):

        self.nodes = nodes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.cost_func = cost_func
        self.seed = seed
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matrices = list()
        self.classification = None
         
        self.reset_weights()
        self._set_classification()

    def fit(
        self, 
        X : np.ndarray,
        t : np.ndarray,
        scheduler : Scheduler = Adam(eta=0.01, rho = 0.9, rho2 = 0.99), 
        batches: int = 1,
        epochs: int = 100,
        lam: float = 0.01,

    ):
        """
        Description: 

        This function handles the technicalities to run the training of the feed forward and the backpropogating functions.
        This includes, but are not limited to splitting the data into batches, choosing the wanted learning rate optimizer,
        and storing the errors from training into initialized arrays. 

        Arguments: 

        arg[1] : X (np.ndarray) - The training data
        arg[2] : t (np.ndarray) - The target 
        arg[3] : Scheduler - Default is learning rate optimizer Adam
        arg[4] : Batches (integer) - Default is 1
        arg[5] : Epochs (integer) - Number of training iterations. Default is 100
        arg[6] : Lmd (float) - Regularization parameter. Default is 0.01

        Returns: 
        
        The function returns the resulting scores from the training. 
        """

        #setup
        np.random.seed(self.seed)

        #Initialize empty arrays
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = X.shape[0] // batches #Division returning only integers - the remaining is used as a final batch

        #Remove to do SGD
        #X, t = resample(X,t)

        # create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(copy(scheduler))
            self.schedulers_bias.append(copy(scheduler))

        print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, Lambda={lam}")

        try:
            for e in range(epochs):
                #Shuffling the matrix for SGD - remove if not using SGD 
                X, t = shuffle(X,t)

                for i in range(batches):
                    # Minibatch gradient descent
                    if i == batches - 1:
                        X_batch = X[i * batch_size :, :]
                        t_batch = t[i * batch_size :, :]
                    else:
                        X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                        t_batch = t[i * batch_size : (i + 1) * batch_size, :]

                    self._feedforward(X_batch)
                    self._backpropagate(X_batch, t_batch, lam)

                    # reset schedulers for each epoch
                for sch in self.schedulers_weight:
                    sch.reset()

                for sch in self.schedulers_bias:
                    sch.reset()

                # compute performance metrics
                pred_train = self.predict(X)
                self.cost_func.derive = False 
                train_error = self.cost_func(pred_train, t, self.weights, lam)
                train_errors[e] = train_error 

                if self.classification:
                    train_acc = self._accuracy(pred_train, t)
                    train_accs[e] = train_acc

                # Used for progress bar
                progression = e / epochs
                print_length = self._progress_bar(
                    progression,
                    train_error=train_errors[e],
                    train_acc=train_accs[e],
                )
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

                # visualization of training progression (similiar to tensorflow progression bar)
        sys.stdout.write("\r" + " " * print_length)
        sys.stdout.flush()
        self._progress_bar(
            1,
            train_error=train_errors[e],
            train_acc=train_accs[e],
        )
        sys.stdout.write("")

        # return performance metrics 
        scores = dict()
        scores["train_errors"] = train_errors

        if self.classification:
            scores["train_accs"] = train_accs

        return scores 
    
    def predict(self, X:np.ndarray, *, treshold = 0.5): 

        """
        Description: 

        After training the data, this function predicts which values to be "true". 

        Arguments:

        arg[1] : X (np.ndarray) - The design matrix X 
        arg[2] : Treshold (optional) - Default value of 0.5 which is the minimum treshold for the classification
            values to be predicted as "true".  
        
        Returns:

        A vector containing 0 or 1 for each row in the given design matrix. 0 is False and 1 is True. 
        """

        predict = self._feedforward(X)

        if self.classification:
            return np.argmax(predict, axis=1)
        else:
            return predict
        
    def reset_weights(self):


        """    
        Description:
        
        Resets the weights after training, in order to run new training with non-biased weights. 
        """
            
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.nodes) - 1):
            weight_array = np.random.randn(self.nodes[i] + 1, self.nodes[i + 1])
            weight_array[0, :] = np.random.randn(self.nodes[i + 1]) * 0.01

            self.weights.append(weight_array)

        
    def _feedforward(self, X: np.ndarray):
        """
        Description:

        This function runs the feed forward algorithm.
        All of the activations are calculated from a weighted sum stored in arrays from the previous activations.

        Arguments:

        arg[1] : X (np.ndarray) - The design matrix X. 

        Returns:
        
        The output layer. 
        """

        # reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        # If X = vector - turn into matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        #Adding a column of zeros to the design matrix, to append the biases. 
        bias = np.ones((X.shape[0], 1)) * 0.1
        X = np.hstack([bias, X])

        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        #Feed Forward
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                z = a @ self.weights[i]
                self.z_matrices.append(z)
                a = self.hidden_activation(z)
                #Adding new weights 
                bias = np.ones((a.shape[0], 1)) * 0.01
                a = np.hstack([bias, a])
                self.a_matrices.append(a)
            else:
                try:
                    # For the final output layer
                    z = a @ self.weights[i]
                    a = self.output_activation(z)
                    self.a_matrices.append(a)
                    self.z_matrices.append(z)
                except OverflowError:
                    print(
                        "OverflowError in fit() in FFNN\nHOW TO DEBUG ERROR: Consider lowering your learning rate or scheduler specific parameters such as momentum, or check if your input values need scaling"
                    )
        return a


    def  _backpropagate(self, X, t, lam):

        """
        Description:
        The function contains the algortihm for backpropogation, which updates the weights and biases as it moves from 
        the last output layer and backwards till the input layer is reached. It computes the gradients for each layer, which
        is used to actively update weights and biases with the use of an optimizer. The default optimizer is Adam, but this
        can be changed when calling the fit function.
        The backpropogation includes gradient clipping to prevent overflow errors. 

        Arguments:
        arg[1] : X - the design matrix X
        arg[2] : t - the target 
        arg[3] : lam (float) - regularization parameter with default value of 0.01

        Returns:
        The function has no return value, but updates the weights and biases. 
        """

        #Prepping the output function to be differentiated 
        output_func = self.output_activation
        output_func.derive = True

        #Prepping the hidden layer to be differentiated
        hidden_func = self.hidden_activation
        hidden_func.derive = True 

        #Starting from the last layer continuing backwards
        for i in range(len(self.weights) -1, -1, -1):
            if i == len(self.weights) -1: #last layer
                #If single class classification
                if (self.output_activation.__class__.__name__ == 'Softmax'):
                    delta_matrix = self.a_matrices[i+1] - t 
                else:
                    cost_func = self.cost_func
                    cost_func.derive = True 
                    c_d = cost_func(self.a_matrices[-1],t, self.weights[i], lam)
                    if self.classification:
                        delta_matrix = output_func(self.z_matrices[i + 1]) * c_d
                    else:
                        delta_matrix = c_d 

            else:
                delta_matrix = (self.weights[i + 1][1:, :] @ delta_matrix.T).T * hidden_func(self.z_matrices[i + 1])

            # calculate gradient
            gradient_weights = self.a_matrices[i][:, 1:].T @ delta_matrix 
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])
            if self.cost_func.norm == 'L1':
                gradient_weights += lam * np.sign(self.weights[i][1:,:])
            elif self.cost_func.norm == 'L2':
                gradient_weights += 2*lam * self.weights[i][1:,:]
            else:
                gradient_weights += self.weights[i][1:, :] * lam

            #Because some of the results are blowing up (overflow error) we introduce clipping of the norm:
            #https://www.geeksforgeeks.org/deep-learning/understanding-gradient-clipping/
            norm = np.linalg.norm(gradient_weights)
            clip_factor = 1 / norm
            if norm > 1.0:
                gradient_weights = gradient_weights * clip_factor
    
            update_matrix = np.vstack(
                [self.schedulers_bias[i].update_change(gradient_bias),
                 self.schedulers_weight[i].update_change(gradient_weights)])

            # update weights and bias
            self.weights[i] -= update_matrix

    def _accuracy(self, predictions: np.ndarray, targets: np.ndarray):
            """
            Description:
            Calulates the average value of a prediction to target, to evaluate the accuracy. 

            Arguments:
            arg[1] : Predictions (np.ndarray) - The predicted values from the training. The input is an vector, filled with either:
            - 1 and 0's if a classification problem. 
            - Numbers (float) ia regression problem. 
            arg[2] : Targets (np.ndarray) - A vector containing the target values. 

            Returns:
            A float that evluates the avergage of targets and predictions. This is a percentage of correctly classified instances. 
            """
            if self.classification:
                pred_class = np.argmax(predictions, axis=1)
                if len(targets.shape)>1 and len(targets[:,0])>1:
                    target_class = np.argmax(targets, axis=1)
                else:
                    target_class = targets.flatten()
                return np.mean(pred_class == target_class)
            else:
                assert predictions.shape == targets.shape
                return np.average((targets == predictions))
        
    def _set_classification(self):

            """
            Description:
            Decides wether FFNN is a regression or a classification problem based on the cost function being called.
            The cost functions are called on through call methods and therefore recognized by __class__.__name__. 

            False implies that it is a regression problem. 
            True implies that it is a classification problem. 
            """
            
            self.classification = False 
            if (self.cost_func.__class__.__name__ == 'BCE_logloss' or self.cost_func.__class__.__name__ == 'MCE_multiclass'):
                self.classification = True

    def _progress_bar(self, progression, **kwargs):
            """
            Description: 
            Making a progress bar displaying the progress of the traning. 

            Arguments:
            arg[1] : Progression (list) from Fit function

            Returns:
            The progression bar. 

            """
            print_length = 40 
            num_equals = int(progression * print_length)
            num_not = print_length - num_equals
            arrow = ">" if num_equals > 0 else ""
            bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
            perc_print = self._format(progression * 100, decimals=5)
            line = f"  {bar} {perc_print}% "

            for key in kwargs:
                if not np.isnan(kwargs[key]):
                    value = self._format(kwargs[key], decimals=4)
                    line += f"| {key}: {value} "
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
            return len(line)
        
    def _format(self, value, decimals=4):
            """
            Description: 
            Formats decimal numbers displayed in the progress bar created above 

            Arguments: 
            arg[1] : Value - created in the _progress_bar function
            """

            if value > 0:
                v = value 
            elif value < 0:
                v = -10 * value 
            else:
                v = 1
            n = 1 + math.floor(math.log10(v))
            if n >= decimals -1:
                return str(round(value))
            return f'{value:.{decimals-n-1}f}'        

#All functions used in assignments 1a - 1d

#Importing necessary packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import (
    PolynomialFeatures,
)  # use the fit_transform method of the created object!
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import seaborn as sns 
from numpy.random import rand
from numpy.random import seed

#Setting a random seed to reproduce the same data
np.random.seed(3155)

#Create dataset function
def dataset_runge():
    n = 500
    x = np.linspace(-1,1, n) #x within interval [-1,1]
    denominator = 1+(25*x**2)
    y = 1/denominator + np.random.normal(0, 1, x.shape) #with noice
    return x, y, n

#This function creates a designmatrix X either with or without the intercept.
#Arguments: vector x with n values. P is the polynomial degree.
def polynomial_features(x, p, intercept=bool): 
    n = len(x)
    X = np.zeros((n, p + 1)) 
    if intercept == True: #Keeps a first column filled with 1´s 
        for i in range(p+1):
            X[:, i] = x**i
    elif intercept == False: #""Jumps"" to the first column of values
        for i in range(1,p+1):
            X[:,i] = x**i
    else: #Error message if one forgets to declare wether the intercept should be included or not
        raise TypeError(f"Please include a boolean response to the function parameter the intercept") 
    return X #returns design matrix X 

#Creating the OLS parameter Theta
#Arguments: X from polynomial feature (or created by Scikit learn). y is our function. 
def OLS_parameters(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y  #Optimization of OLS theta. 
    return theta

#Creating the Ridge Parameter Theta
#Arguments: X from polynomial feature (or created by Scikit learn). y is our function. Lambda is a set regularization parameter.
def Ridge_parameters(X, y, lambda_own):
    p = (len(X[0,:]))
    #Assumes X is scaled and has no intercept column
    #np.eye calculates a matrix with values on the diagonal and else zeros. 
    return np.linalg.pinv(X.T @ X + lambda_own * np.eye(p,p)) @ X.T @ y #Returns the optimization of Ridge theta


#Full function calculating the MSE and R2 scores of the OLS regression. 
#Arguments: Does not include the intercept as default. Degree is the wished polynomial degree. 
def mse_poly_plot_OLS(degree, intercept=False): 
    poly_deg = np.arange(1, degree+1)
    results = {}

    #Running the function for varying values of n in our dataset
    for n in [300, 400, 500]:
        #Defining the dataset
        x = np.linspace(-1, 1, n)
        denominator = 1+(25*x**2)
        y = 1.0 / denominator + np.random.normal(0, 1, x.shape)

        #Initialize empty arrays for later use
        mse_train_list = np.zeros(degree)
        mse_test_list = np.zeros(degree)
        R2_test = np.zeros(degree)
        R2_train = np.zeros(degree)
        beta_matrix = np.zeros((degree, degree+1))
        beta_array = np.arange(degree+1)

        #Range for the chosen polynomial degree
        for p in range(1, degree+1):
            #Calculate design matrix X
            X = polynomial_features(x, p, intercept=intercept)
            #Split data into subsets of traning and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
            #Centre data (Does not include standard deviation)
            scaler = StandardScaler(with_std=False)
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            y_mean = np.mean(y_train)
            y_scaled_train = (y_train - y_mean)

            #Calculate the parameters through the optimization of OLS Theta (called beta in our function but it can be called on as whatever variable one wishes)
            beta = OLS_parameters(X_train_s, y_scaled_train)

            #Make the predictions: y tilde
            y_pred_train = X_train_s @ beta + y_mean
            y_pred_test = X_test_s @ beta + y_mean

            #Calculate MSE and R2 scores using the Scikit learn functions. 
            #Links to the functions used: 
            #MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
            #R2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
            mse_train_list[p-1] = MSE(y_train, y_pred_train)
            mse_test_list[p-1] = MSE(y_test, y_pred_test)
            R2_test[p-1] = r2_score(y_test, y_pred_test)
            R2_train[p-1] = r2_score(y_train, y_pred_train)

            # Storing betas in a matrix and plotting the betas
            for i in range(len(beta)):
                beta_matrix[p-1, i] = beta[i]
            plt.scatter(beta_array, beta_matrix[p-1, :], label=f'p={p}')

        #Add information to the scatter plot of the betas 
        plt.xlabel(r'$\theta$ index')
        plt.ylabel(r'Value of $\theta$')
        plt.title(fr'$\theta$ for n={n}')
        plt.legend(bbox_to_anchor=(1.2,0.5), loc='center right')
        plt.show()

        #Storing the results from the MSE and R2 functions for plotting later. 
        #Using the functionality of a dictionary. 
        results[n] = {
            "mse_train": mse_train_list,
            "mse_test": mse_test_list,
            "R2_train": R2_train,
            "R2_test": R2_test
        }

    #Creating subplot 1 for MSE values. 
    fig1, ax1 = plt.subplots(1,2, figsize=(12,5))

    #Plotting for all n values chosen earlier in code - but in the same plot. 
    for n, vals in results.items():
        ax1[0].plot(poly_deg, vals["mse_test"], label=f"MSE test (n={n})")
        ax1[1].plot(poly_deg, vals["mse_train"], label=f"MSE train (n={n})")

    #Adding information to the subplot 1
    ax1[0].set_title("MSE OLS Regression Test ")
    ax1[1].set_title("MSE OLS Regression Train")
    ax1[0].set_xlabel("Polynomial degree")
    ax1[1].set_xlabel("Polynomial degree")
    ax1[0].set_ylabel("MSE")
    ax1[0].set_ylabel("MSE")
    ax1[0].grid(True)
    ax1[1].grid(True)
    ax1[1].legend()
    ax1[0].legend()

    #Creating subplot 2 for R2 scores. 
    #Plotting for all n values chosen earlier in code - but in the same plot. 
    fig2, ax2 = plt.subplots(1,2, figsize=(12,5))
    for n, vals in results.items():
        ax2[0].plot(poly_deg, vals["R2_test"], label=fr'$R^2$ test (n={n})')
        ax2[1].plot(poly_deg, vals["R2_train"], label=fr'$R^2$ train (n={n})')
    
    #Adding information to the subplot 2 
    ax2[0].set_title(r"$R^2$ OLS Regression Test ")
    ax2[0].set_xlabel("Polynomial degree")
    ax2[0].set_ylabel(r"$R^2$")
    ax2[1].set_ylabel(r"$R^2$")
    ax2[1].set_xlabel("Polynomial degree")
    ax2[0].grid(True)
    ax2[1].grid(True)
    ax2[0].legend()
    ax2[1].legend()
    ax2[1].set_title(r"$R^2$ OLS Regression Train")      
    ax2[0].legend() 
    plt.show()

    return beta


#Full function calculating the MSE and R2 scores for the Ridge regression. 
#Arguments: Does not include the intercept as default. Degree is the wished polynomial degree and here sat as a default value that can be changed. 
def heatmap_ridge(intercept=False, degree=16):
    #Defining varying n´s and lambdas to iterate through
    n_list=(300, 400, 500)
    nlambdas = 16
    lambdas = np.logspace(-5, 1, nlambdas)
    
    #Iteration for the chosen n values
    for n in [300,400,500]:
        #Creating the dataset
        x = np.linspace(-1, 1, n)
        denominator = 1+(25*x**2)
        y = 1.0 / denominator + np.random.normal(0, 1, x.shape)

        #Initialize empty matrices for storing the results for the heatmaps
        mse_train = np.zeros((degree, nlambdas))
        mse_test  = np.zeros((degree, nlambdas))
        r2_train  = np.zeros((degree, nlambdas))
        r2_test   = np.zeros((degree, nlambdas))

        #Iterations through the chosen polynomial degree (p)
        for p in range(1, degree + 1):
            #Creating the design matrix X 
            X = polynomial_features(x, p, intercept=intercept)
            #Splitting the data into subsets of training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3155)
            #Scaling the data with standard deviation for the Ridge Regression
            scaler = StandardScaler(with_std=True)
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)
            y_mean = y_train.mean()
            y_center = y_train - y_mean / np.std(y)

            #Ridge Regression optimizer for the varying lambdas defined above in the code 
            for j, lmb in enumerate(lambdas):
                beta = Ridge_parameters(X_train_s, y_center, lmb)

                #Create the predictions of y tilde
                y_pred_tr = X_train_s @ beta + y_mean
                y_pred_te = X_test_s  @ beta + y_mean

                #Calculate MSE values and R2 scores using the Scikit learn functions: 
                #Links to the functions used: 
                #MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
                #R2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
                mse_train[p-1, j] = MSE(y_train, y_pred_tr)
                mse_test [p-1, j] = MSE(y_test,  y_pred_te)
                r2_train [p-1, j] = r2_score(y_train, y_pred_tr)
                r2_test  [p-1, j] = r2_score(y_test,  y_pred_te)

        #Creating default information for the heatmaps
        xticks = [f'{lmb:.1e}' for lmb in lambdas]
        yticks = np.arange(1, degree + 1)
        cmap = 'PiYG' #Colormap to make it pink :) 

        #Subplot 1 for the MSE training data
        fig_mse, axes_mse = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        #Plotting the training data and providing plot information
        sns.heatmap(mse_train, ax=axes_mse[0], xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar=True)
        axes_mse[0].set_title(f'MSE Ridge Regression Train n={n}')
        axes_mse[0].set_xlabel(r'$\lambda$')
        axes_mse[0].set_ylabel('Polynomial degree')
        axes_mse[0].invert_yaxis()
        axes_mse[0].invert_xaxis()
        axes_mse[0].tick_params(axis='x', rotation=45)

        #Plotting the test data and providing plot information
        sns.heatmap(mse_test, ax=axes_mse[1], xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar=True)
        axes_mse[1].set_title(f'MSE Ridge Regression Test n={n}')
        axes_mse[1].set_xlabel(r'$\lambda$')
        axes_mse[1].set_ylabel('Polynomial degree')
        axes_mse[1].invert_yaxis()
        axes_mse[1].invert_xaxis()
        axes_mse[1].tick_params(axis='x', rotation=45)

        #Create subplot 2 
        fig_r2, axes_r2 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        #Plotting the training data and providing plot information
        sns.heatmap(r2_train, ax=axes_r2[0], xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar=True, vmin=0.0, vmax=1.0)
        axes_r2[0].set_title(f'$R^2$ Ridge Regression Train n={n}')
        axes_r2[0].set_xlabel(r'$\lambda$')
        axes_r2[0].set_ylabel('Polynomial degree')
        axes_r2[0].invert_yaxis()
        axes_r2[0].invert_xaxis()
        axes_r2[0].tick_params(axis='x', rotation=45)
        #Plotting the test data and providing plot information
        sns.heatmap(r2_test, ax=axes_r2[1], xticklabels=xticks, yticklabels=yticks, cmap=cmap, cbar=True, vmin=0.0, vmax=1.0)
        axes_r2[1].set_title(f'$R^2$ Ridge Regression Test n={n}')
        axes_r2[1].set_xlabel(r'$\lambda$') 
        axes_r2[1].set_ylabel('Polynomial degree')
        axes_r2[1].invert_yaxis()
        axes_r2[1].invert_xaxis()
        axes_r2[1].tick_params(axis='x', rotation=45)
        plt.show()

        #This plots return two subplots for three values of n


#Calculating the Gradient Descent for Ridge and OLS 
#Function for calculating the Gradient Descent for both OLS and Ridge
#Arguments: x and y are the dataset created outside of the function. n_feat is the number of iterations, while the degree is sat to a fixed value of 15 polynomials
#create dataset 
n = 500
x = np.linspace(-1,1, n) #x within interval [-1,1]
denominator = 1+(25*x**2)
y = 1/denominator + np.random.normal(0, 1, x.shape)
iter = 1

def gradient(x,y,n_feat, degree=15):
    np.random.seed(3155)
    #define variables 
    num_iters = 500
    lam = 0.1 #fixed lambda for the Ridge
    mse_test_plot = []
    r2_test_plot = []
    mse_test_plot_ridge = []
    r2_test_plot_ridge = []
    for eta_vary in [0.01, 0.05, 0.1]:
        n = len(y)
        r2_gdOLSn_test = np.zeros(degree)
        r2_gdRidgen_test = np.zeros(degree)
        degree_plot = np.linspace(1,degree, degree)
        mse_gdOLSn_test = np.zeros(degree)
        mse_gdRidgen_test = np.zeros(degree)
        theta_gdOLSn = np.zeros(n_feat) #OLS
        theta_gdRidgen = np.zeros(n_feat) #Ridge
        
        #loop
        for p in range(1,degree+1):
        #Create X_polynomial and scale 
            X = polynomial_features(x,15, intercept = False)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            scaler = StandardScaler(with_std=True)
            scaler.fit(X_train) 
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            y_mean = np.mean(y_train)
            y_std = np.std(y_train)
            y_scaled_train = (y_train - y_mean)/y_std

            #gradient descent OLS 
            for t in range(num_iters):
                grad_OLSn = (2.0/n)*X_train_s.T @ (X_train_s @ theta_gdOLSn-y_scaled_train)
                theta_gdOLSn -= grad_OLSn * eta_vary

            #gradient descent Ridge
            tol = 1e-10
            for t in range(num_iters):
            # Compute gradients for Ridge
                grad_Ridgen = (2.0/n)*X_train_s.T @ (X_train_s @ (theta_gdRidgen)-y_scaled_train)+2*lam*theta_gdRidgen
                # Update parameters theta
                theta_gdRidgen -= grad_Ridgen * eta_vary

                if (np.linalg.norm(grad_Ridgen*eta_vary) < tol):
                    print(f'loop broken at {str(t)} for degree: {p}')
                    break
        
            #predicting vals 
            y_pred_test_OLS = (X_test_s @ theta_gdOLSn + y_mean)
            y_pred_test_Ridge = (X_test_s @ theta_gdRidgen + y_mean)
            #predicting r2 scores
            r2_gdOLSn_test[p-1] = r2_score(y_test, y_pred_test_OLS)
            r2_gdRidgen_test[p-1] = r2_score(y_test, y_pred_test_Ridge)
            #predicting mse 
            mse_gdOLSn_test[p-1] =  MSE(y_test, y_pred_test_OLS)
            mse_gdRidgen_test[p-1] = MSE(y_test, y_pred_test_Ridge)

        mse_test_plot.append(mse_gdOLSn_test)
        r2_test_plot.append(r2_gdOLSn_test)
        mse_test_plot_ridge.append(mse_gdRidgen_test)
        r2_test_plot_ridge.append(r2_gdRidgen_test)
    
    liste = [0.01, 0.05, 0.1]
        
    fig1, ax1 = plt.subplots(1,2,figsize=(14,6))
    for i, eta_vary in enumerate(liste):
        ax1[0].plot(degree_plot,mse_test_plot[i], label = f"Eta = {eta_vary}")
        ax1[1].plot(degree_plot,r2_test_plot[i], label = f'Eta = {eta_vary}')
    ax1[0].set_title("MSE for OLS with GD")                                    
    ax1[1].set_title("R2 for OLS with GD")
    ax1[0].set_xlabel('Polynomial degree')
    ax1[0].set_ylabel('MSE')
    ax1[0].grid(True)
    ax1[0].legend()
    ax1[1].set_xlabel('Polynomial degree')
    ax1[1].set_ylabel('R2 score')
    ax1[1].grid(True)
    ax1[1].legend()
    fig1 = plt.suptitle('Gradient Descent for OLS - Test data. N = 500')
    fig1 = plt.tight_layout()

    fig2,ax2 = plt.subplots(1,2,figsize=(14,6))
    for j, eta_vary in enumerate(liste):
        ax2[0].plot(degree_plot, mse_test_plot_ridge[j], label = f'Eta = {eta_vary}')
        ax2[1].plot(degree_plot, r2_test_plot_ridge[j], label = f'Eta = {eta_vary}')
    ax2[0].set_title("MSE for Ridge with GD")                                    
    ax2[1].set_title("R2 for Ridge with GD")
    ax2[0].set_xlabel('Polynomial degree')
    ax2[0].set_ylabel('MSE')
    ax2[0].grid(True)
    ax2[0].legend()
    ax2[1].set_xlabel('Polynomial degree')
    ax2[1].set_ylabel('R2 score')
    ax2[1].grid(True)
    ax2[1].legend()
    fig2 = plt.suptitle(f'Gradient Descent for Ridge - Test data. N = 500')
    fig2 = plt.tight_layout()
    #Plot the two figures 
    plt.show()


#Introducing various effective optimizers to the gradient descent function
#Gradient Descent with Momentum for both OLS and Ridge
#Arguments: x and y are the dataset created outside of the function. n_feat is the number of iterations, while the degree is sat to a fixed value of 16 polynomials
def gradient_momentum(x, y, n_feat, degree=16):
    #define variables 
    num_iters = 500
    eta = 0.01
    lam = 0.1 #fixed lambda for the Ridge
    mom = 0.3 #Momentum parameter

    #Initialize arrays of zeros for later use
    v_ols = np.zeros(n_feat)
    v_ridge = np.zeros(n_feat)
    theta_gdOLSn = np.zeros(n_feat) #OLS
    theta_gdRidgen = np.zeros(n_feat) #Ridge
    r2_gdOLSn_test = np.zeros(degree)
    r2_gdRidgen_test = np.zeros(degree)
    r2_gdOLSn_train = np.zeros(degree)
    r2_gdRidgen_train = np.zeros(degree)
    degree_plot = np.linspace(1,degree, degree)
    mse_gdOLSn_test = np.zeros(degree)
    mse_gdOLSn_train = np.zeros(degree)
    mse_gdRidgen_test = np.zeros(degree)
    mse_gdRidgen_train = np.zeros(degree)

    #Initialize lists to be used in the comparison subplot in the function all_learning_rates_plot
    mse_momentum_test_ols = []
    mse_momentum_test_ridge = []
    r2_momentum_test_ols = []
    r2_momentum_test_ridge = []

    #looping through the selected number of polynomial degrees (p)
    for p in range(1,degree+1):
    #Create design matrix X and split the data into training and test sets of 20% test size and 80% train size
        X = polynomial_features(x,15, intercept = False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #Scaling the data - with standard deviation 
        scaler = StandardScaler(with_std=True)
        scaler.fit(X_train) 
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled_train = (y_train - y_mean)/y_std

        #Calculating the gradient descent for the OLS Regression
        for t in range(num_iters):
            #compute the gradients for OLS
            grad_OLSn = (2.0/n)*X_train_s.T @ (X_train_s @ theta_gdOLSn-y_scaled_train)
            #calculating the change using the momentum method
            v_ols = mom * v_ols + eta *grad_OLSn
            #updating theta with v (change in momentum)
            theta_gdOLSn = theta_gdOLSn - v_ols

        #Calculating the gradient descent for the Ridge Regression
        tol = 1e-10
        for t in range(num_iters):
        # Compute the gradients for Ridge
            grad_Ridgen = (2.0/n)*X_train_s.T @ (X_train_s @ (theta_gdRidgen)-y_scaled_train)+2*lam*theta_gdRidgen
            # calculating the change using the momentum method
            v_ridge = mom * v_ridge + eta * grad_Ridgen
            #updating theta with v (the change in momentum)
            theta_gdRidgen = theta_gdRidgen - v_ridge
            
            #If test to make sure the update values of theta are higher than a given tolerance value - to avoid unneseccary looping through smaller values
            #If the gradient * the learning rate is smaller than the tolerance value of 1e-10 the loop will break.  
            if (np.linalg.norm(grad_Ridgen*eta) < tol):
                print(f'loop broken at {str(t)} for degree: {p}') #Lets us know when running the code at which iteration the loop was broken for each polynomial degree
                break
        
        #predicting y tilde for both OLS and Ridge - for both test and train data. Though we will only plot the test data, but the train data is then avaible for use if wished. 
        y_pred_train_OLS = (X_train_s @ theta_gdOLSn + y_mean)
        y_pred_test_OLS = (X_test_s @ theta_gdOLSn + y_mean)
        y_pred_train_Ridge = (X_train_s @theta_gdRidgen + y_mean)
        y_pred_test_Ridge = (X_test_s @ theta_gdRidgen + y_mean)

        #Calculate MSE values and R2 scores using the Scikit learn functions: 
        #Links to the functions used: 
        #MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
        #R2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        r2_gdOLSn_test[p-1] = r2_score(y_test, y_pred_test_OLS)
        r2_gdRidgen_test[p-1] = r2_score(y_test, y_pred_test_Ridge)
        r2_gdOLSn_train[p-1] = r2_score(y_train, y_pred_train_OLS)
        r2_gdRidgen_train[p-1] = r2_score(y_train, y_pred_train_Ridge)
        mse_gdOLSn_test[p-1] =  MSE(y_test, y_pred_test_OLS)
        mse_gdOLSn_train[p-1] =  MSE(y_train, y_pred_train_OLS)
        mse_gdRidgen_test[p-1] = MSE(y_test, y_pred_test_Ridge)
        mse_gdRidgen_train[p-1] =  MSE(y_train, y_pred_train_Ridge)

        #Appending the results to the initizialed empty lists to later plot the results for the optimizers
        mse_momentum_test_ols.append(mse_gdOLSn_test)
        mse_momentum_test_ridge.append(mse_gdRidgen_test)
        r2_momentum_test_ols.append(r2_gdOLSn_test)
        r2_momentum_test_ridge.append(r2_gdRidgen_test)

    return mse_momentum_test_ols, mse_momentum_test_ridge, r2_momentum_test_ols, r2_momentum_test_ridge

#The same process as above is repeated for the other optimization problems. There will therefore be less comments in the following codes, expect from the new variables and calculations used
#Gradient Descent with AdaGrad for both OLS and Ridge

def gradient_adagrad(x, y, n_feat, degree=16):
    np.random.seed(3155)
    #define variables 
    num_iters = 500
    eta = 0.01
    lam = 0.1 #fixed lambda for the Ridge
    epsilon = 1e-8 #Small value to avoid division by zero
    theta_gdOLSn = np.zeros(n_feat) #OLS
    theta_gdRidgen = np.zeros(n_feat) #Ridge
    r2_gdOLSn_test = np.zeros(degree)
    r2_gdRidgen_test = np.zeros(degree)
    r2_gdOLSn_train = np.zeros(degree)
    r2_gdRidgen_train = np.zeros(degree)
    degree_plot = np.linspace(1,degree, degree)
    mse_gdOLSn_test = np.zeros(degree)
    mse_gdOLSn_train = np.zeros(degree)
    mse_gdRidgen_test = np.zeros(degree)
    mse_gdRidgen_train = np.zeros(degree)
    Giter_ols = 0.0
    Giter_ridge = 0.0

    #Initialize arrays to be used in the comparison subplot in a later function: 
    mse_adagrad_test_ols = []
    mse_adagrad_test_ridge = []
    r2_adagrad_test_ols = []
    r2_adagrad_test_ridge = []

    
    #loop
    for p in range(1,degree+1):
    #Create X and scale the data
        X = polynomial_features(x,15, intercept = False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler(with_std=True)
        scaler.fit(X_train) 
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled_train = (y_train - y_mean)/y_std

        #Gradient of OLS 
        for t in range(num_iters):
            grad_OLSn = (2.0/n)*X_train_s.T @ (X_train_s @ theta_gdOLSn-y_scaled_train)
            Giter_ols += grad_OLSn * grad_OLSn #creates the sum of the squares of the gradients
            update_ols = grad_OLSn * eta / (epsilon + np.sqrt(Giter_ols)) #update value of the gradient with the Adagrad
            theta_gdOLSn -= update_ols #update theta
    

        #Gradient of Ridge
        tol = 1e-10
        for t in range(num_iters):
        # Compute gradients for Ridge
            grad_Ridgen = (2.0/n)*X_train_s.T @ (X_train_s @ (theta_gdRidgen)-y_scaled_train)+2*lam*theta_gdRidgen
            Giter_ridge += grad_Ridgen * grad_Ridgen #creates the sum of the squares of the gradients
            update_ridge = grad_Ridgen * eta / (epsilon + np.sqrt(Giter_ridge)) #update value of the gradient with the Adagrad
            theta_gdRidgen -= update_ridge #update theta
            
    
            if (np.linalg.norm(grad_Ridgen*eta) < tol): 
                print(f'loop broken at {str(t)} for degree: {p}')
                break
        
        #predicting y tilde for both OLS and Ridge - for both test and train data. Though we will only plot the test data, but the train data is then avaible for use if wished.
        y_pred_train_OLS = (X_train_s @ theta_gdOLSn + y_mean)
        y_pred_test_OLS = (X_test_s @ theta_gdOLSn + y_mean)
        y_pred_train_Ridge = (X_train_s @theta_gdRidgen + y_mean)
        y_pred_test_Ridge = (X_test_s @ theta_gdRidgen + y_mean)

        #Calculate MSE values and R2 scores using the Scikit learn functions: 
        #Links to the functions used: 
        #MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
        #R2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        r2_gdOLSn_test[p-1] = r2_score(y_test, y_pred_test_OLS)
        r2_gdRidgen_test[p-1] = r2_score(y_test, y_pred_test_Ridge)
        r2_gdOLSn_train[p-1] = r2_score(y_train, y_pred_train_OLS)
        r2_gdRidgen_train[p-1] = r2_score(y_train, y_pred_train_Ridge)
        mse_gdOLSn_test[p-1] =  MSE(y_test, y_pred_test_OLS)
        mse_gdOLSn_train[p-1] =  MSE(y_train, y_pred_train_OLS)
        mse_gdRidgen_test[p-1] = MSE(y_test, y_pred_test_Ridge)
        mse_gdRidgen_train[p-1] =  MSE(y_train, y_pred_train_Ridge)

        #Appending the results to the initizialed empty lists to later plot the results for the optimizers
        mse_adagrad_test_ols.append(mse_gdOLSn_test)
        mse_adagrad_test_ridge.append(mse_gdRidgen_test)
        r2_adagrad_test_ols.append(r2_gdOLSn_test)
        r2_adagrad_test_ridge.append(r2_gdRidgen_test)

    return mse_adagrad_test_ols, mse_adagrad_test_ridge, r2_adagrad_test_ols, r2_adagrad_test_ridge

#Repeat for RMSProp
def gradient_rmsprop(x, y, n_feat, degree=16):
    #define variables 
    num_iters = 500
    eta = 0.01
    lam = 0.1 #fixed lambda for the Ridge
    epsilon = 1e-8 #Small value to avoid division by zero
    Beta_decay = 0.9 #Decay rate - defined as 0.9
    theta_gdOLSn = np.zeros(n_feat) #OLS
    theta_gdRidgen = np.zeros(n_feat) #Ridge
    r2_gdOLSn_test = np.zeros(degree)
    r2_gdRidgen_test = np.zeros(degree)
    r2_gdOLSn_train = np.zeros(degree)
    r2_gdRidgen_train = np.zeros(degree)
    degree_plot = np.linspace(1,degree, degree)
    mse_gdOLSn_test = np.zeros(degree)
    mse_gdOLSn_train = np.zeros(degree)
    mse_gdRidgen_test = np.zeros(degree)
    mse_gdRidgen_train = np.zeros(degree)
    Giter_ols = 0.0
    Giter_ridge = 0.0

    #Initialize arrays to be used in the comparison subplot in a later function: 
    mse_rmsprop_test_ols = []
    mse_rmsprop_test_ridge = []
    r2_rmsprop_test_ols = []
    r2_rmsprop_test_ridge = []

    
    #loop
    for p in range(1,degree+1):
    #Create X and scale the data
        X = polynomial_features(x,15, intercept = False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler(with_std=True)
        scaler.fit(X_train) 
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled_train = (y_train - y_mean)/y_std

        #gradient descent OLS 
        for t in range(num_iters):
            grad_OLSn = (2.0/n)*X_train_s.T @ (X_train_s @ theta_gdOLSn-y_scaled_train)
            Giter_ols = (Beta_decay * Giter_ols + (1-Beta_decay) * grad_OLSn * grad_OLSn) #Moving average of the sqaured gradients
            update_ols = grad_OLSn * eta / (epsilon + np.sqrt(Giter_ols)) #Updating OLS gradient with RMSProp
            theta_gdOLSn -= update_ols #Updating theta 
    

        #gradient descent Ridge
        tol = 1e-10
        for t in range(num_iters):
        # Compute gradients for Ridge
            grad_Ridgen = (2.0/n)*X_train_s.T @ (X_train_s @ (theta_gdRidgen)-y_scaled_train)+2*lam*theta_gdRidgen 
            Giter_ridge = (Beta_decay * Giter_ridge + (1-Beta_decay) * grad_Ridgen * grad_Ridgen) #Moving average of the sqaured gradients
            update_ridge = grad_Ridgen * eta / (epsilon + np.sqrt(Giter_ridge)) #Updating Ridge gradient with RMSProp
            theta_gdRidgen -= update_ridge
    
            if (np.linalg.norm(grad_Ridgen*eta) < tol):
                print(f'loop broken at {str(t)} for degree: {p}')
                break
        
        #Predicting y tilde for both OLS and Ridge - for both test and train data. Though we will only plot the test data, but the train data is then avaible for use if wished.
        y_pred_train_OLS = (X_train_s @ theta_gdOLSn + y_mean)
        y_pred_test_OLS = (X_test_s @ theta_gdOLSn + y_mean)
        y_pred_train_Ridge = (X_train_s @theta_gdRidgen + y_mean)
        y_pred_test_Ridge = (X_test_s @ theta_gdRidgen + y_mean)

        #Calculate MSE values and R2 scores using the Scikit learn functions:
        #Links to the functions used:
        #MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics
        #R2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        r2_gdOLSn_test[p-1] = r2_score(y_test, y_pred_test_OLS)
        r2_gdRidgen_test[p-1] = r2_score(y_test, y_pred_test_Ridge)
        r2_gdOLSn_train[p-1] = r2_score(y_train, y_pred_train_OLS)
        r2_gdRidgen_train[p-1] = r2_score(y_train, y_pred_train_Ridge)
        mse_gdOLSn_test[p-1] =  MSE(y_test, y_pred_test_OLS)
        mse_gdOLSn_train[p-1] =  MSE(y_train, y_pred_train_OLS)
        mse_gdRidgen_test[p-1] = MSE(y_test, y_pred_test_Ridge)
        mse_gdRidgen_train[p-1] =  MSE(y_train, y_pred_train_Ridge)

        #Appending the results to the initizialed empty lists to later plot the results for the optimizers
        mse_rmsprop_test_ols.append(mse_gdOLSn_test)
        mse_rmsprop_test_ridge.append(mse_gdRidgen_test)
        r2_rmsprop_test_ols.append(r2_gdOLSn_test)
        r2_rmsprop_test_ridge.append(r2_gdRidgen_test)

    return mse_rmsprop_test_ols, mse_rmsprop_test_ridge, r2_rmsprop_test_ols, r2_rmsprop_test_ridge


#Lastly we have ADAM: a combination of both Momentum and RMSProp

#ADAM 
def gradient_adam(x, y, n_feat, degree=16):
    #define variables 
    num_iters = 500
    eta = 0.01
    lam = 0.1 #fixed lambda for the Ridge
    epsilon = 1e-8 #Small value to avoid division by zero
    theta_gdOLSn = np.zeros(n_feat) #OLS
    theta_gdRidgen = np.zeros(n_feat) #Ridge
    r2_gdOLSn_test = np.zeros(degree)
    r2_gdRidgen_test = np.zeros(degree)
    r2_gdOLSn_train = np.zeros(degree)
    r2_gdRidgen_train = np.zeros(degree)
    degree_plot = np.linspace(1,degree, degree)
    mse_gdOLSn_test = np.zeros(degree)
    mse_gdOLSn_train = np.zeros(degree)
    mse_gdRidgen_test = np.zeros(degree)
    mse_gdRidgen_train = np.zeros(degree)
    theta1 = 0.9 #First moment decay rate
    theta2 = 0.999 #Second moment decay rate
    first_moment_ols = 0.0
    second_moment_ols = 0.0
    first_moment_ridge = 0.0 
    second_moment_ridge = 0.0

    #Initialize arrays to be used in the comparison subplot in a later function:
    mse_adam_test_ols = []
    mse_adam_test_ridge = []
    r2_adam_test_ols = []
    r2_adam_test_ridge = []
    
    #loop thorugh polynomial degree
    for p in range(1,degree+1):
    #Create X and scaling the data 
        X = polynomial_features(x,15, intercept = False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler(with_std=True)
        scaler.fit(X_train) 
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled_train = (y_train - y_mean)/y_std

        #gradient descent OLS 
        for t in range(num_iters):
            t += 1
            grad_OLSn = (2.0/n)*X_train_s.T @ (X_train_s @ theta_gdOLSn-y_scaled_train)
            #Calculating the first moment of Adam - the mean estimate
            first_moment_ols = theta1 * first_moment_ols + (1 - theta1) * grad_OLSn
            #Calculating the second moment of Adam - the variance estimate 
            second_moment_ols = theta2 * second_moment_ols + (1 - theta2) * grad_OLSn * grad_OLSn
            #Bias correction for the first and second moments
            first_term_ols = first_moment_ols / (1-theta1 **t)
            second_term_ols = second_moment_ols / (1-theta2 **t)
            #Updating theta with Adam
            theta_gdOLSn -= eta * first_term_ols / (np.sqrt(second_term_ols) + epsilon)
    

        #gradient descent Ridge
        tol = 1e-10
        for t in range(1,num_iters+1):
        # Compute gradients for Ridge
            grad_Ridgen = (2.0/n)*X_train_s.T @ (X_train_s @ (theta_gdRidgen)-y_scaled_train)+2*lam*theta_gdRidgen
            #Calculating the first moment of Adam - the mean estimate
            first_moment_ridge = theta1 * first_moment_ridge + (1 - theta1) * grad_Ridgen
            #Calculating the second moment of Adam - the variance estimate
            second_moment_ridge = theta2 * second_moment_ridge + (1 - theta2) * grad_Ridgen * grad_Ridgen
            #Bias correction for the first and second moments
            first_term_ridge = first_moment_ridge / (1-theta1 **t)
            second_term_ridge = second_moment_ridge / (1-theta2 **t)
            #Updating theta with Adam
            theta_gdRidgen -= eta * first_term_ridge / (np.sqrt(second_term_ridge) + epsilon)

    
            if (np.linalg.norm(grad_Ridgen*eta) < tol):
                print(f'loop broken at {str(t)} for degree: {p}')
                break
        
        #predicting y tilde for both OLS and Ridge - for both test and train data. Though we will only plot the test data, but the train data is then avaible for use if wished. 
        y_pred_train_OLS = (X_train_s @ theta_gdOLSn + y_mean)
        y_pred_test_OLS = (X_test_s @ theta_gdOLSn + y_mean)
        y_pred_train_Ridge = (X_train_s @theta_gdRidgen + y_mean)
        y_pred_test_Ridge = (X_test_s @ theta_gdRidgen + y_mean)
        
        #Calculate MSE values and R2 scores using the Scikit learn functions:
        #Links to the functions used:
        #MSE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
        #R2: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        r2_gdOLSn_test[p-1] = r2_score(y_test, y_pred_test_OLS)
        r2_gdRidgen_test[p-1] = r2_score(y_test, y_pred_test_Ridge)
        r2_gdOLSn_train[p-1] = r2_score(y_train, y_pred_train_OLS)
        r2_gdRidgen_train[p-1] = r2_score(y_train, y_pred_train_Ridge) 
        mse_gdOLSn_test[p-1] =  MSE(y_test, y_pred_test_OLS)
        mse_gdOLSn_train[p-1] =  MSE(y_train, y_pred_train_OLS)
        mse_gdRidgen_test[p-1] = MSE(y_test, y_pred_test_Ridge)
        mse_gdRidgen_train[p-1] =  MSE(y_train, y_pred_train_Ridge)

        #Appending the results to the initizialed empty lists to later plot the results for the optimizers
        mse_adam_test_ols.append(mse_gdOLSn_test)
        mse_adam_test_ridge.append(mse_gdRidgen_test)
        r2_adam_test_ols.append(r2_gdOLSn_test)
        r2_adam_test_ridge.append(r2_gdRidgen_test)

    return mse_adam_test_ols, mse_adam_test_ridge, r2_adam_test_ols, r2_adam_test_ridge

#To visualize the results from the optimizers, the last function creates subplots for OLS and Ridge regression containing all optimization methods with the MSE and R2 scores:
def all_learning_rates_plot(x, y, n_feat=16):
    #Run all the optimization codes:
    mse_momentum_ols, mse_momentum_ridge, r2_momentum_ols, r2_momentum_ridge = gradient_momentum(x,y,n_feat)
    mse_adagrad_ols, mse_adagrad_ridge, r2_adagrad_ols, r2_adagrad_ridge = gradient_adagrad(x,y,n_feat)
    mse_rmsprop_ols, mse_rmsprop_ridge , r2_rmsprop_ols, r2_rmsprop_ridge = gradient_rmsprop(x,y,n_feat)
    mse_adam_ols, mse_adam_ridge, r2_adam_ols, r2_adam_ridge = gradient_adam(x,y,n_feat)
    #linspace for the abscissa 
    degree = 16
    x_lab = np.linspace(1,degree,degree)

    #Subplot 1 for OLS - all optimizers
    fig1,ax1 = plt.subplots(1,2,figsize=(16,8))

    ax1[0].plot(x_lab, mse_momentum_ols[0], label = 'Momentum', color = 'm')
    ax1[0].plot(x_lab, mse_adagrad_ols[0], label = 'AdaGrad', color = 'darkolivegreen')
    ax1[0].plot(x_lab, mse_rmsprop_ols[0], label = 'RMSProp', color = 'royalblue')
    ax1[0].plot(x_lab, mse_adam_ols[0], label = 'Adam', color = 'sienna')
    ax1[1].plot(x_lab, r2_momentum_ols[0], label = 'Momentum', color = 'm')
    ax1[1].plot(x_lab, r2_adagrad_ols[0], label = 'Adagrad', color = 'darkolivegreen')
    ax1[1].plot(x_lab, r2_rmsprop_ols[0], label = 'RMSProp', color = 'royalblue')
    ax1[1].plot(x_lab, r2_adam_ols[0], label = 'Adam', color = 'sienna')
    fig1 = plt.suptitle(f'Gradient Descent with updated learningrates for OLS with N = 500')
    fig1 = plt.tight_layout()

    #Subplot 2 for Ridge - all optimizers
    fig2,ax2 = plt.subplots(1,2,figsize=(16,8))
    ax2[0].plot(x_lab,mse_momentum_ridge[0], label = 'Momentum', color = 'm')
    ax2[0].plot(x_lab, mse_adagrad_ridge[0], label = 'AdaGrad', color = 'darkolivegreen')
    ax2[0].plot(x_lab, mse_rmsprop_ridge[0], label = 'RMSProp', color = 'royalblue')
    ax2[0].plot(x_lab, mse_adam_ridge[0], label = 'Adam', color = 'sienna')
    ax2[1].plot(x_lab, r2_momentum_ridge[0], label = 'Momentum', color = 'm')
    ax2[1].plot(x_lab, r2_adagrad_ridge[0], label = 'Adagrad', color = 'darkolivegreen')
    ax2[1].plot(x_lab, r2_rmsprop_ridge[0], label = 'RMSProp', color = 'royalblue')
    ax2[1].plot(x_lab, r2_adam_ridge[0], label = 'Adam', color = 'sienna')

    #setting titles and labels 
    ax1[0].set_xlabel('Polynomial Degree')
    ax1[1].set_xlabel('Polynomial Degree')
    ax2[0].set_xlabel('Polynomial Degree')
    ax2[1].set_xlabel('Polynomial Degree')

    ax1[0].set_ylabel('MSE')
    ax2[0].set_ylabel('MSE')
    ax1[1].set_ylabel(r'$R^2$')
    ax2[1].set_ylabel(r'$R^2$')

    ax1[0].set_title(f'MSE for Test data')
    ax2[0].set_title(f'MSE for Test data')
    ax1[1].set_title(r' $R^2$ score for Test data')
    ax2[1].set_title(r' $R^2$ score for Test data')

    ax1[0].grid(True)
    ax1[1].grid(True)
    ax2[0].grid(True)
    ax2[1].grid(True)
    ax1[0].legend()
    ax1[1].legend()
    ax2[0].legend()
    ax2[1].legend()

    fig2 = plt.suptitle(f'Gradient Descent with updated learningrates for Ridge with N = 500')
    fig2 = plt.tight_layout()

#Common comments
#Initialize empty arrays for later use
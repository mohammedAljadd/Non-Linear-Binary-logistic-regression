# Non-Linear-Binary-logistic-regression

In the previous project [Linear-Binary-logistic-regression](https://github.com/mohammedAljadd/Linear-Binary-logistic-regression), we saw how to implement logistic regression with linear decision boundary, in this one, we will implement non linear classifier that will classify some data points with a non linear boundary. We will be using Scikit learn library. If you want to train your model from scratch, you will find everything in the previous project.

# Dataset

The shape of out dataset is (118, 3), so we have 118 exampls, two features and one output vector. If we plot the data points, it seems we cannot fit to our data with a linear decision boundary.

![alt text](https://github.com/mohammedAljadd/Non-Linear-Binary-logistic-regression/blob/main/plots/data.PNG)

One way to ﬁt the data better is to create more features from each data point. We will map the features into all polynomial terms of x1 and x2 up to the sixth power. 
So intead of having h(θ) = θ0 + θ.x1 + θ2.x2 our  hypothesis will be like this :  h(θ) = θ0 + θ1.x1 + θ2.x2 + θ3.x1² + θ4.x1.x2 + .. + θ29.x2⁶

So this is our new features :

![alt text](https://miro.medium.com/max/916/1*-n0H6dB-gVYUXg3nGvTAgw.png)

# Libraries 

    from sklearn import datasets                        
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    
    
# Functions :

    def sigmoid(z):
        sig = 1 / (1 + np.exp(-z))
        return sig

    def mapFeature(X1, X2, degree):
        res = np.ones(X1.shape[0])
        for i in range(1,degree + 1):
            for j in range(0,i + 1):
                res = np.column_stack((res, (X1 ** (i-j)) * (X2 ** j)))
        return res

As the decision boundary is not linear, we will calculate h(θ_optimum) and we will plot the contour corresponding to level = 0 because the boundary is equivalent to h(θ_optimum) = 0 


    def plotDecisionBoundary(theta,degree, axes):
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        U,V = np.meshgrid(u,v)
        # convert U, V to vectors for calculating additional features
        # using vectorized implementation
        U = np.ravel(U)
        V = np.ravel(V)
        Z = np.zeros((len(u) * len(v)))

        # Feature mapping
        X_poly = mapFeature(U, V, degree)
        X_poly = np.hstack((np.ones((X_poly.shape[0],1)),X_poly))
        Z = X_poly.dot(theta)

        # reshape U, V, Z back to matrix
        U = U.reshape((len(u), len(v)))
        V = V.reshape((len(u), len(v)))
        Z = Z.reshape((len(u), len(v)))

        cs = axes.contour(U,V,Z,levels=[0],cmap= "Greys_r")
        axes.legend(labels=['class 1', 'class 0', 'Decision Boundary'])
        return cs
        
        
 # Loading data and variables initialization :
 
    data = pd.read_csv('data.txt',header=None)
    data = data.to_numpy()
    m = data.shape[0]
    x1 = data[:,0].reshape((m, 1))
    x2 = data[:,1].reshape((m, 1))
    y = data[:,2].reshape((m, 1))
    x = np.hstack((x1,x2))

    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.1, random_state=0)
    degree = 6 # polynomial degree
    
    
# Model training :


    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression()
    y_train = y_train.ravel()
    X_train = mapFeature(X_train[:,0], X_train[:,1], degree)
    logisticRegr.fit(X_train, y_train)
    intercept = logisticRegr.intercept_
    coefs = logisticRegr.coef_
    optimum = np.vstack((intercept,coefs.reshape(X_train.shape[1],1))) # concatenate the intercept with coefficients of θ
    
    
# Plotting decision boundary

    fig, axes = plt.subplots();
    axes.set_xlabel('Feature 1')
    axes.set_ylabel('Feature 2')
    plt.scatter(x1[x1==x1-y],x2[x2==x2-y],c='r',label='class 0')
    plt.scatter(x1[x1!=x1-y],x2[x2!=x2-y],c='g',label='class 1')
    plotDecisionBoundary(optimum, degree, axes)
    
    
![alt text](https://github.com/mohammedAljadd/Non-Linear-Binary-logistic-regression/blob/main/plots/boundary.PNG)


# Model evaluation :
Let's evaluate our model in test data :

    X_test = mapFeature(X_test[:,0], X_test[:,1], degree)
    R_test=logisticRegr.score(X_test,y_test)*100
    print('Accuracy of your model is ',np.round(R_test),'%')
    
Accuracy of your model is  83.0 %

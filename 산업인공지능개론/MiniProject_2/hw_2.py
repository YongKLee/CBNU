import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


def DecisionTreeReg():

    df = pd.read_csv('CT_log.csv',sep=',')

    X = df[['Core_load_0']].values
    y = df['Core 0 Temp'].values


    tree = DecisionTreeRegressor(max_depth = 3)
    tree.fit(X,y)

    
    sort_idx = X.flatten().argsort()
    plt.scatter(X[sort_idx],y[sort_idx], c ='lightblue')
    plt.plot(X[sort_idx],tree.predict(X[sort_idx]),
                                      color = 'red',linewidth=2)
    plt.xlabel('Load')
    plt.ylabel('degree')
    plt.show()



def RegTest():
    df = pd.read_csv('CT_log.csv', sep=',')

    X = df[['Core_load_0']].values
    y = df['Core 0 Temp'].values

    sc_x = StandardScaler()
    sc_y = StandardScaler()

    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y[:,np.newaxis]).flatten()
    X_train,X_test,y_train,y_test = train_test_split(X_std,y_std,test_size=0.5,random_state=123)

    linear = LinearRegression()
    ridge = Ridge(alpha=1.0,random_state=0)
    lasso = Lasso(alpha=1.0, random_state=0)
    enet = ElasticNet(alpha=1.0, l1_ratio=0.5)

    linear.fit(X_train,y_train)
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    enet.fit(X_train, y_train)

    linear_pred = linear.predict(X_train)
    ridge_pred = ridge.predict(X_train)
    lasso_pred = lasso.predict(X_train)
    enet_pred = enet.predict(X_train)

    print('Linear - RMSE for training data      : ',np.sqrt(mean_squared_error(y_train, linear_pred)))
    print('Ridge - RMSE for training data       : ', np.sqrt(mean_squared_error(y_train, ridge_pred)))
    print('Lasso - RMSE for training data       : ', np.sqrt(mean_squared_error(y_train, lasso_pred)))
    print('Elastic Net - RMSE for training data : ', np.sqrt(mean_squared_error(y_train, enet_pred)))

    linear_pred = linear.predict(X_test)
    ridge_pred = ridge.predict(X_test)
    lasso_pred = lasso.predict(X_test)
    enet_pred = enet.predict(X_test)

    print('Linear - RMSE for test data      : ', np.sqrt(mean_squared_error(y_test, linear_pred)))
    print('Ridge - RMSE for test data       : ', np.sqrt(mean_squared_error(y_test, ridge_pred)))
    print('Lasso - RMSE for test data       : ', np.sqrt(mean_squared_error(y_test, lasso_pred)))
    print('Elastic Net - RMSE for test data : ', np.sqrt(mean_squared_error(y_test, enet_pred)))



def Ransac():
    df = pd.read_csv('CT_log.csv', sep=',')

    X = df[['Core_load_0']].values
    y = df['Core 0 Temp'].values

    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                             loss='absolute_loss',residual_threshold=5.0,random_state=0)

    ransac.fit(X,y)

    inlier_mssk = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mssk)
    line_X = np.arange(3,10,1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])

    plt.scatter(X[inlier_mssk], y[inlier_mssk], c = 'steelblue', edgecolors='white', marker='o', label ='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c = 'limegreen', edgecolors='white', marker='s', label ='Outliers')
    plt.plot(line_X, line_y_ransac,color='black',lw = 2)
    plt.xlabel('Core Load')
    plt.ylabel('Core degree')
    plt.legend(loc = 'upper left')
    plt.show()

def SVR_Test():
    df = pd.read_csv('CT_log.csv', sep=',')
    X = df[['Core_load_0']].values
    y = df['Core 0 Temp'].values

    svr_rbf = SVR(kernel='rbf', C = 100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C = 100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree = 3, epsilon = .1, coef0 = 1)

    lw = 2

#    svrs=[svr_rbf,svr_lin,svr_poly]
#    kernel_label = ['RBF','Linear', 'Polyn                                                                                                                                                                                                                                                                                                                                                                                                                       omial']
#    model_color = ['m','c','g']

    svrs=[svr_rbf,svr_lin]
    kernel_label = ['RBF','Linear']
    model_color = ['m','c','g']
    fig, axes = plt.subplots(nrows =1 , ncols =2, figsize=(9,7),sharey= True)

    print('enter for loop')
    for ix, svr in enumerate(svrs):
        axes[ix].plot(X, svr.fit(X,y).predict(X), color = model_color[ix], lw = lw,
                      label = '{} model'.format(kernel_label[ix]))

        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor ="none",
                      edgecolor=model_color[ix], s = 50, label = '{} support vectors'.format(kernel_label[ix]))

        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)),svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor = 'none', edgecolor='k',s= 50 , label = 'other training data')

        axes[ix].legend(loc = 'upper center', bbox_to_anchor=(0.5,1.1),ncol =1 ,fancybox = True, shadow = True)

    fig.text(0.5,0.04, 'data', ha = 'center', va = 'center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation = 'vertical')
    #fig.suptitile("Support Vector Regression", fontsize = 14)
    plt.show()
if __name__ == '__main__':
    DecisionTreeReg()
    RegTest()
    Ransac()
    SVR_Test()
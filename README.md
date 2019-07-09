Projection Pursuit Dimension Reduction
======================================

A scikit-learn compatible Python 3 package for Projection Pursuit Dimension Reduction. 
This class implements a very general framweork for projection pursuit, giving access to 
methods ranging from PP-PCA to CAPI generalized betas.  
The package uses the grid algorithm\[1\], the most numerically stable and accurate PP algorithm. 

Description
-----------

Projection pursuit (PP) provides a very general framework for dimension reduction and regression. The
`ppdire` package provides a framework to calculate PP estimates based on a wide variety of projection 
indices. 

While the package will also work with user-defined projection indices, a set of projection indices are 
included into the package as two ancillary classes: 
- `dicomo` for (co-)moment statistics 
- `capi` specifically for analyzing financial market returns based on a linear combination of co-moments 

When using the `dicomo` class as a plugin, several well-known multivariate dimension reduction techniques 
are accessible, as well as robust alternatives thereto. For more details, see the Example below. 

Note: all the methods contained in this package have been designed for continuous data. They do not work correctly for caetgorical or textual data. 
        
The code is aligned to ScikitLearn, such that modules such as GridSearchCV can flawlessly be applied to it. 

The repository contains
- The estimator (ppdire.py) 
- A class to estimate co-moments (dicomo.py)
- A class for the co-moment analysis projection index (capi.py)
- Ancillary functions for co-moment estimation (_dicomo_utils.py)

How to install
--------------
The package is distributed through PyPI, so install through: 
        
        pip install ppdire
    

The ppdire class
================

Dependencies
------------
- From <sklearn.base>: BaseEstimator,TransformerMixin,RegressorMixin
- From <sklearn.utils>: _BaseComposition
- copy
- <scipy.stats>:
- From <scipy.linalg>: pinv2
- numpy 
- From <statsmodels.regression.quantile_regression>: QuantReg
- From <sklearn.utils.extmath>: svd_flip
- From <sprm>: rm, robcent
- From <sprm._m_support_functions>: MyException
- warnings


Parameters
----------
- `projection_index`, function or class. `dicomo` and `capi` supplied in this
            package can both be used, but user defined projection indices can 
            be processed 
- `pi_arguments`, dict. Dict of arguments to be passed on to `projection index` 
- `n_components`, int. number of components to be estimated 
- `trimming`, float. trimming percentage for projection index, to be entered as pct/100 
- `alpha`, float. Continuum coefficient. Only relevant if `ppdire` is used to 
            estimate (classical or robust) continuum regression.  
- `center`, str. How to center the data. options accepted are options from
            `sprm.robcent` 
- `center_data`, bool. 
- `scale_data`, bool. Note: if set to `False`, convergence to correct optimum 
            is not a given. Will throw a warning. 
- `whiten_data`, bool. Typically used for ICA (kurtosis as PI)
- `square_pi`, bool. Whether to square the projection index upon evaluation. 
- `copy`, bool. Whether to make a deep copy of the input data or not. 
- `verbose`, bool. Set to `True` prints the iteration number. 
- `return_scaling_object`, bool.
Note: several interesting parameters can also be passed to the `fit` method.   

Attributes
----------
Attributes always provided 
-  `x_weights_`: X block PPDIRE weighting vectors (usually denoted W)
-  `x_loadings_`: X block PPDIRE loading vectors (usually denoted P)
-  `x_scores_`: X block PPDIRE score vectors (usually denoted T)
-  `x_ev_`: X block explained variance per component
-  `x_Rweights_`: X block SIMPLS style weighting vectors (usually denoted R)
-  `x_loc_`: X block location estimate 
-  `x_sca_`: X block scale estimate
-  `crit_values_`: vector of evaluated values for the optimization objective. 
-  `Maxobjf_`: vector containing the optimized objective per component. 

Attributes created when more than one block of data is provided: 
-  `C_`: vector of inner relationship between response and latent variables block
-  `coef_`: vector of regression coefficients, if second data block provided 
-  `intercept_`: intercept
-  `coef_scaled_`: vector of scaled regression coeeficients (when scaling option used)
-  `intercept_scaled_`: scaled intercept
-  `residuals_`: vector of regression residuals
-  `y_ev_`: y block explained variance 
-  `fitted_`: fitted response
-  `y_loc_`: y location estimate
-  `y_sca_`: y scale estimate

Attributes created only when corresponding input flags are `True`:
-   `whitening_`: whitened data matrix (usually denoted K)
-   `mixing_`: mixing matrix estimate
-   `scaling_object_`: scaling object from `sprm.robcent`


Methods
--------
- `fit(X, *args, **kwargs)`: fit model 
- `predict(X)`: make predictions based on fit 
- `transform(X)`: project X onto latent space 
- `getattr()`: get list of attributes
- `setattr(*kwargs)`: set individual attribute of sprm object 

The `fit` function takes several optional input arguments. These are flags that 
typically would not need to be cross-validated. They are: 
-   `y`, numpy vector or 1D matrix, either as `arg` directly or as `kwarg`
-   `h`, int. Overrides `n_components` for an individual call to `fit`. Use with caution. 
-   `dmetric`, str. Distance metric used internally. Defaults to `'euclidean'`
-   `ndir`, int. Number of directions to be calculated in grid plane. A higher 
    value increase the quality of the solution. 
-   `maxiter`, int. Maximal number of iterations allowed for convergence.
-   `compression`, bool. Whether to SVD compress data if they are flat (p>n). 
-   `mixing`, bool. Return mixing matrix? 
-   `regopt`, str. Which type of regression to apply for regression of y onto the 
    x scores. Defaults to `'OLS'`. Other options are `'robust'` (calls `sprm.rm`) or 
    `'quantile'` (calls `statsmodels.regression.quantreg`). Further parameters 
    to these methods can be passed on as additional `kwargs`. 
  

Ancillary functions 
-------------------
- `dicomo` (class):  (co-)moments 
- `capi` (class): co-moment analysis projection index 

Example
-------
To run a toy example: 
- Source packages and data: 

        import pandas as ps
        data = ps.read_csv("./Returns_shares.csv")
        columns = data.columns[2:8]
        data = data.values[:,2:8]
        X = data[:,0:5]
        y = data[:,5]
        X0 = X.astype('float')
        y0 = y.astype('float')
        
- Estimate and predict by SPRM
        
        from sprm import sprm
        res_sprm = sprm(2,.8,'Hampel',.95,.975,.999,'median','mad',True,100,.01,'ally','xonly',columns,True)
        res_sprm.fit(X0[:2666],y0[:2666])
        res_sprm.predict(X0[2666:])
        res_sprm.transform(X0[2666:])
        res_sprm.weightnewx(X0[2666:])
        res_sprm.get_params()
        res_sprm.set_params(fun="Huber")
        
- Cross-validated using GridSearchCV: 
        
        import numpy as np
        from sklearn.model_selection import GridSearchCV 
        res_sprm_cv = GridSearchCV(sprm(), cv=10, param_grid={"n_components": [1, 2, 3], 
                                   "eta": np.arange(.1,.9,.05).tolist()})  
        res_sprm_cv.fit(X0[:2666],y0[:2666])  
        res_sprm_cv.best_params_
        
        
The Robust M (RM) estimator
===========================

RM has been implemented to be consistent with SPRM. It takes the same arguments, except for 'eta' and 'n_components', 
because it does not perform dimension reduction nor variable selection. For the same reasons, the outputs are limited to regression
outputs. Therefore, dimension reduction outputs like x_scores_, x_loadings_, etc. are not provided. 
        
  Estimate and predict by RM: 
  
        from sprm import rm
        res_rm = rm('Hampel',.95,.975,.999,'median','mad','specific',True,100,.01,columns,True)
        res_rm.fit(X0[:2666],y0[:2666])
        res_rm.predict(X0[2666:])
        
The Sparse NIPALS (SNIPLS) estimator
====================================

SNIPLS is the non-robust sparse univariate PLS algorithm \[3\]. SNIPLS has been implemented to be consistent with SPRM. It takes the same arguments, except for 'fun' and 'probp1' through 'probp3', since these are robustness parameters. For the same reasons, the outputs are limited to sparse dimension reduction and regression outputs. Robustness related outputs like x_caseweights_ cannot be provided.
        
  Estimate and predict by SNIPLS: 
  
        from sprm import snipls
        res_snipls = snipls(n_components=4, eta=.5)
        res_snipls.fit(X0[:2666],y0[:2666])
        res_snipls.predict(X0[2666:])
        
        
 
Plotting functionality
======================

The file sprm_plot.py contains a set of plot functions based on Matplotlib. The class sprm_plot contains plots for sprm objects, wheras the class sprm_plot_cv contains a plot for cross-validation. 

Dependencies
------------
- pandas
- numpy
- matplotlib.pyplot
- for plotting cross-validation results: sklearn.model_selection.GridSearchCV

Paramaters
----------
- res_sprm, sprm. An sprm class object that has been fit.  
- colors, list of str entries. Only mandatory input. Elements determine colors as: 
    - \[0\]: borders of pane 
    - \[1\]: plot background
    - \[2\]: marker fill
    - \[3\]: diagonal line 
    - \[4\]: marker contour, if different from fill
    - \[5\]: marker color for new cases, if applicable
    - \[6\]: marker color for harsh calibration outliers
    - \[7\]: marker color for harsh prediction outliers
- markers, a list of str entries. Elements determkine markers for: 
    - \[0\]: regular cases 
    - \[1\]: moderate outliers 
    - \[2\]: harsh outliers 
    
Methods
-------
- plot_coeffs(entity="coef_",truncation=0,columns=[],title=[]): Plot regression coefficients, loadings, etc. with the option only to plot the x% smallest and largets coefficients (truncation) 
- plot_yyp(ytruev=[],Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False): Plot y vs y predicted. 
- plot_projections(Xn=[],label=[],components = [0,1],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False): Plot score space. 
- plot_caseweights(Xn=[],label=[],names=[],namesv=[],title=[],legend_pos='lower right',onlyval=False,mode='overall'): Plot caseweights, with the option to plot 'x', 'y' or 'overall' case weights for cases used to train the model. For new cases, only 'x' weights can be plotted. 

Remark
------
The latter 3 methods will work both for cases that the models has been trained with (no additional input) or new cases (requires Xn and in case of plot_ypp, ytruev), with the option to plot only the latter (option onlyval = True). All three functions have the option to plot case names if supplied as list.       

Ancillary classes
------------------ 
- sprm_plot_cv has method eta_ncomp_contour(title) to plot sklearn GridSearchCV results 
- ABline2D plots the first diagonal in y vs y predicted plots. 

Example (continued) 
-------------------
- initialize some values: 
        
        colors = ["white","#BBBBDD","#0000DD",'#1B75BC','#4D4D4F','orange','red','black']
        markers = ['o','d','v']
        label = ["AIG"]
        names = [str(i) for i in range(1,len(res_sprm.y)+1)]
        namesv = [str(i) for i in range(1,len(y0[2667:])+1)]
        
- run sprm.plot: 
        
        from sprm import sprm_plot
        res_sprm_plot = sprm_plot(res_sprm,colors)
        
- plot coefficients: 

        res_sprm_plot.plot_coeffs(title="All AIG SPRM scaled b")
        res_sprm_plot.plot_coeffs(truncation=.05,columns=columns,title="5% smallest and largest AIG sprm b")
        
  ![AIG sprm regression coefficients](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_b.png "AIG SPRM regression coefficients")

- plot y vs y predicted, training cases only: 

        res_sprm_plot.plot_yyp(label=label,title="AIG SPRM y vs. y predicted")
        res_sprm_plot.plot_yyp(label=label,names=names,title="AIG SPRM y vs. y predicted")

  ![AIG sprm y vs y predicted, taining set](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_yyp_train.png "AIG SPRM y vs y predicted, training set")
  
- plot y vs y predicted, including test cases
  
        res_sprm_plot.plot_yyp(ytruev=y0[2667:],Xn=X0[2667:],label=label,names=names,namesv=namesv,title="AIG SPRM y vs. 
                y predicted")            
        res_sprm_plot.plot_yyp(ytruev=y0[2667:],Xn=X0[2667:],label=label,title="AIG SPRM y vs. y predicted")
        
   ![AIG sprm y vs y predicted, taining set](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_yyp_train_test.png "AIG SPRM y vs y predicted")

- plot y vs y predicted, only test set cases: 

        res_sprm_plot.plot_yyp(ytruev=y0[2667:],Xn=X0[2667:],label=label,title="AIG SPRM y vs. y predicted",onlyval=True)
  
- plot score space, options as above, with the second one shown here: 

        res_sprm_plot.plot_projections(Xn=X0[2667:],label=label,names=names,namesv=namesv,title="AIG SPRM score space, components 1 and 2")
        res_sprm_plot.plot_projections(Xn=X0[2667:],label=label,title="AIG SPRM score space, components 1 and 2")
        res_sprm_plot.plot_projections(Xn=X0[2667:],label=label,namesv=namesv,title="AIG SPRM score space, components 1 and 2",onlyval=True)
        
  
   ![AIG sprm score space](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_T12.png "AIG SPRM score space")

- plot caseweights, options as above, with the second one shown here:

        res_sprm_plot.plot_caseweights(Xn=X0[2667:],label=label,names=names,namesv=namesv,title="AIG SPRM caseweights")
        res_sprm_plot.plot_caseweights(Xn=X0[2667:],label=label,title="AIG SPRM caseweights")
        res_sprm_plot.plot_caseweights(Xn=X0[2667:],label=label,namesv=namesv,title="AIG SPRM caseweights",onlyval=True)  
        
   ![AIG sprm caseweights](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_caseweights.png "AIG SPRM caseweights")
   

- plot cross-validation results: 

        from sprm import sprm_plot_cv
        res_sprm_plot_cv = sprm_plot_cv(res_sprm_cv,colors)
        res_sprm_plot_cv.eta_ncomp_contour()
        res_sprm_plot_cv.cv_score_table_
        
  ![AIG sprm CV results](https://github.com/SvenSerneels/sprm/blob/master/img/AIG_CV.png "AIG SPRM CV results")
  
        
References
----------
1. [Robust Multivariate Methods: The Projection Pursuit Approach](https://link.springer.com/chapter/10.1007/3-540-31314-1_32), Peter Filzmoser, Sven Serneels, Christophe Croux and Pierre J. Van Espen, in: From Data and Information Analysis to Knowledge Engineering,
        Spiliopoulou, M., Kruse, R., Borgelt, C., Nuernberger, A. and Gaul, W., eds., 
        Springer Verlag, Berlin, Germany,
        2006, pages 270--277.
2. [Partial robust M regression](https://doi.org/10.1016/j.chemolab.2005.04.007), Sven Serneels, Christophe Croux, Peter Filzmoser, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 79 (2005), 55-64.
3. [Sparse and robust PLS for binary classification](https://onlinelibrary.wiley.com/doi/abs/10.1002/cem.2775), I. Hoffmann, P. Filzmoser, S. Serneels, K. Varmuza, Journal of Chemometrics, 30 (2016), 153-162.
        

Release notes
=============

Version 0.2.1
-------------
- sprm now takes both numeric (n,1) np matrices and (n,) np.arrays as input 


Version 0.2.0
-------------
Changes compared to version 0.1: 
- All functionalities can now be loaded in modular way, e.g. to use plotting functions, now source the plot function separately:
        
        from sprm import sprm_plot 
        
- The package now includes a robust M regression estimator (rm.py), which is a multiple regression only variant of sprm. 
  It is based on the same iterative re-weighting scheme, buit does not perform dimension reduction, nor variable selection.
- The robust preprocessing routine (robcent.py) has been re-written so as to be more consistent with sklearn.

Version 0.3
-----------
All three estimators provided as separate classes in module:

        from sprm import sprm 
        from sprm import snipls
        from sprm import rm
        
Also, sprm now includes a check for zero scales. It will remove zero scale variables from the input data, and only use 
columns corresponding to nonzero predictor scales in new data. This check has not yet been built in for snipls or rm 
separately. 
        
Plus some minor changes to make it consistent with the latest numpy and matplotlib versions. 

Work to do
----------
- optimize alignment to sklearn
- optimize for speed 
- extend to multivariate responses (open research topic !)
- suggestions always welcome 

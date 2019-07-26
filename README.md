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
- `ndir`, int. Number of directions to calculate per iteration.
- `maxiter`, int. Maximal number of iterations.
- `regopt`, str. Regression option for regression step y~T. Can be set
                to `'OLS'` (default), `'robust'` (will run `sprm.rm`) or `'quantile'` 
                (`statsmodels.regression.quantreg`). 
- `center`, str. How to center the data. options accepted are options from
            `sprm.robcent` 
- `center_data`, bool. 
- `scale_data`, bool. Note: if set to `False`, convergence to correct optimum 
            is not a given. Will throw a warning. 
- `whiten_data`, bool. Typically used for ICA (kurtosis as PI)
- `square_pi`, bool. Whether to square the projection index upon evaluation. 
- `compression`, bool. If `True`, an internal SVD compression step is used for 
            flat data tables (p > n). Speds up the calculations. 
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
-   `mixing`, bool. Return mixing matrix? 
-   Further parameters to the regression methods can be passed on here 
    as additional `kwargs`. 
  

Ancillary functions 
-------------------
- `dicomo` (class):  (co-)moments 
- `capi` (class): co-moment analysis projection index 

Examples
========

Load and Prepare Data
---------------------
To run a toy example: 
- Source packages and data: 
        
        # Load data
        import pandas as ps
        import numpy as np
        data = ps.read_csv("./data/Returns_shares.csv")
        columns = data.columns[2:8]
        (n,p) = data.shape
        datav = np.matrix(data.values[:,2:8].astype('float64'))
        y = datav[:,0]
        X = datav[:,1:5]
        
        # Scale data
        from sprm import robcent
        centring = robcent()
        Xs = centring.fit(X)
        
Comparison of PP estimates to Scikit-Learn 
------------------------------------------
Let us at first run `ppdire` to produce slow, approximate PP estimates of 
PCA and PLS. This makes it easy to verify that the algorithm is correct. 
        
- Projection Pursuit as a slow, approximate way to compute PCA. Compare:
        
        # PCA ex Scikit-Learn 
        import sklearn.decomposition as skd
        skpca = skd.PCA(n_components=4)
        skpca.fit(Xs)
        skpca.components_.T # sklearn outputs loadings as rows ! 
        
        # PP-PCA  
        from ppdire import dicomo, ppdire
        pppca = ppdire(projection_index = dicomo, pi_arguments = {'mode' : 'var'}, n_components=4)
        pppca.fit(X)
        pppca.x_loadings_
        
- Likewise, projection pursuit as a slow, approximate way to compute PLS. Compare: 

        # PLS ex Scikit-Learn 
        skpls = skc.PLSRegression(n_components=4)
        skpls.fit(Xs,(y-np.mean(y))/np.std(y))
        skpls.x_scores_
        skpls.coef_ 
        Xs*skpls.coef_*np.std(y) + np.mean(y) 
        
        # PP-PLS 
        pppls = ppdire(projection_index = dicomo, pi_arguments = {'mode' : 'cov'}, n_components=4, square_pi=True)
        pppls.fit(X,y)
        pppls.x_scores_
        pppls.coef_scaled_ # Column 4 should agree with skpls.coef_
        pppls.fitted_  
        
Remark: Dimension Reduction techniques based on projection onto latent variables, 
such as PCA, PLS and ICA, are sign indeterminate with respect to the components. 
Therefore, signs of estimates by different algorithms can be opposed, yet the 
absolute values should be identical up to algorithm precision.  Here, this implies
that `sklearn` and `ppdire`'s `x_scores_` and `x_loadings` can have opposed signs,
yet the coefficients and fitted responses should be identical. 


Robust projection pursuit estimators
------------------------------------

- Robust PCA based on the Median Absolute Deviation (MAD) \[2\]. 
        
        lcpca = ppdire(projection_index = dicomo, pi_arguments = {'mode' : 'var', 'center': 'median'}, n_components=4)
        lcpca.fit(X)
        lcpca.x_loadings_
        # To extend to Robust PCR, just add y 
        lcpca.fit(X,y,ndir=1000,regopt='robust')
        
- Robust Continuum Regression \[3\] based on trimmed covariance: 

        rcr = ppdire(projection_index = dicomo, pi_arguments = {'mode' : 'continuum'}, n_components=4, trimming=.1, alpha=.5)
        rcr.fit(X,y=y,ndir=1000,regopt='robust')
        rcr.x_loadings_
        rcr.x_scores_
        rcr.coef_scaled_
        rcr.predict(X)
        
Remark: for RCR, the continuum parameter `alpha` tunes the result from multiple 
regression (`alpha` -> 0) via PLS (`alpha` = 1) to PCR (`alpha` -> Inf). Of course, 
the robust PLS option can also be accessed through `pi_arguments = {'mode' : 'cov'}, trimming=.1`. 


Projection pursuit to analyze market data (CAPI)
------------------------------------------------

        from ppdire import capi 
        est = ppdire(projection_index = capi, pi_arguments = {'max_degree' : 3,'projection_index': dicomo, 'scaling': False}, n_components=1, trimming=0,center_data=True,scale_data=True)
        est.fit(X,y=y,ndir=200)
        est.x_weights_
        # These data aren't the greatest illustration. Evaluating CAPI 
        # projections, makes more sense if y is a market index, e.g. SPX 
        
        
Cross-validating through `scikit-learn` 
---------------------------------------

        from sklearn.model_selection import GridSearchCV
        rcr_cv = GridSearchCV(ppdire(projection_index=dicomo, 
                                    pi_arguments = {'mode' : 'continuum'}), 
                              cv=10, 
                              param_grid={"n_components": [1, 2, 3], 
                                          "alpha": np.arange(.1,3,.3).tolist(),
                                          "trimming": [0, .15]
                                         }
                             )
        rcr_cv.fit(X[:2666],y[:2666]) 
        rcr_cv.best_params_
        rcr_cv.predict(X[2666:])
        
        
Data compression
----------------
While `ppdire` is very flexible and can project according to a very wide variety 
of projection indices, it can be computationally demanding. For flat data tables,
a workaround has been built in.  

        # Load flat data 
        datan = ps.read_csv("./ppdire/data/Glass_df.csv")
        X = datan.values[:,100:300]
        y = datan.values[:,2]
        
        # Now compare
        rcr = ppdire(projection_index = dicomo, 
                    pi_arguments = {'mode' : 'continuum'}, 
                    n_components=4, 
                    trimming=.1, 
                    alpha=.5, 
                    compression = False)
        rcr.coef_
        
        rcr = ppdire(projection_index = dicomo, 
                    pi_arguments = {'mode' : 'continuum'}, 
                    n_components=4, 
                    trimming=.1, 
                    alpha=.5, 
                    compression = True)
        rcr.coef_
        
However, compression will not work properly if the data contain several low scale 
varables. In this example, it will not work for `X = datan.values[:,8:751]`. This 
will throw a warning, and `ppdire` will continue woithout compression. 
        
        
        
        
Calling the projection indices independently 
--------------------------------------------
Both `dicomo` and `capi` can be useful as a consistent framework to call moments
themselves, or linear combinations of them. Let's extract univariate columns from
the data: 

        # Prepare univariate data
        x = datav[:,1]
        y = datav[:,2]
        
Now calculate some moments and compare them to `numpy`: 
        
        # Variance 
        covest = dicomo() 
        # division by n
        covest.fit(x,biascorr=False)
        np.var(x)
        # division by n-1 
        covest.fit(x,biascorr=True)
        np.var(x)*n/(n-1)
        # But we can also trim variance: 
        covest.fit(x,biascorr=False,trimming=.1)
        
        # MAD  
        import statsmodels.robust as srs
        covest.set_params(center='median')
        srs.mad(x)
        
        # 4th Moment 
        import scipy.stats as sps
        # if center is still median, reset it
        covest.set_params(center='mean')
        covest.fit(x,order=4)
        sps.moment(x,4)
        # Again, we can trim: 
        covest.fit(x,order=4,trimming=.2)
        
        #Kurtosis 
        covest.set_params(mode='kurt')
        sps.kurtosis(x,fisher=False,bias=False) 
        #Note that in scipy: bias = False corrects for bias
        covest.fit(x,biascorr=True,Fisher=False)
        
    
Likewise: co-moments

        # Covariance 
        covest.set_params(mode='com')
        data.iloc[:,2:8].cov() #Pandas Calculates n-1 division
        covest.fit(x,y=y,biascorr=True)
        
        # M4 (4th co-moment)
        covest.set_params(mode='com')
        covest.fit(x,y=y,biascorr=True,order=4,option=1)
        
        # Co-kurtosis
        covest.set_params(mode='cok')
        covest.fit(x,y=y,biascorr=True,option=1)
        
        
These are just some options of the set that can be explored in `dicomo`. 

        
References
----------
1. [Robust Multivariate Methods: The Projection Pursuit Approach](https://link.springer.com/chapter/10.1007/3-540-31314-1_32), Peter Filzmoser, Sven Serneels, Christophe Croux and Pierre J. Van Espen, in: From Data and Information Analysis to Knowledge Engineering,
        Spiliopoulou, M., Kruse, R., Borgelt, C., Nuernberger, A. and Gaul, W., eds., 
        Springer Verlag, Berlin, Germany,
        2006, pages 270--277.
2. Robust principal components and dispersion matrices via projection pursuit, Chen, Z. and Li, G., Research Report, Department of Statistics, Harvard University, 1981.
3. [Robust Continuum Regression](https://www.sciencedirect.com/science/article/abs/pii/S0169743904002667), Sven Serneels, Peter Filzmoser, Christophe Croux, Pierre J. Van Espen, Chemometrics and Intelligent Laboratory Systems, 76 (2005), 197-204.

    

Work to do
----------
- optimize alignment to `sklearn`
- align to some of `sprm` plotting functions
- optimize for speed 
- extend to multivariate responses (open research topic !)
- suggestions always welcome 

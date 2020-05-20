import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import get_scorer

import tensorflow as tf
from tensorflow import keras

def RandomSeedSearchCV(random_model_maker,X_train,y_train,w_train=None,N=50,
                        validation=0.1,cv=5,scoring='mean_squared_error',metric=None,metric_needs_weights=False,
                        plot_summary=True,shield_seed=True,verbose=True,
                        random_state=None,custom_fit=None,fit_params=None,**model_maker_kwargs):
    '''
    Generalized random-search model tuner! Uses random seeds and user-made model-maker function to search
    through "acceptable" models and find those that achieve the best cross_validation score.
    
    This function is just a shell to execute the search. It doesn't know anything about the models
    or the parameter space to search through; all of that is encapsulated in the user-provided model-making
    function.
    
    INPUTS:
      random_model_maker   - tl;dr, Function that takes integer and returns model. Returned model must implement
                                     fit and predict methods, like standard sklearn estimators do.
                                     
                             This should be a function defined by the user ahead of time. The function should
                             accept a random seed and then internally construct a model with randomly drawn 
                             hyperparameters. The idea behind this is to take responsibility of knowing the parameter
                             space and generating models within it off of the searcher and instead placing that 
                             responsibility on the user-provided model-maker. randomseed_searchCV will never know 
                             what type of model you're fitting or what range of which parameters you're sampling.
                             All it will know is that it can give random_model_maker an integer, and it will 
                             return a model that can be fit and used to predict.
                             
      **model_maker_kwargs - Any additional arguments to be passed to the model maker.
                           
      X_train,y_train      - Training data and labels.
      validation           - Fraction to split off for validation testing. Default is 0.1. If None,False,0,or 1
                              are provided, validation testing will not be done.
      cv                   - Integer N for N-fold cross validation.
      scoring              - Metric to use when ranking models.
      plot_summary         - Boolean whether or not to produce a beautiful summary plot!
      shield_seed          - Boolean whether or not to forcibly prevent the model maker from globally changing
                             the numpy random seed.
      verbose              - Boolean whether or not to show progress.
      random_state         - Random state to use for splitting off validation dataset.
    '''
    
    if verbose: progress = simple_progress()
    
    if shield_seed:
        modmkr = preserve_state(random_model_maker)
    else:
        modmkr = random_model_maker
    
    use_custom_fit = not custom_fit is None

    if fit_params is None:
        fit_params = {}
    
    #Get metric callable from sklearn. L8r I'm gonna make it
    # so the user can provide a callable metric themselves, because I
    # don't like that sklearn uses negative mse instead of positive. It drives me bonkers.
    if metric is None:
        metric = get_metric(scoring)
    
    #Split validation set out of the training set, if 
    if validation is None or validation==False or validation>=1 or validation <=0:
        use_val = False
        Xtra,ytra = X_train,y_train
        Xval,yval = None,None
        if not w_train is None:
            wtra = w_train
            wval = None
            fit_params.update({'sample_weight':wtra})
    else:
        use_val = True
        if w_train is None:
            Xtra,Xval,ytra,yval = train_test_split(X_train,y_train,
                                                   test_size=validation,
                                                   random_state=random_state)
        else:
            Xtra,Xval,ytra,yval,wtra,wval = train_test_split(X_train,y_train,w_train,
                                                             test_size=validation,
                                                             random_state=random_state)
            fit_params.update({'sample_weight':wtra})
    
    if not w_train is None and metric_needs_weights:
        metric = lambda model,X,y,w=wtra,old_metric=metric:old_metric(model,X,y,w)
    else:
        metric_needs_weights = False

    #Draw random seeds to use for model generation.
    seeds = np.random.choice(10*N,N,replace=False)
    
    #Create empty lists to store model performance measures.
    if cv > 1 :cv_scores = np.array([])
    train_metric = np.array([])
    valid_metric = np.array([])
    times = np.array([])
    
    #For each model, find cv_score and metrics. Also store training time.
    for i,seed in enumerate(seeds):
        if verbose: progress.update("%d/%d: Seed = %d"%(i+1,N,seed))
        model = modmkr(seed,**model_maker_kwargs)
        if cv > 1:
            cv_score = np.mean(cross_val_score(model,Xtra,ytra,scoring=metric,cv=cv,fit_params=fit_params))
            cv_scores = np.append(cv_scores,cv_score)
        
        start_time = time()
        if use_custom_fit:
            model = custom_fit(model,Xtra,ytra,**fit_params)
        else:
            model.fit(Xtra,ytra,**fit_params)
        train_metric = np.append(train_metric, metric(model,Xtra,ytra))
        if use_val:
            if metric_needs_weights:
                valid_metric = np.append(valid_metric, metric(model,Xval,yval,wval))
            else:
                valid_metric = np.append(valid_metric, metric(model,Xval,yval))
        times = np.append(times, time()-start_time)

    #Plot a summary when done.
    if plot_summary:
        fig,ax = plt.subplots()
        scat = ax.scatter(train_metric,valid_metric,c=times,cmap='coolwarm',s=50)
        cbar = fig.colorbar(scat,ax=ax)
        ax.set_xlabel("Training Metric")
        ax.set_ylabel("Validation Metric")
        cbar.ax.set_ylabel("Training Time")
        mi,ma = 0,1.1*np.max([np.max(train_metric),np.max(valid_metric)])
        ax.set_xlim(mi,ma)
        ax.set_ylim(mi,ma)
        ax.plot([mi,ma],[mi,ma],ls='--',color='black')
    
    if cv > 1: 
        sort = np.argsort(cv_scores)
        return np.c_[seeds[sort],cv_scores[sort],train_metric[sort],valid_metric[sort],times[sort]]
    else:
        sort = np.argsort(valid_metric)
        return np.c_[seeds[sort],train_metric[sort],valid_metric[sort],times[sort]]

################################################################################
###########################  Random Model-Makers  ##############################
################################################################################

def rand_util(inp1=0,inp2=1,dist='uniform',dtype='float',P_None=0.0,override=None):
    '''
    Convenience function to take a randomly sampled number in a variety of ways.

    INPUTS:
      inp1,inp2 - Inputs to the random number generator. If type='uniform', low = inp1 and high = inp2.
                                                         If type='normal', mean = inp1 and std = inp2.
      dist  - Type of random sampling. Options: 
               'uniform' - Draw from uniform distribution ranging from inp1 to inp2.
               'normal'  - Draw from normal distribution with mean = inp1 and std = inp2.
      dtype - Output type. 'int' or 'float'
      P_None - Probability of returning None instead of a numerical value.
      Override - If anything besides None is provided, no random sampling will take place, override will
                 be the returned value.
    OUTPUT:
      x - Random variable
    '''
    #Check if override given.
    if not override is None:
        return override

    #Draw to see if None will be returned.
    is_None = np.random.uniform(0.,1.) < P_None
    if is_None:
        return None
    
    #Draw value to be returned
    if dist=='uniform':
        if dtype=='int':
            return np.random.randint(inp1,inp2)
        elif dtype=='float':
            return np.random.uniform(inp1,inp2)
        else:
            raise ValueError("Unrecognized dtype %s. Try one of 'int', 'float'."%(dtype))
    elif dist=='normal':
        if dtype=='int':
            return int(np.round(np.random.normal(inp1,inp2)))
        elif dtype=='float':
            return np.random.normal(inp1,inp2)
        else:
            raise ValueError("Unrecognized dtype %s. Try one of 'int', 'float'."%(dtype))
    else:
        raise ValueError("Unrecognized distribution %s. Try one of 'uniform', 'normal'."%(dist))

#RandomForestRegressor random model maker.
def randomseed_rfr_maker(seed,**kwargs):
    distparams = {
         'n_estimators':None,         'ne_lo':20, 'ne_hi':200,
         'max_depth':None,            'md_lo': 3, 'md_hi' :15, 'md_Pnone':0.5,
         'min_samples_split':None,    'mss_lo':2, 'mss_hi':50,
         'min_samples_leaf':None,     'msl_lo':2, 'msl_hi':25,
         'max_features':None,         'mf_lo' :3, 'mf_hi' :13, 'mf_Pnone':0.3,
         'min_impurity_decrease':None,'mid_lo':0.,'mid_hi':0.4,
         'n_jobs':4}
    distparams.update(kwargs) #Update default distribution parameters with user input.

    #Set random seed.
    np.random.seed(seed)
    
    # Randomly draw parameters and store in kwargs dictionary.
    hyperparams = {}
    # (1) Draw n_estimators
    hyperparams['n_estimators'] =          rand_util(distparams['ne_lo'],distparams['ne_hi'],dist='uniform',dtype='int',
                                                override=distparams['n_estimators'])
    # (2) Draw max_depth
    hyperparams['max_depth'] =             rand_util(distparams['md_lo'],distparams['md_hi'],dist='uniform',dtype='int',
                                                P_None=distparams['md_Pnone'],override=distparams['max_depth'])
    # (3) Draw min_samples_split
    hyperparams['min_samples_split'] =     rand_util(distparams['mss_lo'],distparams['mss_hi'],dist='uniform',dtype='int',
                                                override=distparams['min_samples_split'])
    # (3) Draw min_samples_leaf
    hyperparams['min_samples_leaf'] =      rand_util(distparams['msl_lo'],distparams['msl_hi'],dist='uniform',dtype='int',
                                                override=distparams['min_samples_leaf'])
    # (4) Draw max_features
    hyperparams['max_features'] =          rand_util(distparams['mf_lo'],distparams['mf_hi'],dist='uniform',dtype='int',
                                                P_None=distparams['mf_Pnone'],override=distparams['max_features'])
    # (5) Draw min_impurity_decrease
    hyperparams['min_impurity_decrease'] = rand_util(distparams['mid_lo'],distparams['mid_hi'],dist='uniform',dtype='float',
                                                override=distparams['min_samples_leaf'])
    #Set n_jobs
    hyperparams['n_jobs'] = distparams['n_jobs']
    
    #Create the model!
    model = RandomForestRegressor(**hyperparams)
    
    return model

#Example Neural Net random model maker.
def randomseed_ann_maker(seed,input_shape,output_shape,output_activation,loss,optimizer,**kwargs):
    uberhyperparams = {
         'n_hidden':None,             'nl_lo':1, 'nl_hi':50,  #Number of layers
         'nnpl_lo':2,                 'nnpl_hi':20,          #Number of nodes per layer
         'cone':None,                 'P_cone':0.5,
         'activation':None,           'activation_opts':['sigmoid','tanh','relu','selu'],
         'global_activation':None,    'P_global_activation':0.5,
         'initializer':None,          'initializer_opts':['he_normal','he_uniform','glorot_normal','glorot_uniform']}
    uberhyperparams.update(kwargs) #Update defaults with user input.
    uhp = uberhyperparams #abbreviate for sanity.

    #Set random seed.
    np.random.seed(seed)

    hp = {} #Harry Potter. Jk, it's hyperparams

    hp['n_hidden'] = rand_util(uhp['nl_lo'],uhp['nl_hi'],dist='uniform',dtype='int',
                               override=uhp['n_hidden'])

    if uhp['cone'] is None:
        uhp['cone'] = np.random.uniform() < uhp['P_cone']

    if uhp['global_activation'] is None:
        uhp['global_activation'] = np.random.uniform() < uhp['P_global_activation']
    
    if uhp['global_activation']:
        activation = np.random.choice(uhp['activation_opts'])
    
    if uhp['initializer'] is None:
        hp['initializer'] = np.random.choice(uhp['initializer_opts'])
    else:
        hp['initializer'] = uhp['initializer']

    
    model = keras.models.Sequential()
    inp_layer=True
    n_nodes = uhp['nnpl_hi'] 
    for n in range(hp['n_hidden']):
        #Determine number of nodes for this layer.
        if uhp['cone']:
            nnpl_hi = n_nodes
            nnpl_lo = np.max([nnpl_hi - int(np.ceil((n_nodes-uhp['nnpl_lo'])/(hp['n_hidden']-n))) , uhp['nnpl_lo']])
            n_nodes = rand_util(nnpl_lo,nnpl_hi+1,dist='uniform',dtype='int')
        else:
            n_nodes = rand_util(uhp['nnpl_lo'],uhp['nnpl_hi']+1,dist='uniform',dtype='int')

        #Determine activation function for this layer.
        if not uhp['global_activation']:
            activation = np.random.choice(uhp['activation_opts'])
        if inp_layer:
            model.add(keras.layers.Dense(n_nodes,activation=activation,
                                         input_shape=input_shape,kernel_initializer=hp['initializer']))
            inp_layer=False
        else:
            model.add(keras.layers.Dense(n_nodes,activation=activation,kernel_initializer=hp['initializer']))
    model.add(keras.layers.Dense(output_shape,activation=output_activation,kernel_initializer=hp['initializer']))

    model.compile(loss=loss,optimizer=optimizer)
        
    return model

def get_metric(scoring):
    try:
        metric = get_scorer(scoring)
    except (KeyError, ValueError) as e:
        try:
            scorer = get_scorer('neg_'+scoring)
            metric = lambda *args,**kwargs: -scorer(*args,**kwargs)
        except (KeyError, ValueError) as e:
            try:
                scorer = get_scorer(scoring[4:])
                metric = lambda *args,**kwargs: -scorer(*args,**kwargs)
            except KeyError:
                raise ValueError("Unrecognized sklearn metric")
    return metric

def preserve_state(func):
    # Decorator to prevent a function from affecting 
    #  the numpy random seed outside of its scope.
    def wrapper(*args,**kwargs):
        state = np.random.get_state() #Store the random state before function call.
        ret = func(*args,**kwargs)    #Call function.
        np.random.set_state(state)    #Revert numpy to random state from before function call.
        return ret
    return wrapper

class simple_progress:
    def __init__(self):
        self.longest = 0
    def update(self,message):
        print(self.longest*" ",end="\r") #Erase previous message.
        print(message,end="\r")          #Print message
        if len(message) > self.longest:  #Update longest message length.
            self.longest = len(message)


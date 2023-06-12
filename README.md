# Tiny
TinyML innovation toolkit

## Case studies
1. Sea grass (Jamie has trained model this could be used to start quantisation aware training, pruning and deploying) 
2. Water quality (Lorenza has trained model this could be used to start quantisation aware training, pruning and deploying)
    - for 1. and 2. the technical modelling aspects need working on - Ollie + Jack ?
3. New data - Sentinel-2 satellite-derived bathymetry to map the coast of Dominica, before and after Hurricane Maria (27 September 2017): Will need downloading and cleaning - Rhea ?
4. Space weather dataset: This needs to be identified and then downloaded and cleaning. - Thierry ?
5. Codes specific to deploying on TinyML device and research into what devices are best and why (Arduino nano, sony sprescence, edge GPU and TPUs, FPGAs?). - Will ?
6. Digital twins of the arduino

## Data sources:
Sentinel 1
Sentinel 2
Land Sat
...

## Toolkit
Data downloads
Data cleaning
Model generation
Quantization aware training
Model pruning
Model deployment

## READING

Sentinel mission background and caperbilities:
https://sentinel.esa.int/web/sentinel/missions

Sentinel open acess hub:
https://scihub.copernicus.eu

Sentinel 2 data on Google Earth Engine:
https://developers.google.com/earth-engine/datasets/catalog/sentinel-2

Processing needed for Sentinel 2 level 1 to level 2 data:
https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2

ARSET fundamentals of remote sensing: https://appliedsciences.nasa.gov/join-mission/training/english/arset-fundamentals-remote-sensing

Github tutorial: https://docs.github.com/en/get-started/quickstart/hello-world

Github student developer pack - get acess to github copilot: https://education.github.com/pack

# TinyMLToolkit - build stuff, shrink stuff

This (_very_ experimental) package provides a few different classes for general network building and hyperparamer tuning, in the network_builder.py file, along with some functions for shrinking and converting keras models to .tflite files, contained in the network_shrinker.py file. You can pip install this package by running the following command in the terminal:

  
`pip install git+https://github.com/reac2/Tiny`

This will install the package from the github repo. You can then import the package in your python script by running:

`import tinymltoolkit`

This will work for now, as the repo is public. We will have to change this when the repo goes private.

## network_builder.py
The classes in this module try and build a network to solve an arbitrary classification or regression problem, whilst also trying to find the smallest number of parameters. It's a little bit confusing at the moment and I'll try and sort it all out at some point soon, but here is my best explanation as to how it works - The code uses Optuna for hyperparameter tuning (See [the docs](https://optuna.org)). There are three classes at the moment:
1. GeneralNNRegressor
2. GeneralNNClassifier
3. CNNClassifier

Both GeneralNNRegressor and Classifier produce purely feedforward networks, whilst CNNClassifier outputs a pure Convolutional network with no feedforward layers (that needs fixing lol). The classes are all have a similar structure, so we could implement some kind of base class or something. The structure is like this:

```python
class GeneralNN:
    def __init__(self, X, y, SIZE_PENALTY stuff that needs unifying across the classes):
      '''
      Initialise stuff 
      '''
      pass
    def get_best_trained_model(self):
        '''
        Runs the whole process:
        1. Find the best model parameters 
        2. Constructs the best model
        3. Compiles and trains the model
        '''
        pass
    
    def objective(self, trial):
        '''
        Optuna trial objective function
        '''
        pass
      
    def find_best_model_params(self):
        '''
        Finds the set of model parameters that:
        A. Have the best score
        B. Is the smallest
        The paramater SIZE_PENALTY changes how heavily the function weights the size of the network
        '''
        pass

    def make_model(self, network_params):
        '''
        Constructs the keras neural network from the dictionary of network parameters
        '''
        pass
      
    @staticmethod
    def count_trainable_params(model):
        '''
        Counts the trainable parameters in a keras model - dosen't really need to be a staticmethod in each class
        '''
        pass
```
There are quite a lot of things that can be changed in this, but hopefully this allows for an easier way to find a good network that isn't just enormous. If we can reduce the number of parameters in the network before we even start quantising it then that would be great. We might not even need these in the end as all the shrinking stuff works with any trained keras network. 

## network_shrinker.py
This is the code we want to be paying attention to. It 'works' at the moment, but the total size reduction is only about 4X, and I know it's possible to get around 10X. Perhaps someone can look into it a bit more next week. Essentially, I have wrapped all of the code from the [Tensorflow lite for microcontrollers](https://www.tensorflow.org/lite/microcontrollers) webpage. It's not great, so I've created some issues that I know are currently present. However, I'm sure there are more.



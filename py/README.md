# TinyMLToolkit - build stuff, shrink stuff

This (_very_ experimental) package provides a few different classes for general network building and hyperparamer tuning, in the network_builder.py file, along with some functions for shrinking and converting keras models to .tflite files, contained in the network_shrinker.py file. 

## network_builder
The classes in this module try and build a network to solve an arbitrary classification or regression problem, whilst also trying to find the smallest number of parameters. It's a little bit confusing at the moment and I'll try and sort it all out at some point soon, but here is my best explanation as to how it works - 

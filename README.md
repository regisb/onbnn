# Optimal Naive Bayes Nearest Neighbors (oNBNN)

Optimal NBNN (oNBNN) is a C++ library for classification of objects that come
under the form of sets of multi-dimensional features, such as images.

In order to understand how oNBNN works, please refer yourself to the following
paper: "Towards Optimal Naive Bayes Nearest Neighbors", Behmo, Marcombes,
Dalalyan, Prinet, ECCV 2010 available [here] (http://www.minutebutterfly.de/pro).

## Executive summary

    mkdir build && cd build
    cmake ..
    make
    ../examples/example1 ../data/c101_airplanes/ ../data/c101_car_side/ sift

## Package organisation

This package is organised in three main parts: 

* the source code is located in the src/ folder.
* the example tools are located in the examples/ folder
* the image data required to run the example scripts is located in the data/c101... folders.

The source code of optimal-nbnn is itself split in two main parts: 

* onbnn.cpp and onbnn.h contain the code for model training and label
  prediction. The typical use case is binary classification with one or
  multiple channels.
* the lshkit-onbnn/ folder contains our modified version of the LSHKIT library that we employ 
  for nearest neighbor search.

If you intend to use optimal-nbnn with your own work, what you should do is simply to 
link against the dynamic onbnn library (`libonbnn.so`) and to `#include <onbnn/onbnn.h>`.

## Dependencies

The onbnn library requires the following dependencies for compilation:

* Build tools: cmake, g++, automake, etc.
* GNU Linear Programming Kit (GLPK)
* Boost::FileSystem library (for the examples only)
* GNU Scientific Library (for LSHKIT)

Install under ubuntu: 

    sudo apt-get install cmake build-essential libglpk-dev libboost-filesystem-dev libgsl0-dev

Note that if you do not intend to use the multi-probe LSH nearest neighbor
search provided in this package, you do not need the GSL library. Moreover, the
`boost::file_system` library is only required to build the example scripts.

## Build

oNBNN uses a cmake-based build system. This means that you have 
to execute cmake first, which will produce a makefile best suited
to your environment.

    mkdir build && cd build/
    cmake ..
    make
    sudo make install

You can then run the example1 script which classifies livingroom images vs 
bedroom images using both optimal NBNN and normal NBNN (see Examples section below)

## Usage

#### Datasets

For our examples, we included the first 20 images of each class from the
"Fifteen Scene Categories" dataset (available [here] (http://www-cvr.ai.uiuc.edu/ponce_grp/data/ )). 

We resized these images so that they all have identical maximum size (400
pixels). We sampled SIFT features from each image using using van de Sande's
binary utility (available [here] (http://staff.science.uva.nl/~ksande/research/colordescriptors/)).

Provided you download van de Sande's binary utility, you can obtain the same
text files containing the SIFT (or other) features of each image, by running
the ruby scripts located in the data/ folder: 

    ruby sample_features.rb folder1/ sift
    ruby sample_features.rb folder2/ sift
    
These two commands produce text files that contain the image features. These
filenames are of the form: `imagename___sift.txt`

### Examples

#### Example 1: binary, multi-channel classification

Usage:

    example1 ../data/c101_airplanes/ ../data/c101_car_side/ sift

In order to best understand how oNBNN works, it is recommended to take a look at file 
example1-binary-classification.cpp. 

Before we go any further, let us just mention that it is pretty easy for you to
launch the example script on your own image data, by following these steps:

* put your negative images in data/folder1
* put your positive images in data/folder2
* sample just any kind of features using the colorDescriptor binary file by van
de Sande (see above)

Once you have gathered the images and feature files, you are ready to start the example script.

All the ressources of the library are gathered in the onbnn namespace.
onbnn::BinaryClassifier is what you will use to predict the labels of test data. Both testing and training data come under the form of onbnn::Object instances. 

* Training data is added to a classifier through the `add_data()` method. 
* A classifier is trained thanks to the `train()` method.
* Labels of test data are predicted by the `predict()` method.

## Help
  
For further help and information, please contact Régis Behmo
(onbnn@behmo.com), who is the main maintainer of this project.

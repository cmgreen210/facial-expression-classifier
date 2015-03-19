#Facial Expression Classification
Chris Green, March 2015 :smile: :frowning: :open_mouth:

##Contents
* [TL;DR](#tldr)
* [Overview](#overview)
* [Installation](#install)
* [Model](#model)
* [Usage](#usage)

<a name="tldr"/>
##TL;DR

<a name="overview"/>
##Overview

<a name="install"/>
##Installation
First, install [OpenCV][1] and register for [GraphLab][2]. After this is done run the following:
```bash
git clone https://github.com/cmgreen210/facial-expression-classifier
pip install -r requirements
```
If all tests with the below command pass from the top level directory then you are good to go!
```bash
nosetests .
```

<a name="model"/>
##Model
Each model in the package wraps GraphLab's `NeuralNetClassifier` except for the base logistic regression class which uses Scikit-Learn. The easiest way to train a feed forward neural net with this package is to specify an architecture in `net.conf` and then run the `train_nn.sh` script, specifying the appropriate argruments. You don't need to specify the connections between each layer of the net configuration because the training code assumes a linear order between layers.

This package was written to classify human facial expressions. Data for this problem can be found [here][3]. In order to process and pickle the raw data files see code in `gl_data.py`. If you are experimenting it is recommended that you run your classifications with Dato's GraphLab Create™ with GPU Acceleration on a gpu to speed up your training.

<a name="usage"/>
##Usage
###Train
```bash
./train_nn.sh net.conf 12 data/fer_data.pkl cmgreen210@gmail.com cmgreen210-emotions
```

[1]: http://www.opencv.org "OpenCV"
[2]: https://dato.com/products/create/quick-start-guide.html "GraphLab"
[3]: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data "Data"

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
This package allows you to classify facial expressions in both images and video using deep learning. You'll need to have both [OpenCV][1] and [GraphLab][2] installed as well as other standard packages. The goal is to get a model to recognize human expressions. You can then embed this model in whatever system you're using whether that be a smart advertisement platform or a personal robot assistant who wants to know how your day has been. A live version can be found [here][4].

<a name="overview"/>
##Overview
Can a computer recognize how you're feeling? This is the question that this code tries to answer. It turns out...it's a hard problem but not impossible. This repo uses two main libraries. The first is GraphLab Create. This newly deployed python library has a very accessible deep learning package, a version of which is optimized for gpu's that greatly reduces training times. The second package is the widely used open source library OpenCV that is used for face detection, image processing, and interactions with video streams from both files and webcams. The functionality of this code in the static image case is showcased in a live web app [here][4]. In order to get into the nit and gritty of the library see the [usage](#usage) as well as the source code itself.

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
###Data
Scripts for [data][3] processing, saving, and uploading to AWS S3 can be found in `fec/classifier/gl_data.py`.

###Train
Script for running `train_nn.py` and saving all of the data:
```bash
./train_nn.sh net.conf 12 data/fer_data.pkl cmgreen210@gmail.com cmgreen210-emotions
```
###Image
See the image files in the folder `fec/media/` for the source code. Here's an example of how to call the image classification code:
```python
clf = GraphLabClassifierFromFile('/path/to/saved/graphlab/classifier')
face_detector = FaceDetectorProcessor()

image_classifier = ImageFileClassifier(clf.predict_proba, face_detector)

image, gray_scaled, classifications = image_classifier.classify('/path/to/saved/image')
```
###Video
Frames from user provided video can also be analyzed for facial expressions. The video sources can be either from a file or from a live webcam. The source code can be found at `fec/media/video.py`.

[1]: http://www.opencv.org "OpenCV"
[2]: https://dato.com/products/create/quick-start-guide.html "GraphLab"
[3]: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data "Data"
[4]: http://www.fec.space "Live"



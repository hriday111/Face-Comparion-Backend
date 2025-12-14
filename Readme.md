
# Face Comparator Backend

***

This is a program that takes in input of two images and runs a comparator i.e a comparison to find how if two both images have the face of the same person. It also gets the "distance" between two images of the same or different faces. The distance is inversely proportional the the similarity. 

## Roadmap 

1. Fetch a data set of faces
2. Preprocess images - resize, greyscale, or use rgb
3. The core architecture will be a siamese network, which run two models simultaneously 
4. The siamese network runs two CNNs that produce two embedding vectors, using Euclidean distance we can get the similarity. 
5. Testing model 
6. Quality of Life improvements, such as easy usability, documentation, minor improvments. 

***

## Progress

#### Roadmap Step 1 and 2
The program `PreProcesser.py` was made to walk the directly in the `TrainingSet\` folder. It grabs the images, crops to focus on the face and then resizes to a fixed length 105x105 [px]. The preprocessor has options to run with binarization (this mode grayscales and then binarizes the output) or without (results in a rgb image).

The image is then saves and an np array `*.npy` in the same folder where its corresponding image belongs to with the same base name. 

The next program in the pipeline is `GeneratePairs.py`. This program gets all names of people with multiple images of their face then makes equal amounts of positive and negative pairs. Positive pairs have images of the same person with label 1 and negative pairs have images of different people with label 0.  This data is saved as a csv file `TrainingSet\lfw_pairs.csv`

#### Roadmap Step 3, 4, and 5

The program `Trainer.py` is the next in pipeline. It creates and trains a model and saves it locally. This saved model can be later using when running `Test.py` to test two images.

In `Test.py` a function was written that would take any image  and detect the face in it and return a normalized processed image to match the training data. OpenCVs built in Cascade Classifier was used for this function. However it comes with several limitations.

When calling the `detectMultiScale` method, only a frontfacing cascade was used. So if the person isn't facing the camera, the face won't be detected. There was also chances of false positives. Initially I just increased the `minNeighbors` parameter, but that could result in far away faces being missed. So from my reasearch on the internet I found that using the cascase `haarcascade_frontalface_alt_tree.xml` has far more precision, but it could struggle in detecting multiple faces. 

This brings us to the next part. What if multiple faces are detected? Then we just throw a value error and terminate the program. 

#### Next Steps
TODO

* Pass path through arguments
* Restructure code to improve readability
* Create documentation
* General QoL improvements
* Get evaluation (test of a large data set)
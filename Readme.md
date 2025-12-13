
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

The next program in the pipeline is `GeneratePairs.py`. This program gets all names of people with multiple images of their face then makes equal amounts of positive and negative pairs. Positive pairs have images of the same person with label 1 and negative pairs have images of different people with label 0.  This data is saved as a csv file `lfw_pairs.csv`



# Face Comparator Backend

***

This is a program that takes in input of two images and runs a comparator i.e a comparison to find how if two both images have the face of the same person. It also gets the "distance" between two images of the same or different faces. The distance is inversely proportional the the similarity. 

## Roadmap - Training

1. Fetch a data set of faces
2. Preprocess images - resize, greyscale
3. The core architecture will be a siamese network, which run two models simultaneously 
4. The siamese network runs two CNNs that produce two embedding vectors, using Euclidean distance we can get the similarity. 
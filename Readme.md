
# Face Comparator Backend

***

This is a program that takes in input of two images and runs a comparator i.e a comparison to find how if two both images have the face of the same person. It also gets the "distance" between two images of the same or different faces. The distance is inversely proportional the the similarity. 

**Go to [Pipeline to Run](#pipeline-to-run) to run the program**
## Roadmap 

1. Fetch a data set of faces
2. Preprocess images - resize, greyscale, or use rgb
3. The core architecture will be a siamese network, which run two models simultaneously 
4. The siamese network runs two CNNs that produce two embedding vectors, using Euclidean distance we can get the similarity. 
5. Testing model 
6. Quality of Life improvements, such as easy usability, documentation, minor improvments. 

***

## Progress

### Roadmap Step 1 and 2

The raw training data was used from https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

The program `PreProcesser.py` was made to walk the directly in the `TrainingSet\` folder. It grabs the images, crops to focus on the face and then resizes to a fixed length 105x105 [px]. The preprocessor has options to run with binarization (this mode grayscales and then binarizes the output) or without (results in a rgb image).

The image is then saves and an np array `*.npy` in the same folder where its corresponding image belongs to with the same base name. 

The next program in the pipeline is `GeneratePairs.py`. This program gets all names of people with multiple images of their face then makes equal amounts of positive and negative pairs. Positive pairs have images of the same person with label 1 and negative pairs have images of different people with label 0.  This data is saved as a csv file `TrainingSet\lfw_pairs.csv`

### Roadmap Step 3, 4, and 5

The program `Trainer.py` is the next in pipeline. It creates and trains a model and saves it locally. This saved model can be later using when running `Test.py` to test two images.

In `Test.py` a function was written that would take any image  and detect the face in it and return a normalized processed image to match the training data. OpenCVs built in Cascade Classifier was used for this function. However it comes with several limitations.

When calling the `detectMultiScale` method, only a frontfacing cascade was used. So if the person isn't facing the camera, the face won't be detected. There was also chances of false positives. Initially I just increased the `minNeighbors` parameter, but that could result in far away faces being missed. So from my reasearch on the internet I found that using the cascase `haarcascade_frontalface_alt_tree.xml` has far more precision, but it could struggle in detecting multiple faces. 

This brings us to the next part. What if multiple faces are detected? Then we just throw a value error and terminate the program. 


### Evaluation of the Model
Since I used all the training data for training I needed another set of faces that I could run an evaluation script on my model. 
The dataset I used was https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset

Two metrics were calculated: Equal Error rate, which is how often the model if confused equally between false positive and false negatives. And accuracy which measured how often the model was right.

Using the `GeneratePairs.py` program the csv for Test data was calculated. 

It must be noted that the Testing dataset had two collections. One with Raw Faces. Pictures of images of people. Something that you would take off a camera. A mix of full body pictures, half body, zoomed in, zoomed out. Just natural photos. 
The other collection is or the same raw images, but the faces extracted. So the images are cropped up to the faces only, that we can just throw in the model and test. 

First I performed the evalution metrics on the extracted faces:
```
--- METRICS REPORT ---
Equal Error Rate (EER): 0.4804 (48.04%)
Threshold at EER:       0.5621
Max Accuracy:           0.5332 (53.32%)
Best Accuracy Threshold:0.3930
```
The best accuracy threshold being 0.39 that means that if the distance is less than 0.39 the faces are different. This isn't ideal since the threshold should be somewhere in the middle. The `EER=0.48` and `Max Accuracy=0.53`is also very bad. 

Hence the next step was to get the evaluation metrics when the test is performed on real life images. Here are their metrics:
```
--- METRICS REPORT ---
Equal Error Rate (EER): 0.3470 (34.70%)
Threshold at EER:       0.5452
Max Accuracy:           0.6590 (65.90%)
Best Accuracy Threshold:0.5230
```
Here the threshold is pretty acceptable, however the ERR and accuracy indicate that atleast 1 in 3 images will give the wrong output. 

### Applying Improvements

3 Major improvements were applied. The first was introducing callbacks and validity functions. About 10% of the data from the training set was copied manually to another csv names `TrainingSet\val.csv` and this was used in the call backs. The number of Epochs were also increased to 30 and during each cycle of training a dropout layer was introduced to force the model to not overfit and learn more details. 

The following callbacks were implemented
- Callback to stop if the learning rate is very low was also implemented as to not waste any time.
- Callback to reduce learning rate if the validation loss doesn't improve
- Callback to create a checkpoint if the validation loss improves.

The results after these improvements were the following:
'''
Equal Error Rate (EER): 0.3262 (32.62%)
Threshold at EER:       0.3482
Max Accuracy:           0.6829 (68.29%)
Best Accuracy Threshold:0.3146
'''

This shows an improved accuracy of 68% and gives as a Threshold value to use. Futhermore, experimentation were done with the built-in cascades from opencv for frontalface detection. This brought up the accuracy by upto 69%, but also ever so slightly changed the threshold. 
'''
--- METRICS REPORT ---
Equal Error Rate (EER): 0.3201 (32.01%)
Threshold at EER:       0.3505
Max Accuracy:           0.6912 (69.12%)
Best Accuracy Threshold:0.3309
'''

The program `main.py` can be run where the threshold is set, on any two images. 
You can run it in 3 different modes
```
python main.py -same
python main.py -different
python main.py -custom path_to_img1 path_to_img2
```
There are 3 hardcoded images in `main.py`, 2 of which are of the same person. Using the arguments -same and -different will run a test on them, and -custom runs a test on custom images.

**It is important that the custom image you supply contains only one person**


***

## Pipeline to Run

**I highly recomment to not compute the preprocessing and pair generating and test out the `main.py` directly**
**Skip straight to step 3, run `Trainer.py` and then `main.py` with the above mentioned arguments**
1. First the PreProcessor is run on Training set and Validation set. Simply run `python PreProcessor.py -training` or `-validation`. 
2. Next to generate positive and negative pair CSVs run `python GeneratePairs.py`. This will generate pairs for training and validation sets. This also automatically creates a `TrainingSet\val.csv` validation file explicitly used for training
3. Run `python Trainer.py`
4. Run `main.py` with with above mentioned arguments. 
5. To test the validation metrics, you can run `python RunValidation.py`.

***

## Known Limitations

The biggest Limitation to this model, espesially where it bottlenecks is the OpenCV face classification. 
During every Validation run, a pair of images was skipped if either of the images detected 0 faces or more then 1 face. Even changing parameters when calling the methods for face detection and using different cascades, across every run around 80 image pairs were ignored out of approx 840 pairs. Hence this could be one bottleneck and hence affecting the accuracy score.

Another limitation is the use of this CNN model that was trained on a relatively small dataset and had custom CONV2D layers. Using pre-trained models like MobileNetV2 which has been trained on millions of images and replacing the final classification layer with our custom layer with our data. Although I am not certain this would improve my accuracy but this is what my research showed me.


## Future Improvements

The `main.py` along with the pretrained model file `.keras` can be contanerized running as an http server. Where the image paths are web urls and not file paths. Then it can be run as a REST api, and a frontend can be build to interactively use this program. 

I would also like to look into the benefits of converting this model into `frugally-deep` or `tensorflow-lite` to do the prediction in C++ which removes the python interpreter overhead
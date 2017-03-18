# Project summary

## Files

My project includes the following files:
* model.py.ipynb **This is a jupyter notebook** At the end of the notebook.
* drive.py 
* model.h5 
* writeup_report.md 
* video.mp4

## Submission

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The final video is also included (video.mp4) as well as on youtube: 

https://www.youtube.com/watch?v=3qJBjIvFs9k

I've also written a medium article about the intermediate layers:

https://medium.com/@tempflip/how-robotic-cars-see-the-world-6af0808451fa#.ub60oe7er


## Training data

I'm not using the data provided by udacity, but collected it myself. The reaseon for doing this I wanted to see the difference between trainig on different about and quality of data. I'm both using track 1 and track 2 data.

I'm loading all the data into a pandas DataFrame. I'm throwing away the points which has a steering angle less than +-0.2 . The reason for this that the points which has a 0 steering angle can't improve the model at all; and I've noticed that if I train only on relatively higher steering angles, I can better results. I'm using a generator function for loading the images, as I wanted to create a scaleble way to load images into the memory.

## Model Architecture

My model is very simple:

```
def crop(d):
    return d[:,80:120,:,:]

def normalize(d):
    return d / 255 - 0.5

def reduce_palette(d):
    import tensorflow as tf 
    n = 40
    return tf.ceil((d/n)*n)

model = Sequential()

model.add(Lambda(reduce_palette, input_shape=(160, 320, 3)))
model.add(Lambda(normalize))
model.add(Lambda(crop))

model.add(Convolution2D(9,3,3, border_mode="same", name="conv1"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.1))
model.add(Activation('relu', name="relu1"))

model.add(Convolution2D(18,3,3, border_mode="same", name="conv2"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.1))
model.add(Activation('relu', name="relu2"))

model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
```

I'm starting with some pre-processing steps (cropping, normalizing and reducing palette.) I've noticed that if I reduce palette my model gets better results -- this is probably because it creates a more generic understanding of the input image. Also, a smaller croppend window looks better that a larger one.

After pre-processing I'm adding to convolutional layers, both with 3x3 windows, a 2x2 pooling layers and a RELU activation function.
Please see my reflections about the inside data here:

https://medium.com/@tempflip/how-robotic-cars-see-the-world-6af0808451fa#.ub60oe7er

After the convolutinal layers I'm adding 2 fully connected layers, and the final result is a 1x1 matric (the steering angle.)

I've tryed to add more convolutional layers, but they did not improve the performance -- it looks like after 2 convolutions the network can get enough data to create a generic understanding of the camere positions compared to the road edges.

Also, adding more fully connected layers did not improve much the performance.

## Training Strategy

## Layer visualizations


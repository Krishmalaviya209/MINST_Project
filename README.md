# MINST_Project

Importing Necessary Libraries :
- TensorFlow and its Keras API are imported for building and training the neural network.
- Numpy is used for numerical operations.
- PIL (Python Imaging Library) is imported for image-related tasks.


Load the MNIST Dataset:
- The code loads the MNIST dataset from the library, which consists of a large collection of 28x28 pixel grayscale images of handwritten digits (0 to 9) for both training and testing.

Resource : Link to the dataset - https://www.tensorflow.org/datasets/catalog/mnist

Preprocess the Data:
* The images in the dataset are preprocessed as follows:
  - Reshaped to (28, 28, 1) to match the expected input shape for the neural network.
  - Normalized by dividing the pixel values by 255 to scale them to the range [0, 1].
  - The labels are one-hot encoded using to_categorical to convert them into a binary matrix representation.
 
CNN Model:
* The CNN model is constructed using Keras Sequential API, which allows for the sequential stacking of layers.
* The CNN architecture consists of several layers:
    - Convolutional Layer 1: It has 32 filters with a 3x3 kernel, ReLU activation, and input shape (28, 28, 1).
    - MaxPooling Layer 1: Reduces the spatial dimensions of the feature maps.
    - Convolutional Layer 2: 64 filters with a 3x3 kernel and ReLU activation.
    - MaxPooling Layer 2: Further reduces spatial dimensions.
    - Convolutional Layer 3: Another 64 filters with a 3x3 kernel and ReLU activation.
    - Flattening Layer: Converts the 2D feature maps into a 1D vector.
    - Fully Connected (Dense) Layer 1: 64 units with ReLU activation.
    - Fully Connected Layer 2: 10 units with softmax activation, representing the 10 possible digit classes.

      CNN Representation :
      
Input (28x28x1)
|
Conv2D (32, 3x3, relu)
|
MaxPooling2D (2x2)
|
Conv2D (64, 3x3, relu)
|
MaxPooling2D (2x2)
|
Conv2D (64, 3x3, relu)
|
Flatten
|
Dense (64, relu)
|
Dense (10, softmax)
|
Output

Compile the Model:
* The model is compiled with the following settings:
    - Optimizer: 'adam' - A popular optimization algorithm.
    - Loss Function: 'categorical_crossentropy' - Appropriate for multiclass classification problems.
    - Metrics: 'accuracy' - To monitor the accuracy during training.

Train the Model:
- The model is trained using the training data (images and one-hot encoded labels).
- The training process consists of 5 epochs, meaning the model will make five passes through the entire training dataset.
- Batch size is set to 64, which means the model's weights are updated after processing each batch of 64 training examples.




Link to amazing interactive MINST model : https://transcranial.github.io/keras-js/#/mnist-cnn


  

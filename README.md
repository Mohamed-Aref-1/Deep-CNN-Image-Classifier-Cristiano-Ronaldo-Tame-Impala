# Deep-CNN-Image-Classifier-Cristiano-Ronaldo-Tame-Impala
 The Deep-CNN-Image-Classifier "Cristiano Ronaldo Tame Impala" project demonstrates the successful implementation of a CNN-based image classifier using TensorFlow. The model achieved an impressive accuracy rate, reaching 99% accuracy on both training and validation data in the final epoch. 


The Deep-CNN-Image-Classifier "Cristiano Ronaldo Tame Impala" is a project that focuses on using deep learning techniques to classify images of Cristiano Ronaldo and Tame Impala. The project utilizes the TensorFlow framework, specifically TensorFlow-GPU, along with other libraries such as OpenCV and Matplotlib. By employing Convolutional Neural Networks (CNNs), the project aims to achieve high accuracy in distinguishing between images of Cristiano Ronaldo and Tame Impala.

Methodology:
The project uses the following components and techniques to build the image classifier:

TensorFlow: The deep learning framework TensorFlow is employed for creating and training the CNN-based image classifier. It provides the necessary tools and functions to define, compile, and train the model.

OpenCV and cv2: OpenCV is utilized for image preprocessing and manipulation tasks, while cv2 provides the Python bindings for the OpenCV library. These libraries enable tasks such as image loading, resizing, and format handling.

Matplotlib: Matplotlib is used for visualizing the training process and displaying the accuracy and loss metrics of each epoch.

Neural Network Architecture: The CNN model is built using the Sequential API from Keras, a high-level API of TensorFlow. The architecture consists of multiple convolutional layers with pooling operations to extract meaningful features from the input images. The final layers include dense (fully connected) layers responsible for classification.

Model Architecture:
The CNN model architecture used in the project is as follows:
Conv2D: The initial layer has 16 filters with a 3x3 kernel size and a stride of 1. It uses the ReLU activation function and accepts input images of size 256x256x3.

MaxPooling2D: After each convolutional layer, a max pooling operation is applied to downsample the feature maps.

Additional Convolutional and MaxPooling Layers: The model includes additional layers with 32 filters and 16 filters, respectively, followed by max pooling operations.

Flatten: The output of the last convolutional layer is flattened into a 1D tensor to be fed into the dense layers.

Dense: The flattened tensor is passed through two dense layers with 256 units and ReLU activation, and a final dense layer with 1 unit and sigmoid activation for binary classification.

Model Compilation: The model is compiled using the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric.

Training Results:
The model is trained for 30 epochs, with the following accuracy and loss metrics:
Epoch 1/30: Loss: 0.7510 | Accuracy: 0.5078 | Validation Loss: 0.6467 | Validation Accuracy: 0.8125
Epoch 2/30: Loss: 0.6714 | Accuracy: 0.5312 | Validation Loss: 0.6651 | Validation Accuracy: 0.5625
...
Epoch 30/30: Loss: 0.0226 | Accuracy: 1.0000 | Validation Loss: 0.0093 | Validation Accuracy: 1.0000

The accuracy steadily improves over the epochs, reaching 100% accuracy on both training and validation data in the final epoch.

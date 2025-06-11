# CNN Image Classification on CIFAR-100 with Data Augmentation

This project demonstrates the implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-100 dataset. The primary goal is to build a robust classifier and evaluate the impact of **data augmentation** on model performance and generalization.

The notebook, `bomba.ipynb`, contains two key experiments:
1.  Training a CNN on the original CIFAR-100 dataset.
2.  Training the same CNN architecture on an augmented version of the dataset to improve robustness.

## Methodology

The project follows these key steps:

1.  **Dataset**:
    *   The **CIFAR-100 dataset** is used, which consists of 60,000 32x32 color images in 100 classes, with 600 images per class.
    *   The dataset is split into 50,000 training images and 10,000 test images.
    *   Pixel values are normalized to a range of [0, 1].

2.  **Data Augmentation**:
    *   To prevent overfitting and help the model generalize better, `ImageDataGenerator` is used to create modified versions of the training images on-the-fly.
    *   The augmentations include:
        *   Random rotations (up to 20 degrees)
        *   Width and height shifts
        *   Horizontal flips

3.  **Model Architecture**:
    *   A standard Convolutional Neural Network (CNN) is built using a `Sequential` model in Keras.
    *   The architecture consists of:
        *   Multiple `Conv2D` layers with 'relu' activation for feature extraction.
        *   `MaxPooling2D` layers to downsample the feature maps.
        *   A `Flatten` layer to prepare the data for the fully connected layers.
        *   `Dense` layers for classification, including a `Dropout` layer in the augmented model to further combat overfitting.
        *   The final output layer has 100 neurons corresponding to the 100 CIFAR classes.

4.  **Training and Evaluation**:
    *   The model is compiled with the `adam` optimizer and `SparseCategoricalCrossentropy` loss function.
    *   Two separate training sessions are conducted:
        1.  **With Data Augmentation**: The model is trained for 50 epochs using the augmented data stream.
        2.  **Without Data Augmentation**: The model is trained for 10 epochs on the original, normalized data.
    *   Training and validation accuracy are plotted to visualize the learning process.

## Results & Observations

The project compares the performance of the model under two different training conditions:

*   **Without Data Augmentation (10 epochs):**
    *   **Test Accuracy:** ≈ 35.6%
*   **With Data Augmentation (50 epochs):**
    *   **Test Accuracy:** ≈ 32.3%

Interestingly, in this specific setup, the model trained without augmentation achieved a slightly higher accuracy in fewer epochs. This could be due to several factors:
*   The augmentation parameters might be too aggressive for the dataset.
*   The model architecture may need more complexity or longer training to fully benefit from the augmented data.
*   The non-augmented model, despite higher accuracy, might be more prone to overfitting on unseen data, whereas the augmented model's learning curve suggests a more stable, generalizable training process.

This experiment highlights that data augmentation is a powerful technique but requires careful tuning of its parameters and training duration to achieve optimal results.

## How to Run

1.  Clone this repository to your local machine.
2.  Ensure you have the required Python libraries installed. You can use the `pip install` commands at the top of the notebook or run:
    ```bash
    pip install tensorflow pandas matplotlib
    ```
3.  Open and run the cells in the `bomba.ipynb` Jupyter Notebook. The notebook will automatically download the CIFAR-100 dataset.

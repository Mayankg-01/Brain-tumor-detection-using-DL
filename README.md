Brain Tumor Detection using Deep Learning
Using Python, TensorFlow, and Keras to build a Convolutional Neural Network (CNN) that classifies brain MRI scans for the presence of tumors. The model is trained on the Brain Tumor MRI Dataset from Kaggle.

Check:

Jupyter Notebook: brain_tumour_detection_using_deep_learning.ipynb

Dataset: Brain Tumor MRI Dataset on Kaggle

Project Workflow
First, we start by preparing our data:

Loading MRI images from their respective directories (yes/no).

Preprocessing images by resizing them to a uniform size.

Applying data augmentation to increase the diversity of the training set and reduce overfitting.

Then we build and train our model:

Construct a Sequential CNN model with convolutional, pooling, and dense layers.

Compile the model using an appropriate optimizer and loss function.

Train the model on the prepared dataset, validating its performance on a separate test set.

Finally, we evaluate the model and make predictions:

Visualize the training history (accuracy and loss) to assess learning.

Evaluate the final model accuracy on the test data.

Use the trained model to predict whether new, unseen MRI images contain tumors.

Key Objective
The primary goal is to answer the following question:

Can we build a deep learning model to accurately and automatically classify brain MRI scans as either containing a tumor or not?

To answer this, we walk through several key deep learning and data processing methods:

Building a CNN architecture from scratch.

Using ImageDataGenerator for preprocessing and data augmentation.

Splitting data for training and testing.

Compiling, fitting, and evaluating a Keras model.

Visualizing results with Matplotlib.

Setup and Usage
Prerequisites: Python 3.7+, pip, and Jupyter Notebook or Google Colab (recommended).

Clone & Install Dependencies:

git clone [https://github.com/your-username/brain-tumor-detection.git](https://github.com/your-username/brain-tumor-detection.git)
cd brain-tumor-detection
pip install -r requirements.txt

Prepare Dataset:

Download the Brain Tumor MRI Dataset.

Upload the dataset zip file to a folder in your Google Drive.

Run in Colab:

Upload the .ipynb file to Google Colab.

Mount your Google Drive and update the file path in the notebook to point to your dataset zip file.

Execute the cells sequentially.

Results
The notebook visualizes the training/validation accuracy and loss. It also reports the final model accuracy on the test set and shows prediction examples on sample images.

Contributing
Contributions are welcome. Please fork the repository, create a new branch for your feature, and submit a pull request.

License
This project is licensed under the MIT License.

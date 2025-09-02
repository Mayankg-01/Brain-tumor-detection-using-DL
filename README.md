# Brain Tumor Detection using Deep Learning

## 📌 Project Overview

-   Detects brain tumors from MRI images using **Deep Learning**.\
-   Utilizes **Transfer Learning with VGG16**.\
-   Classifies MRI scans into **4 categories**:
    -   Glioma\
    -   Meningioma\
    -   Pituitary\
    -   No Tumor

------------------------------------------------------------------------

## ⚙️ Requirements

-   Python 3.x\
-   Libraries:
    -   TensorFlow / Keras\
    -   NumPy\
    -   Matplotlib\
    -   Seaborn\
    -   scikit-learn\
    -   PIL (Pillow)

------------------------------------------------------------------------

## 📂 Dataset

-   Dataset structure:
    -   `/Training/` → training images\
    -   `/Testing/` → testing images\
-   Image size: **128 × 128 pixels**\
-   Loaded using preprocessing utilities.

------------------------------------------------------------------------

## 🧠 Model Architecture

-   Base model: **VGG16** (pre-trained on ImageNet).\
-   Custom layers added on top:
    -   Flatten\
    -   Dropout (0.3, then 0.2)\
    -   Dense (128 neurons, ReLU)\
    -   Output Dense (Softmax, 4 classes)\
-   **Last 3 VGG16 layers unfrozen** for fine-tuning.

------------------------------------------------------------------------

## 🏋️ Training Setup

-   Image size: **128 × 128**\
-   Batch size: **20**\
-   Epochs: **5**\
-   Optimizer: **Adam**\
-   Learning Rate: **0.0001**\
-   Loss: **Sparse Categorical Crossentropy**

------------------------------------------------------------------------

## 📊 Evaluation

-   Metrics used:
    -   Accuracy\
    -   Classification report\
    -   Confusion matrix\
    -   ROC curve\
-   Visualization of training/validation loss and accuracy included.

------------------------------------------------------------------------

## 💾 Saving & Loading

-   Trained model saved as:

    ``` bash
    model.h5
    ```

-   Can be reloaded using:

    ``` python
    from tensorflow.keras.models import load_model
    model = load_model("model.h5")
    ```

------------------------------------------------------------------------

## 🚀 Usage

1.  Clone this repository

    ``` bash
    git clone <repo_url>
    cd brain_tumour_detection
    ```

2.  Install dependencies

    ``` bash
    pip install -r requirements.txt
    ```

3.  Place dataset in the proper structure (`Training/`, `Testing/`).\

4.  Run Jupyter Notebook

    ``` bash
    jupyter notebook brain_tumour_detection_using_deep_learning.ipynb
    ```

5.  Train & evaluate the model.

------------------------------------------------------------------------

## 📈 Results

-   Successfully detects tumors into 4 classes.\
-   Achieved high accuracy on test data (see notebook plots).

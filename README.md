#  Brain Tumor Classification using CNN

This project detects brain tumors from MRI images using a Convolutional Neural Network (CNN). It classifies images into four categories: **Glioma, Meningioma, Pituitary Tumor, and Normal**.

##  Project Structure

- `CNN.ipynb`: Jupyter Notebook for training the CNN model.
- `app.py`: Streamlit web application for deployment.
- `requirements.txt`: List of dependencies.
- `cnn_model.pth`: Trained model weights (Generated after running `CNN_fixed.ipynb`).

##  Installation

1.  **Clone the repository** (or download files):
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2.  **Install Dependencies**:
    Make sure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

##  Usage

### 1. Train the Model
Run the Jupyter Notebook to download the dataset, train the model, and save `cnn_model.pth`.
- Open `CNN.ipynb` in Jupyter Notebook or Google Colab.
- Run all cells.
- The model will be saved as `cnn_model.pth`.

### 2. Run the Web App
Once the model is trained, start the Streamlit app:
```bash
streamlit run app.py
```
- A new tab will open in your browser.
- Upload an MRI image to get a prediction.

##  Model Details
- **Architecture**: Custom CNN with 3 Convolutional Blocks.
- **Input Size**: 128x128 pixels.
- **Classes**: Glioma Tumor, Meningioma Tumor, Normal, Pituitary Tumor.

## Credits
- Dataset: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

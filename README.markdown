# MarineVision-CBANet

## Description
MariNet is a deep learning project for classifying marine life species using a convolutional neural network. It leverages the EfficientNetB3 model enhanced with a Convolutional Block Attention Module (CBAM) to achieve high accuracy in identifying marine species from images. The project uses TensorFlow and is designed to work with the Marine Life Classification Dataset.

## Installation
To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kushalava917/MariNET.git
   cd MariNet
   ```

2. **Install Python dependencies**:
   Ensure you have Python 3.7+ installed. Install the required packages using pip:
   ```bash
   pip install tensorflow kagglehub pillow numpy
   ```

3. **Download the dataset**:
   The project uses the Marine Life Classification Dataset from Kaggle. Ensure you have a Kaggle account and API key set up:
   - Install the Kaggle CLI: `pip install kaggle`
   - Set up your Kaggle API key: `kaggle config path`
   - The dataset is automatically downloaded via `kagglehub` in the script, but verify access:
     ```bash
     kaggle datasets download -d markyousri/marine-life-classification-dataset
     ```

4. **Verify TensorFlow GPU support (optional)**:
   If using a GPU, ensure CUDA and cuDNN are installed for TensorFlow. Check TensorFlow version:
   ```python
   import tensorflow as tf
   print(tf.__version__)
   ```

## Usage
To use the project for training the model or predicting marine species from images, follow these examples:

1. **Train the model**:
   Run the Jupyter notebook `DL_Project.ipynb` to download the dataset, preprocess data, train the model, and save it:
   ```bash
   jupyter notebook DL_Project.ipynb
   ```
   Alternatively, convert the notebook to a Python script and run:
   ```bash
   jupyter nbconvert --to script DL_Project.ipynb
   python DL_Project.py
   ```

2. **Predict marine species from an image**:
   Use the saved model to classify a marine species image. Example:
   ```python
   from tensorflow.keras.models import load_model
   from PIL import Image
   import numpy as np
   import tensorflow as tf

   # Load the model
   model = load_model("marine_life_species_classifier.keras")

   # Define image size and class names (adjust based on your dataset)
   img_size = (300, 300)
   class_names = ["class1", "class2", "class3", "class4", "class5"]  # Replace with actual class names

   def predict_image(img_path):
       img = Image.open(img_path).convert("RGB").resize(img_size)
       img_array = tf.keras.preprocessing.image.img_to_array(img)
       img_array = tf.expand_dims(img_array, 0)
       img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
       predictions = model.predict(img_array)
       class_idx = np.argmax(predictions[0])
       class_name = class_names[class_idx]
       confidence = predictions[0][class_idx]
       return f"Predicted class: {class_name} with confidence: {confidence:.2f}"

   # Example usage
   print(predict_image("path/to/your/image.jpg"))
   ```

3. **Upload images in Google Colab**:
   If running in Colab, upload an image and predict:
   ```python
   from google.colab import files
   uploaded = files.upload()
   for filename in uploaded.keys():
       print(predict_image(filename))
   ```

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request with a detailed description of your changes.

Please ensure your code follows PEP 8 style guidelines and includes relevant tests.

## License
This project is licensed under the MIT License. See the [[LICENSE](LICENSE)](https://github.com/Kushalava917/MariNET/blob/df6f610fdaa9a73d9c90c0ded6f8c60512df952a/LICENSE) file for details.

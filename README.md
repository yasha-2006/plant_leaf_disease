# 🌿 Plant Leaf Disease Detection System

A deep learning-based **Plant Leaf Disease Detection System** that identifies plant diseases from leaf images using a trained **TensorFlow/Keras image classification model**.

The application provides an easy-to-use **Streamlit web interface** where users can upload a plant leaf image and receive a predicted disease category.

---

## 🚀 Project Overview

Plant diseases can significantly affect crop productivity and agricultural sustainability. Early identification of diseases can help farmers and agricultural professionals take appropriate preventive measures.

This project uses **Deep Learning and Computer Vision** to classify plant leaf images into different categories.

The current application supports the following predictions:

* 🍂 **Early Blight**
* 🌱 **Healthy**
* 🍁 **Late Blight**

The trained model is automatically downloaded from Google Drive when the application starts if the model file is not already available locally.

---

## ✨ Features

* 🌿 Plant leaf disease classification
* 📷 Upload leaf images directly through the web interface
* 🤖 TensorFlow/Keras deep learning model
* ⚡ Automatic trained-model download
* 🖥️ Interactive Streamlit interface
* 🔍 Image preprocessing and prediction
* 📊 Supports three disease classes
* 🌾 Designed with sustainable agriculture in mind

---

## 🧠 Technologies Used

| Technology       | Purpose                                    |
| ---------------- | ------------------------------------------ |
| **Python**       | Core programming language                  |
| **TensorFlow**   | Deep learning framework                    |
| **Keras**        | Neural network model loading and inference |
| **Streamlit**    | Web application interface                  |
| **NumPy**        | Numerical and image-array processing       |
| **Pillow**       | Image loading and processing               |
| **gdown**        | Downloading the trained model              |
| **Git & GitHub** | Version control and project hosting        |

---

## 🔬 How the System Works

The application follows this workflow:

```text
          ┌──────────────────────┐
          │   User uploads       │
          │   leaf image         │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   Image Loading      │
          │   & Preprocessing    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ TensorFlow / Keras   │
          │ Trained Model        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   Classification     │
          │    Prediction        │
          └──────────┬───────────┘
                     │
                     ▼
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Early Blight   Healthy     Late Blight
```

---

## 📸 Application Workflow

### 1. Open the Application

Launch the Streamlit application and select the **Disease Recognition** option from the sidebar.

### 2. Upload an Image

Upload an image of a plant leaf using the file uploader.

### 3. Preview the Image

The application displays the uploaded image for inspection.

### 4. Predict the Disease

Click the **Predict** button.

The trained TensorFlow/Keras model processes the image and returns the predicted class.

### 5. View the Result

The application displays the predicted disease category.

---

## 🏗️ Project Structure

```text
plant_leaf_disease/
│
├── .devcontainer/
│
├── Disease.png
│       # Application/interface image
│
├── app.py
│       # Main Streamlit application
│
├── requirements.txt
│       # Python dependencies
│
└── README.md
        # Project documentation
```

The trained model is downloaded automatically by `app.py` when it is not present locally.

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yasha-2006/plant_leaf_disease.git
```

Move into the project directory:

```bash
cd plant_leaf_disease
```

---

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The project requirements include packages such as:

* TensorFlow
* Keras
* Streamlit
* NumPy
* Pillow
* gdown
* Pandas
* Matplotlib
* Seaborn

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The terminal will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 🖥️ Application Interface

The application contains two main sections:

### 🏠 Home

The home page introduces the **Plant Disease Detection System for Sustainable Agriculture**.

### 🔍 Disease Recognition

The disease recognition page allows users to:

1. Upload a leaf image
2. Preview the image
3. Run the prediction
4. View the predicted disease

---

## 🤖 Machine Learning Model

The application uses a trained **TensorFlow/Keras image classification model**.

Before prediction, the uploaded image is resized to:

```text
128 × 128 pixels
```

The image is then converted into a numerical array and passed to the trained neural network.

The model returns prediction probabilities for the supported classes. The class with the highest predicted probability is selected as the final prediction.

### Supported Classes

```text
0 → Early_Blight
1 → Healthy
2 → Late_Blight
```

---

## ☁️ Model Download

The trained model is not stored directly inside the GitHub repository.

Instead, the application checks whether the model file exists locally:

```text
trained_plant_disease_model.keras
```

If it does not exist, the application downloads the trained model from Google Drive using `gdown`.

This keeps the large model file outside the Git repository.

---

## 📊 Prediction Pipeline

```text
Input Leaf Image
       ↓
Image Upload
       ↓
Resize to 128 × 128
       ↓
Convert Image → NumPy Array
       ↓
TensorFlow/Keras Model
       ↓
Prediction Probabilities
       ↓
Highest Probability Class
       ↓
Disease Classification
```

---

## 🌱 Sustainable Agriculture

This project demonstrates how Artificial Intelligence can support sustainable agriculture by assisting with early plant disease identification.

Potential applications include:

* 🌾 Crop monitoring
* 🔍 Early disease identification
* 👨‍🌾 Agricultural decision support
* 📱 AI-assisted farming applications
* 🌱 Sustainable crop management
* 🚜 Smart agriculture systems

The prediction should be treated as an **AI-based classification aid**, not as a replacement for professional agricultural diagnosis.

---

## 🔮 Future Enhancements

The project can be further improved by adding:

* 📱 Mobile-friendly deployment
* 🌐 Cloud deployment
* 📈 Confidence scores for predictions
* 🧠 Additional plant disease classes
* 🌿 Support for multiple crop species
* 📷 Camera-based real-time detection
* 📊 Disease statistics dashboard
* 💡 Disease treatment/recommendation information
* 🗺️ Geographical crop-disease monitoring
* 🤖 Integration with smart farming systems

---

## 🛠️ Possible Improvements to the Model

Future versions could explore:

* Transfer learning using **MobileNet**
* **ResNet**
* **EfficientNet**
* Data augmentation
* Hyperparameter optimization
* Larger and more diverse datasets
* Cross-validation
* Confusion matrix analysis
* Precision, recall and F1-score evaluation

---

## 📚 Learning Outcomes

Through this project, the following concepts can be explored:

* Python programming
* Image classification
* Computer Vision
* Deep Learning
* TensorFlow/Keras
* Neural Networks
* Image preprocessing
* Streamlit application development
* Model deployment
* AI applications in agriculture

---

## 📦 Requirements

All required Python packages are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🔗 Repository

**GitHub Repository:**

`https://github.com/yasha-2006/plant_leaf_disease`

---

## 👩‍💻 Author

**Yashashree M**

B.E. – Artificial Intelligence and Machine Learning

---

## ⭐ Acknowledgement

This project was developed as an educational implementation demonstrating the application of **Deep Learning and Computer Vision in agriculture**.

---

## 📄 License

This project is intended for **educational and learning purposes**. Please review the licenses and terms associated with any datasets, trained models, or external resources used with the project.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub!

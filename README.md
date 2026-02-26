# 🧠 Handwritten Digit Recognition using Deep Learning

This project implements a simple and effective **Handwritten Digit Recognition System** using Deep Learning with TensorFlow/Keras.  
It predicts handwritten digits (0–9) from grayscale images based on patterns learned from training data.

## 📁 Dataset

This project uses the MNIST handwritten digits dataset.

- 60,000 training images  
- 10,000 testing images  
- Image size: 28 × 28 pixels  
- Grayscale images  
- 10 output classes (digits 0–9)


## 📊 Project Workflow

### 1️⃣ Data Preprocessing

- Loaded dataset using TensorFlow  
- Normalized pixel values  
- Split into training and testing sets  

### 2️⃣ Model Architecture

- Flatten Layer (Input: 28×28)  
- Dense Layer (128 neurons, ReLU activation)  
- Dense Layer (128 neurons, ReLU activation)  
- Output Layer (10 neurons, Softmax activation)

### 3️⃣ Model Training

- Optimizer: Adam  
- Loss Function: Sparse Categorical Crossentropy  
- Epochs: 3  

The trained model is saved as:

handwritten_model.keras

### 4️⃣ Model Evaluation

- Evaluated on test dataset  
- Prints Loss and Accuracy  
- Measures performance on unseen data  

### 5️⃣ Custom Image Prediction

The system supports prediction on custom digit images:

- Loads image using OpenCV  
- Converts to grayscale  
- Inverts pixel values  
- Predicts digit using trained model  
- Displays output using Matplotlib  

Image format example:

digits/digit1.png  
digits/digit2.png  

## 🛠 Tools & Libraries

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Matplotlib  

## 📈 Key Insights

- Neural networks effectively classify handwritten digits  
- Data normalization improves accuracy  
- Model performs well on standard MNIST test data  
- Preprocessing consistency is important for custom predictions  

## ⚠️ Limitations

- Uses fully connected Dense layers instead of CNN  
- Performance may drop for noisy custom images  
- Not deployed as an API or web application  

## 🚀 Future Improvements

- Upgrade to Convolutional Neural Network (CNN)  
- Add validation split & callbacks  
- Build Flask or FastAPI API  
- Deploy as a web app  

## 👨‍💻 Author

**Syed Danish Ahmed**  
**Aspiring Data Scientist | Computer Engineering Student**  

If you find this project helpful, consider ⭐ starring the repository. Your support is greatly appreciated!

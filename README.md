# 🌿 Plant Disease Prediction using CNN

This project leverages **Convolutional Neural Networks (CNNs)** to detect and classify plant leaf diseases from images. It is designed to assist farmers and agricultural specialists in identifying plant health issues early, enabling prompt and effective treatment to improve crop yields.

---

## 🧠 Project Objectives

- Build a CNN model for automatic plant disease classification.
- Use image data (e.g., PlantVillage dataset) to train and evaluate the model.
- Achieve high accuracy and generalization using data augmentation and model optimization.
- Provide a foundation for potential deployment in web/mobile applications for real-time usage.

---

## 📂 Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes**: Includes both healthy and diseased leaves for crops like:
  - Tomato (Early Blight, Late Blight, Healthy)
  - Potato (Early Blight, Late Blight, Healthy)
  - Corn (Common Rust, Leaf Blight, Healthy)
- **Format**: Image files categorized into folders by class label.

---

## 🛠️ Technologies Used

| Tool           | Purpose                        |
|----------------|--------------------------------|
| Python         | Programming Language           |
| TensorFlow/Keras | Model Building & Training    |
| NumPy & Pandas | Data Manipulation              |
| OpenCV         | Image Processing               |
| Matplotlib     | Visualization                  |
| Jupyter Notebook | Development Environment      |

---

## 📐 Model Architecture

- **Input Layer**: Resized RGB leaf images
- **Convolutional Layers**: Feature extraction
- **MaxPooling Layers**: Dimensionality reduction
- **Dropout Layers**: Prevent overfitting
- **Dense Layers**: Final classification
- **Activation**: ReLU + Softmax (for multi-class output)

---

## 📊 Model Performance

| Metric         | Score         |
|----------------|---------------|
| Training Accuracy | > 95%      |
| Validation Accuracy | > 90%    |
| Loss Curves    | Showed good convergence with minimal overfitting
| Evaluation     | Confusion matrix, classification report

---

## 📁 Directory Structure

plant-disease-prediction/
│
├── dataset/ # Training and testing images
│ ├── train/
│ └── test/
│
├── plant_disease_prediction.ipynb # Main Jupyter notebook
├── models/ # Saved CNN model (e.g., model.h5)
├── predictions/ # Sample outputs
├── requirements.txt # List of dependencies
├── LICENSE
└── README.md

yaml
Copy
Edit

---

## 🚀 How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/plant-disease-prediction.git
   cd plant-disease-prediction
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook plant_disease_prediction.ipynb
Train or Test the model:

You can retrain the model on your dataset.

You can load a saved .h5 model and run predictions on new images.

🧪 Sample Prediction
Input Image	Model Output
Tomato - Early Blight

📜 License
This project is licensed under the MIT License.
See the LICENSE file for more information.

👨‍💻 Author
Sandeep Gope
📧 Email: [your-email@example.com]
🔗 LinkedIn: linkedin.com/in/sandeepgope

📌 Future Enhancements
✅ Deploy model via Streamlit or Flask

✅ Real-time disease detection using webcam or mobile camera

✅ Integration with drone imagery or satellite data

✅ Support for more crops and regional datasets

🙏 Acknowledgements
PlantVillage Dataset

TensorFlow/Keras community

Open-source contributors


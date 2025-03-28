# Leaf Disease Detection using Deep Learning

<!-- ![Demo](https://github.com/shukur-alom/leaf-diseases-detect/blob/main/Media/website.gif) -->

This project is a leaf disease detection system that uses deep learning techniques, including transfer learning, to identify and classify 33 different types of leaf diseases. The model has been trained on a large dataset of images and is designed to help agricultural professionals and enthusiasts diagnose plant diseases in a fast and accurate manner.

<img width="1354" alt="Screenshot 2025-03-28 at 12 04 49 PM" src="https://github.com/user-attachments/assets/fbef5e9b-f80e-498e-b7b5-0efeaed84d61" />


---

## Features

- **Accurate Disease Detection**: Identifies 33 different types of leaf diseases with high accuracy.
- **User-Friendly Interface**: Powered by Streamlit for an intuitive and interactive web-based interface.
- **Fast Inference**: Optimized for quick predictions, even on large datasets.
- **Transfer Learning**: Leverages pre-trained models for improved performance and reduced training time.
- **Scalable**: Can be extended to include more diseases or plant types with additional training data.

---

## Tech Stack

- **Programming Language**: Python
- **Frameworks and Libraries**:
  - TensorFlow/Keras: For building and training the deep learning model.
  - Streamlit: For creating the web-based user interface.
  - NumPy, Pandas: For data manipulation and preprocessing.
  - OpenCV: For image processing.
- **Deployment**: Streamlit-based local deployment (can be extended to cloud platforms like AWS, GCP, or Azure).
- **Dataset**: Custom dataset containing images of 33 leaf diseases.

---

## Usage

To use the model for leaf disease detection, follow these steps:

1. Make sure you have a Python environment set up with the necessary libraries installed. You can use the provided `requirements.txt` file to set up the required dependencies.

   ```
   pip install -r requirements.txt
   ```

2. Run the application:

   ```
   streamlit run main.py
   ```

3. Upload an image of a leaf, and the system will classify the disease and provide the result.

---

## Model Details

The leaf disease detection model is built using deep learning techniques, and it uses transfer learning to leverage the pre-trained knowledge of a base model. The model is trained on a dataset containing images of 33 different types of leaf diseases. For more information about the architecture, dataset, and training process, please refer to the code and documentation provided.

---

## Future Enhancements

- **Mobile App Integration**: Develop a mobile app for on-the-go disease detection.
- **Cloud Deployment**: Host the application on a cloud platform for global accessibility.
- **Real-Time Detection**: Integrate with IoT devices for real-time disease monitoring in fields.
- **Additional Diseases**: Expand the dataset to include more plant species and diseases.

---

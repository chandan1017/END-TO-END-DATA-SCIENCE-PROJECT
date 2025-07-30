# END-TO-END-DATA-SCIENCE-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: SAI CHANDAN P B

INTERN ID: CT04DH251

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTHOSH

## 📝 Project Overview

As part of my internship at CodTech, I successfully completed Task 3: End-to-End Data Science Project, which focused on building and deploying an image classification model to distinguish between cats and dogs. The goal of this project was to implement a complete data science workflow—from dataset preparation and model training to web-based deployment—demonstrating how machine learning models can be integrated into real-world applications. The dataset was manually created and structured within the dataset/train/ directory, containing two subfolders—cats/ and dogs/—each with four sample images (e.g., cat1.jpeg to cat4.jpeg and dog1.jpeg to dog4.jpeg). I used Python 3.13 along with PyTorch to build and train a Convolutional Neural Network (CNN), while libraries like Torchvision facilitated image transformations and NumPy and Matplotlib supported numerical operations and visualizations. Model training was conducted via the train_model.py script, and the trained model was saved as cat_dog_model.pth in the models/ directory for later use.

For deployment, I used Flask, a lightweight Python web framework, to create a simple and interactive user interface. The core classification logic was implemented in main.py, which handled model loading and image prediction, while app.py served as the application’s entry point. The front-end was built with HTML and placed in the templates/index.html file, allowing users to upload images and receive real-time predictions on whether the image was a cat or a dog. Uploaded files were temporarily stored in the static/uploads/ directory for processing. All development was carried out in Visual Studio Code (VS Code) on a Windows environment. The model achieved approximately 75% accuracy on this small dataset, which was validated through console outputs and successful predictions on test images. This task gave me valuable hands-on experience in deep learning, model deployment, and building end-to-end machine learning pipelines applicable in various real-world domains.

#OUTPUT

<img width="621" height="587" alt="Image" src="https://github.com/user-attachments/assets/43615f19-af6a-42a8-874d-6b04ea3fcd98" />

<img width="557" height="572" alt="Image" src="https://github.com/user-attachments/assets/4a7ddffb-d6b7-4456-8cb0-4e8baefc1c97" />

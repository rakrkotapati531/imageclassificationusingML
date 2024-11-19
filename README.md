# Image Classifier using Machine Learning

This project is an image classifier built using machine learning techniques, specifically Support Vector Machines (SVM) with a Radial Basis Function (RBF) kernel. The dataset comprises images from three distinct categories: "pretty sunflower," "ruby ball leather," and "Ice cream cone." Images are downloaded from Bing using the Bing Image Downloader library.

## Technologies Used:
- **Python:** The primary programming language for the project.
- **scikit-learn:** Utilized for building and training the machine learning model. GridSearchCV is employed for hyperparameter tuning, optimizing SVM parameters (C and gamma).
- **Bing Image Downloader:** A Python library used to download images from Bing for creating the dataset.
- **streamlit:** A web application framework for creating interactive and user-friendly interfaces. The Streamlit app allows users to upload an image and receive predictions from the trained model.
- **numpy and matplotlib:** Used for array operations, data manipulation, and visualization.
- **PIL (Pillow):** Employed for handling image files and facilitating image-related operations.
- **pickle:** Used to serialize and deserialize the trained SVM model for easy storage and retrieval.

## Model Details:
- **Support Vector Machine (SVM):** SVM is a powerful supervised learning algorithm used for classification and regression tasks. In this project, an SVM classifier is employed for image classification.
- **Radial Basis Function (RBF) Kernel:** The RBF kernel is selected for its ability to handle non-linear relationships in the data, making it well-suited for image classification tasks.

## Steps:
1. **Data Collection:**
   In this step, images are collected from Bing using the Bing Image Downloader library. The search terms used for image retrieval are 'pretty sunflower,' 'ruby ball leather,' and 'Ice cream cone.' The images are downloaded and stored 
   in the 'images' directory.
   
  

3. **Preprocessing:**
   The collected images undergo preprocessing to make them suitable for model training. Each image is resized to a common size of 150x150 pixels with three color channels (RGB). Additionally, the images are flattened to create a one-dimensional array for each image.



   

5. **Model Training:**
   A Support Vector Machine (SVM) classifier is employed for the image classification task. The scikit-learn library is used for implementing the SVM model. Hyperparameter tuning is performed using GridSearchCV to find the optimal     
   values for the regularization parameter (C) and the kernel coefficient (gamma).
  ![image](https://github.com/charan1207/Image_Classification_using_machine_learning/assets/28255223/6c304389-62c2-47ff-99fc-9245f4eb8426)


   
7. **Model Evaluation:**
   The trained model's performance is evaluated using a test set that was previously separated from the dataset. The accuracy score is calculated, and a confusion matrix is generated to assess the model's ability to correctly classify 
   images into the specified categories.
   
9. **Model Deployment:**
    The trained SVM model is saved for future use using the pickle library. This serialized model can be easily loaded and used to make predictions on new, unseen images without the need to retrain the model.
   
  
10. **Streamlit Web App:**
    A user-friendly web application is developed using Streamlit. Users can interact with the application by uploading an image through the web interface. The uploaded image is then fed into the trained SVM model, and predictions with 
    probabilities for each class are displayed to the user.



    
## Instructions for Running the Streamlit App:
1. Install required dependencies using `pip install -r requirements.txt`.
2. Run the app using `streamlit run app.py`.
3. Upload an image and click the "Predict" button to see the model's predictions.

Feel free to explore and contribute to the project. If you have any questions or suggestions, please open an issue or reach out.

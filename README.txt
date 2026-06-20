Resume Screening System
=======================

Project Description
-------------------
This project is a Resume Screening System built using Python, Streamlit, and Machine Learning.

The system allows users to upload PDF or DOCX files and automatically:
- detect whether the file is a resume or not
- predict the most suitable job role for resume files
- calculate a score
- show results in table format
- download results as a CSV file
- clear all uploaded files using one button

Main Features
-------------
1. Upload one or more PDF or DOCX files
2. Detect Resume or Not Resume
3. Predict job role for resume files
4. Calculate score using model confidence
5. Show results in column format
6. Download results as CSV
7. Preview extracted text from uploaded files
8. Remove all uploaded files with one button

Supported File Types
--------------------
- PDF (.pdf)
- DOCX (.docx)

Project Folder Structure
------------------------
Resume_Screening_System/
│
├── app.py
├── train_model.py
├── README.txt
├── requirements.txt
│
├── data/
│   ├── resumes.csv
│   └── resume_detector.csv
│
├── models/
│   ├── role_model.pkl
│   ├── role_vectorizer.pkl
│   ├── detector_model.pkl
│   └── detector_vectorizer.pkl
│
└── utils/
    ├── __init__.py
    ├── preprocess.py
    └── extractor.py

Files Explanation
-----------------
1. app.py
   Main Streamlit app. It loads trained models, accepts file uploads, predicts results, and displays scores.

2. train_model.py
   Trains the machine learning models and saves them inside the models folder.

3. README.txt
   Contains project explanation, setup steps, and usage instructions.

4. requirements.txt
   Contains the Python libraries required to run the project.

5. utils/preprocess.py
   Cleans extracted text before prediction.

6. utils/extractor.py
   Extracts text from PDF and DOCX files.

7. data/resumes.csv
   Dataset used to train the job role prediction model.

8. data/resume_detector.csv
   Dataset used to train the Resume / Not Resume model.

How the Score is Calculated
---------------------------
The score is based on two values:

1. Role Confidence
   - How confident the model is about the predicted job role

2. Resume Probability
   - How confident the model is that the uploaded file is actually a resume

Formula used:

    Score = ((Role Confidence x 100) x 0.7) + ((Resume Probability x 100) x 0.3)

Maximum score allowed:

    99.00

This means:
- 70 percent weight is given to predicted job role confidence
- 30 percent weight is given to resume detection confidence
- no extra score boost is applied
- final score will never go above 99.00

Status Rules
------------
- Not Resume -> File is not a resume
- Score below 60 -> Low Match for Predicted Job Role
- Score from 60 to 80 -> Applicable for Predicted Job Role
- Score above 80 -> Selected for Predicted Job Role

How to Run the Project
----------------------
Step 1: Open terminal inside the project folder

Step 2: Install required libraries

    pip install -r requirements.txt

Step 3: Train the models

    python train_model.py

Step 4: Run the Streamlit app

    streamlit run app.py

Step 5: Open the browser link shown in the terminal

Required Python Version
-----------------------
Recommended:

    Python 3.10 or above

Dataset Information
-------------------
1. resumes.csv
   This dataset should contain:
   - resume text
   - job role label

   Possible text column names:
   - resume_text
   - Resume_str
   - Text
   - text

   Possible label column names:
   - category
   - Category
   - label

2. resume_detector.csv
   This dataset should contain:
   - text
   - resume / not-resume label

   Possible text column names:
   - text
   - resume_text
   - Text
   - Resume_str

   Possible label column names:
   - is_resume
   - label
   - category
   - Category

If resume_detector.csv is missing or empty, the project automatically creates a small bootstrap detector dataset using sample non-resume documents.

Libraries Used
--------------
- streamlit
- pandas
- scikit-learn
- PyPDF2
- python-docx
- openpyxl

Conclusion
----------
This project is beginner-friendly and useful for learning:
- text preprocessing
- PDF and DOCX extraction
- machine learning classification
- Streamlit app development

Future improvements can include:
- job description matching
- skill matching
- resume ranking
- Excel export
- better dashboard analytics

End of File
-----------
# 📄 Resume Screening System

> 🚀 A smart Resume Screening System built using **Python**, **Streamlit**, **Machine Learning**, and **NLP** to classify uploaded documents as **Resume** or **Not Resume**, predict the **job role**, and generate a **screening score** with final status.

---

## 🎯 Project Overview

The **Resume Screening System** helps automate the first stage of the recruitment process.  
Instead of checking every resume manually, this system quickly analyzes uploaded files and provides useful screening results.

### ✅ What this system does
- 📂 Upload one or more **PDF** or **DOCX** files
- 📝 Extract text from uploaded documents
- 🔍 Detect whether the file is a **Resume** or **Not Resume**
- 💼 Predict the **job role/category** for valid resumes
- 📊 Calculate a **matching score**
- 📌 Show the final screening **status**
- 📥 Download the result as a **CSV file**

---

## ✨ Features

- 📄 Upload multiple resumes at once
- 🔎 Extract text from PDF and DOCX files
- 🤖 Resume / Not Resume classification
- 💼 Job role prediction
- 📊 Score calculation based on model confidence
- ✅ Final status generation
- 👀 Preview extracted text
- 📥 Download screening results as CSV
- 🧠 Train and save machine learning models

---

## 🛠️ Technologies Used

- 🐍 Python
- 🌐 Streamlit
- 📊 Pandas
- 🤖 Scikit-learn
- 📄 PyPDF2
- 📝 python-docx
- 📂 openpyxl
- 📘 reportlab
- 💾 pickle
- 📁 pathlib

---

## 🧠 Machine Learning Models Used

This project uses **two machine learning models**.

### 1. Resume Detector Model
This model checks whether the uploaded document is:
- ✅ Resume
- ❌ Not Resume

### 2. Role Prediction Model
If the document is a valid resume, this model predicts the most suitable:
- 💼 Job Role / Category

### ⚙️ Techniques Used
- TF-IDF Vectorization
- Logistic Regression
- Text Cleaning / Preprocessing

---

## 🔄 How the System Works

### Step 1: Upload Files
The user uploads one or more files:
- `.pdf`
- `.docx`

### Step 2: Text Extraction
The system extracts text from the uploaded files.

### Step 3: Resume Detection
The extracted text is checked to determine whether it is a **resume** or not.

### Step 4: Job Role Prediction
If the file is a valid resume, the system predicts the **job role**.

### Step 5: Score Calculation
The system calculates the score based on:
- Role confidence
- Resume probability

### Step 6: Final Status
The system shows one of these final results:
- ✅ **Selected for Predicted Job Role**
- 📌 **Applicable for Predicted Job Role**
- ⚠️ **Low Match for Predicted Job Role**
- ❌ **Not Resume**

### Step 7: Export Results
The final results can be downloaded as a **CSV file**.

---

## 📊 Score Calculation

The final score is calculated using:

```text
Final Score = (Role Confidence × 70%) + (Resume Probability × 30%)


🏗️ Project Structure

Resume_Screening_System/
│── app.py
│── train_model.py
│── requirements.txt
│── README.md
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
├── screenshots/
│   ├── home_page.png
│   ├── upload_section.png
│   ├── results_page.png
│   └── text_preview.png
│
└── utils/
    ├── extractor.py
    └── preprocess.py
Requirements

Your requirements.txt should contain:
streamlit
pandas
scikit-learn
PyPDF2
python-docx
openpyxl
reportlab
    
📂 Dataset Details
This project uses two datasets.
1. resumes.csv
Used to train the job role prediction model.
2. resume_detector.csv
Used to train the resume vs not-resume detection model.


📋 Output Columns
The application shows the following result columns:

📄 File Name
🏷️ Document Type
💼 Predicted Job Role
📊 Score
✅ Status


📌 Example Output Status

✅ Selected for Predicted Job Role
📌 Applicable for Predicted Job Role
⚠️ Low Match for Predicted Job Role
❌ Not Resume


🎯 Use Cases

👨‍💼 Resume shortlisting
🧾 Candidate screening
💼 Job role prediction
🏢 Recruitment automation
🎓 Academic mini project / major project
🤖 Practical use of NLP + ML + Streamlit

🔮 Future Improvements

Add support for more file formats
Improve prediction accuracy using advanced NLP models
Add skill extraction from resumes
Add recruiter dashboard
Deploy the project online
Add database integration
Add analytics and graphical reports

👨‍💻 Author
Surjeet Kumar
🔗 GitHub: https://github.com/Surjeet9508

🤝 Contributing
Contributions, suggestions, and improvements are welcome.
Feel free to fork this repository and create a pull request.

📜 License
This project is licensed under the MIT License.

⭐ Support
If you like this project, give it a star ⭐ on GitHub.

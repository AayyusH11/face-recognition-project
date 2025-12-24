🚀 Features

-Real-time face detection and recognition using webcam
-Facial encoding–based identity matching (not pixel matching)
-Automatic Late / Present classification (after 9:00 AM → Late)
-Attendance recorded with Name, Date, Time, and Status
-CSV-based attendance logging
-Prevents duplicate attendance in a single session
-Clean and modular Python implementation

🎥 Demo Video

📌 Watch the working demo of the project here: (https://drive.google.com/drive/folders/1iHT-Pvbr0JuuClgcRleMkXT0QpY1N1I2?usp=sharing)



The demo shows:
-Webcam-based face recognition
-Automatic attendance marking
-Late vs Present classification
-Generated attendance CSV output
-This project relies on local camera access, so a demo video is provided instead of live deployment.

🛠️ Tech Stack

-Python 3.11
-OpenCV – webcam access & image processing
-face-recognition (dlib) – facial encoding & matching
-NumPy – numerical operations
-CSV – attendance storage

📂 Project Structure
```
face-recognition-project/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── image/
│   └── README.txt          # Instructions for adding images
│
└── sample_output/
    └── attendance_sample.csv
```

▶️ How to Run Locally

1️⃣ Clone the repository
```
git clone https://github.com/AayyusH11/face-recognition-project.git
cd face-recognition-project
```

2️⃣ Create and activate virtual environment
```
python -m venv venv
venv\Scripts\activate
```
3️⃣ Install dependencies
```
pip install -r requirements.txt
```
4️⃣ Run the project
```
python main.py
```


Press Q to stop the webcam.

🧾 Attendance Output Format

Attendance is stored in CSV format:
```
Name,Date,Time,Status
Ayush,2025-01-24,08:55:12,Present
Rahul,2025-01-24,09:14:33,Late
```


A sample output is provided in the sample_output/ folder.

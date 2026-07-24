ResumeAI Pro – Smart Resume Analyzer

A full-stack AI-powered Resume Analyzer that evaluates resumes against job descriptions, provides ATS scores, identifies missing skills, and generates personalized improvement recommendations using Machine Learning and Generative AI.

Tech Stack: Flask • Python • MySQL • HTML • CSS • JavaScript • Google Gemini AI • Scikit-learn • NLTK

The backend is built using Flask with SQLAlchemy, PDF/DOCX parsing, TF-IDF, cosine similarity, and Gemini AI integration, while the frontend provides a modern responsive interface for authentication, resume upload, analysis, and dashboard visualization.

Features
User Registration & Login
Secure Session Authentication
Resume Upload (PDF & DOCX)
Job Description Analysis
ATS Score Calculation
Resume & Job Match Score
Missing Skills Detection
Skill Gap Analysis
AI-Powered Resume Feedback
Resume Improvement Suggestions
Analysis History
Dashboard with Statistics
Responsive Modern UI
Project Structure
ResumeAI-Pro/
│
├── app.py                  # Flask Backend
├── index.html              # Frontend (HTML, CSS & JavaScript)
├── requirements.txt
├── README.md
└── database/
Technologies Used
Backend
Python
Flask
Flask SQLAlchemy
Flask CORS
MySQL
PyMySQL
AI & Machine Learning
Google Gemini API
NLTK
Scikit-learn
TF-IDF Vectorizer
Cosine Similarity
File Processing
PyPDF2
python-docx
Frontend
HTML5
CSS3
JavaScript (Vanilla)
Database

The application uses MySQL with three primary tables:

Users
Sessions
Analyses

These tables store authentication details, user sessions, and resume analysis history.

API Endpoints

Authentication
Method	Endpoint
POST	/api/auth/signup
POST	/api/auth/login
POST	/api/auth/logout
GET	/api/auth/me
Resume Analysis
Method	Endpoint
POST	/api/analyze
GET	/api/analyses
GET	/api/analyses/{analysis_id}
DELETE	/api/analyses/{analysis_id}
GET	/api/analyses/stats
Utility
Method	Endpoint
GET	/api/health

How It Works
User creates an account.
User logs in.
Uploads a resume (PDF/DOCX).
Pastes a Job Description.
Backend extracts resume text.
Skills are extracted using NLP.
TF-IDF and Cosine Similarity calculate the job match.
Gemini AI generates detailed feedback.
Results are stored in MySQL.
Dashboard displays scores, insights, and analysis history.
Key Analysis Metrics
ATS Score
Job Match Percentage
Resume Strengths
Weaknesses
Missing Skills
Technical Skills
Soft Skills
Resume Recommendations
AI Feedback
Overall Hiring Recommendation
User Interface

The frontend includes:

Authentication pages
Dashboard
Resume Upload
Loading Screen
Analysis Results
Statistics Cards
Skill Gap Visualization
Responsive Design

The UI is built with modern HTML, CSS, and JavaScript and includes responsive layouts, dashboard cards, animated loading states, and detailed result visualizations.

Future Enhancements
Resume Builder
LinkedIn Profile Analysis
Multi-language Resume Support
Cover Letter Generator
Resume Templates
Company-wise ATS Optimization
Interview Question Generator
Resume Version Comparison
Admin Dashboard
Cloud Deployment

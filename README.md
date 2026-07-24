ResumeAI Pro – Smart Resume Analyzer

A full-stack AI-powered Resume Analyzer that evaluates resumes against job descriptions, provides ATS scores, identifies missing skills, and generates personalized improvement recommendations using Machine Learning and Generative AI.

Tech Stack: Flask • Python • MySQL • HTML • CSS • JavaScript • Google Gemini AI • Scikit-learn • NLTK

The backend is built using Flask with SQLAlchemy, PDF/DOCX parsing, TF-IDF, cosine similarity, and Gemini AI integration, while the frontend provides a modern responsive interface for authentication, resume upload, analysis, and dashboard visualization.

## ✨ Features

### 🔐 Authentication
- User Registration
- User Login
- Secure Session Authentication
- Password Encryption

### 📄 Resume Analysis
- Upload Resume (PDF & DOCX)
- Job Description Analysis
- ATS Score Calculation
- Resume & Job Match Score
- AI-Powered Resume Feedback

### 🎯 Skill Analysis
- Technical Skills Detection
- Missing Skills Identification
- Skill Gap Analysis
- Resume Improvement Suggestions

### 📊 Dashboard & Analytics
- Analysis History
- Dashboard with Statistics
- Previous Resume Reports
- Performance Insights

### 🤖 AI Capabilities
- Google Gemini AI Integration
- Intelligent Resume Evaluation
- Personalized Recommendations
- ATS Optimization Suggestions

### 💻 User Experience
- Modern Responsive UI
- Drag & Drop Resume Upload
- Interactive Dashboard
- Fast Resume Processing
- Mobile-Friendly Design


## 📁 Project Structure

```text
ResumeAI-Pro/
│
├── app.py                 # Flask backend application
├── index.html             # Frontend (HTML, CSS, JavaScript)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── uploads/               # Temporary uploaded resumes
├── static/                # Static assets (CSS, JS, Images)
├── templates/             # HTML templates
├── database/
   ├── schema.sql         # Database schema
   └── resume_analyzer.sql# MySQL database dump
```


## 🛠️ Technologies Used

### Backend
- Python
- Flask
- Flask-SQLAlchemy
- Flask-CORS
- MySQL
- PyMySQL

### 🤖 AI & Machine Learning
- Google Gemini API
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

### 📄 File Processing
- PyPDF2
- python-docx

### 🎨 Frontend
- HTML5
- CSS3
- JavaScript (Vanilla)


## 🗄️ Database

The application uses **MySQL** with three primary tables:

| Table | Purpose |
|--------|----------|
| **Users** | Stores user account information |
| **Sessions** | Manages user authentication sessions |
| **Analyses** | Stores resume analysis results and history |

These tables work together to securely manage authentication, user sessions, and resume analysis records.

## 🔗 API Endpoints

### 🔐 Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | Register a new user |
| POST | `/api/auth/login` | User login |
| POST | `/api/auth/logout` | Logout user |
| GET | `/api/auth/me` | Get logged-in user |

---

### 📄 Resume Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Analyze a resume |
| GET | `/api/analyses` | Retrieve analysis history |
| GET | `/api/analyses/{analysis_id}` | Get a specific analysis |
| DELETE | `/api/analyses/{analysis_id}` | Delete an analysis |
| GET | `/api/analyses/stats` | Retrieve dashboard statistics |

---

### ⚙️ Utility

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check API health status |


## ⚙️ How It Works

1. 👤 User creates an account.
2. 🔐 User logs into the application.
3. 📄 Uploads a resume (PDF/DOCX).
4. 📝 Pastes the Job Description.
5. 📚 Backend extracts resume content.
6. 🧠 NLP identifies technical and soft skills.
7. 📊 TF-IDF & Cosine Similarity calculate the job match score.
8. 🤖 Google Gemini AI generates personalized feedback.
9. 💾 Analysis results are stored in MySQL.
10. 📈 Dashboard displays scores, insights, and analysis history.


## 📊 Key Analysis Metrics

- ✅ ATS Compatibility Score
- 🎯 Job Match Percentage
- 💪 Resume Strengths
- ⚠️ Weaknesses
- 🔍 Missing Skills
- 💻 Technical Skills Analysis
- 🤝 Soft Skills Analysis
- 💡 Resume Improvement Suggestions
- 🤖 AI-Powered Feedback
- 🏆 Overall Hiring Recommendation


## 🖥️ User Interface

The application includes the following user interfaces:

- 🔐 Authentication Pages
- 📊 Dashboard
- 📄 Resume Upload Page
- ⏳ Loading Screen
- 📈 Analysis Results
- 📋 Statistics Cards
- 🎯 Skill Gap Visualization
- 📱 Responsive Design

### UI Highlights

- Modern and clean interface
- Responsive layout for all devices
- Interactive dashboard
- Animated loading states
- Beautiful score visualizations
- Easy-to-use navigation


## 🚀 Future Enhancements

- 📄 Resume Builder
- 🔗 LinkedIn Profile Analysis
- 🌍 Multi-language Resume Support
- ✍️ AI Cover Letter Generator
- 🎨 Professional Resume Templates
- 🏢 Company-specific ATS Optimization
- 🎤 AI Interview Question Generator
- 📊 Resume Version Comparison
- 👨‍💼 Admin Dashboard
- ☁️ Cloud Deployment

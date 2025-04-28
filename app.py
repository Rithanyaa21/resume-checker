# --- Imports ---
import streamlit as st
import os
import fitz  # PyMuPDF for PDFs
import docx
import joblib
import re
import string
import nltk
import pandas as pd
from pymongo import MongoClient, errors as pymongo_errors # Import pymongo errors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
# --- End Imports ---

# --- *** Page Config MUST BE FIRST st command *** ---
st.set_page_config(
    page_title="Resume Checker Pro",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="🚀"
)
# --- End Page Config ---

# Suppress specific warnings if needed (optional)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- NLTK Downloads (Corrected & Quieter) ---
def download_nltk_data(resource, resource_url):
    try:
        nltk.data.find(resource_url)
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        try:
            nltk.download(resource, quiet=True)
            print(f"'{resource}' downloaded.")
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}'. Error: {e}")
download_nltk_data('punkt', 'tokenizers/punkt')
download_nltk_data('stopwords', 'corpora/stopwords')
download_nltk_data('wordnet', 'corpora/wordnet')
# --- End NLTK Downloads ---


# --- MongoDB Connection ---
MONGO_URI = "mongodb+srv://Rithanyaa:Rith212004@cluster0.7j1km.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "resume_screening"

@st.cache_resource(ttl=3600)
def get_mongo_client():
    print("Attempting MongoDB connection...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("MongoDB connection successful.")
        return client
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return None

client = get_mongo_client()

if client is None:
     st.error("Database Connection Error: Could not connect.")
else:
    try:
        db = client[DB_NAME]
        users_collection = db["users"]
        resumes_collection = db["resumes"]
        try:
            users_collection.create_index("email", unique=True)
            print("Unique index on 'email' ensured.")
        except pymongo_errors.OperationFailure as e:
            error_msg = str(e).lower()
            if "index already exists" in error_msg or e.code == 11000 or e.code in [85, 86]:
                print(f"Info: Index on 'email' exists or conflict. Details: {e}")
                if 'dup key: { email: "" }' in error_msg or 'dup key: { email: null }' in error_msg:
                     st.warning(
                         "**Data Consistency Issue:** Duplicate empty/null emails found. Unique email constraint cannot be fully enforced. Please clean the 'users' collection.",
                         icon="⚠️"
                     )
            else: raise e
    except Exception as e:
        st.error(f"Database Error: Could not access collections/ensure index. {e}")
        client = None
# --- End MongoDB ---


# --- Load ML Model & Vectorizer ---
@st.cache_resource
def load_models():
    print("Loading ML models...")
    try:
        model = joblib.load("resume_classifier.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        category_mapping = joblib.load("category_mapping.pkl")
        if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
             print("Warning: Loaded vectorizer lacks vocabulary.")
        print("ML models loaded.")
        return model, vectorizer, category_mapping
    except FileNotFoundError as e:
        st.error(f"Model Loading Error: '{e.filename}' not found.")
        return None, None, None
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None, None, None

model, vectorizer, category_mapping = load_models()

if not all([model, vectorizer, category_mapping]):
    st.error("Essential model files failed to load. Cannot continue.")
    st.stop()

# --- Text Extraction Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        uploaded_file.seek(0)
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text", sort=True) + "\n"
    except Exception as e:
        st.error(f"PDF Extraction Error: {e}")
    return text.strip()

def extract_text_from_docx(uploaded_file):
    text = ""
    try:
        uploaded_file.seek(0)
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"DOCX Extraction Error: {e}")
    return text.strip()
# --- End Text Extraction ---

# --- Text Cleaning Function ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s\+#]', '', text) # Keep letters, numbers, spaces, #, +
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    cleaned = " ".join(words)
    return cleaned
# --- End Text Cleaning ---

# --- Keyword Definitions ---
CATEGORY_KEYWORDS = {
    "Data Science": ["python", "machine learning", "ml", "deep learning", "dl", "nlp", "natural language processing", "data analysis", "data visualization", "data mining", "sql", "nosql", "mongodb", "cassandra", "database", "tableau", "power bi", "qlik", "statistics", "statistical modeling", "hypothesis testing", "a/b testing", "pytorch", "tensorflow", "keras", "scikit learn", "sklearn", "pandas", "numpy", "matplotlib", "seaborn", "regression", "classification", "clustering", "dimensionality reduction", "feature engineering", "model deployment", "mle", "mlops", "jupyter", "notebook", "aws", "sagemaker", "azure", "ml studio", "gcp", "ai platform", "vertex ai", "big data", "spark", "pyspark", "hadoop", "hive", "etl", "airflow", "kafka"],
    "HR": ["recruitment", "recruiting", "human resources", "hr", "employee relations", "talent acquisition", "talent management", "talent development", "onboarding", "offboarding", "orientation", "compensation", "benefits", "payroll", "salary", "reward", "performance management", "performance appraisal", "performance review", "hris", "hrms", "hrbp", "human resources information system", "applicant tracking system", "ats", "taleo", "workday", "successfactors", "greenhouse", "lever", "training", "learning development", "l&d", "corporate training", "personnel", "manpower planning", "hiring", "interviewing", "sourcing", "screening", "selection", "job description", "jd", "compliance", "labor law", "employment law", "policy", "diversity inclusion", "d&i", "employee engagement", "organizational development", "od"],
    "Web Designing": ["html", "html5", "css", "css3", "javascript", "ecmascript", "es6", "js", "web design", "web development", "front end", "frontend", "front-end", "ui", "ux", "user interface", "user experience", "design thinking", "human centered design", "wireframe", "wireframing", "prototype", "prototyping", "mockup", "storyboarding", "usability", "user testing", "accessibility", "a11y", "wcag", "human computer interaction", "hci", "information architecture", "ia", "user research", "persona", "user flow", "journey map", "figma", "adobe xd", "xd", "sketch", "photoshop", "illustrator", "invision", "zeplin", "balsamiq", "axure", "marvel", "framer", "miro", "react", "reactjs", "angular", "angularjs", "vue", "vuejs", "jquery", "redux", "mobx", "vuex", "ngrx", "rxjs", "nextjs", "next js", "nuxtjs", "gatsby", "svelte", "alpinejs", "bootstrap", "tailwind", "material ui", "chakra ui", "ant design", "foundation", "bulma", "semantic ui", "sass", "scss", "less", "stylus", "postcss", "css modules", "bem", "atomic design", "design system", "styled components", "emotion", "css variable", "grid", "flexbox", "webpack", "npm", "node package manager", "yarn", "parcel", "vite", "gulp", "grunt", "responsive design", "rwd", "mobile first", "adaptive design", "cross browser compatibility", "browser developer tools", "dev tools", "debugging", "performance optimization", "web performance", "lighthouse", "core web vitals", "progressive web app", "pwa", "web components", "single page application", "spa", "state management", "git", "github", "gitlab", "bitbucket", "version control", "testing", "jest", "cypress", "storybook", "visual regression testing", "api", "rest", "json", "graphql", "seo", "search engine optimization", "animation", "svg", "canvas", "webgl", "threejs", "cms", "wordpress", "drupal", "joomla"],
    "Software Development": ["java", "c++", "c#", "python", "ruby", "php", "javascript", "typescript", "go", "golang", "swift", "kotlin", "objective c", "scala", "rust", "perl", "elixir", "object oriented programming", "oop", "functional programming", "fp", "solid", "dry", "kiss", "yagni", "design patterns", "gang of four", "gof", "data structures", "algorithms", "complexity analysis", "big o", "agile", "scrum", "kanban", "lean", "xp", "extreme programming", "waterfall", "devops", "devsecops", "ci/cd", "continuous integration", "continuous deployment", "continuous delivery", "jenkins", "gitlab ci", "github actions", "circleci", "travis ci", "spring", "springboot", "spring boot", "java ee", "jakarta ee", "django", "flask", "fastapi", "node.js", "express", "nestjs", "koa", "ruby on rails", "rails", "asp.net", ".net core", ".net", "laravel", "symfony", "phoenix", "sql", "mysql", "postgresql", "postgres", "sql server", "oracle", "sqlite", "database design", "normalization", "orm", "hibernate", "entity framework", "sqlalchemy", "sequelize", "nosql", "mongodb", "redis", "memcached", "cassandra", "elasticsearch", "dynamodb", "couchbase", "cloud computing", "aws", "amazon web services", "azure", "microsoft azure", "google cloud platform", "gcp", "docker", "containerization", "kubernetes", "k8s", "openshift", "serverless", "lambda", "azure functions", "google cloud functions", "infrastructure as code", "iac", "terraform", "ansible", "chef", "puppet", "cloudformation", "api", "rest", "restful", "soap", "graphql", "grpc", "microservices", "monolith", "soa", "event driven architecture", "eda", "message queues", "message broker", "kafka", "rabbitmq", "activemq", "sqs", "pub/sub", "load balancing", "caching", "networking", "tcp/ip", "http", "https", "unit testing", "integration testing", "system testing", "end to end testing", "e2e", "performance testing", "security testing", "tdd", "bdd", "junit", "nunit", "xunit", "pytest", "unittest", "rspec", "phpunit", "selenium", "cypress", "playwright", "postman", "swagger", "openapi", "git", "github", "gitlab", "bitbucket", "svn", "jira", "confluence", "linux", "unix", "bash", "shell scripting", "powershell", "ide", "vscode", "intellij", "eclipse"]
}
# --- End Keyword Definitions ---

# --- *** CORRECTED Keyword-Based ATS Score Function *** ---
def calculate_keyword_ats_score(cleaned_resume, category):
    """
    Calculates a keyword-based ATS score.
    Score calculation is modified: if >= 5 keywords match, score starts at 70%
    and increases, otherwise it's a standard percentage.
    """
    required_keywords = CATEGORY_KEYWORDS.get(category, [])
    if not required_keywords:
        print(f"DEBUG: No keywords defined for category '{category}'. Returning 0.")
        return 0.0 # Return float

    resume_words = set(cleaned_resume.split())
    # --- Debugging Print Statements (Optional: Remove/comment out later) ---
    print("-" * 30)
    print(f"DEBUG: Calculating Keyword ATS Score for Category: '{category}'")
    matched_keywords_list = [keyword for keyword in required_keywords if keyword in resume_words]
    matched_keywords_count = len(matched_keywords_list)
    print(f"DEBUG: Found {matched_keywords_count} Matching Keywords: {matched_keywords_list}")
    print("-" * 30)
    # --- End Debugging Print Statements ---

    total_keywords = len(required_keywords)

    # --- MODIFIED SCORING LOGIC ---
    score_threshold = 5 # Number of keywords needed to trigger the boosted score
    base_boosted_score = 70.0
    bonus_per_keyword_above_threshold = 4.0 # How much score increases per keyword > threshold
    max_score = 99.0 # Cap the score

    # *** CORRECTED CONDITION: Changed > to >= ***
    if matched_keywords_count >= score_threshold:
        # If threshold OR MORE keywords matched, assign a score >= base_boosted_score
        # Logic: Start at 70 for 5 matches, add bonus for each *additional* match above 5
        if matched_keywords_count == score_threshold:
             ats_score = base_boosted_score
        else:
             # 6+ matches -> 70 + bonus for each one *above* 5
             ats_score = base_boosted_score + (matched_keywords_count - score_threshold) * bonus_per_keyword_above_threshold

        ats_score = min(max_score, ats_score) # Ensure score doesn't exceed max_score
        print(f"DEBUG: Matched >= {score_threshold} keywords. Assigning boosted score: {ats_score}")

    elif total_keywords > 0:
        # If FEWER than threshold keywords matched, calculate standard percentage
        ats_score = (matched_keywords_count / total_keywords) * 100
        print(f"DEBUG: Matched < {score_threshold} keywords. Calculating standard percentage: {ats_score}")
    else:
        # Handle case where total_keywords is 0
        ats_score = 0.0
        print(f"DEBUG: No required keywords defined. Score is 0.")
    # --- END MODIFIED SCORING LOGIC ---

    return round(ats_score, 1) # Return score rounded to one decimal place
# --- End Keyword ATS Score ---


# --- Resume Suggestions Function ---
def generate_resume_suggestions(cleaned_resume, category):
    required_keywords = CATEGORY_KEYWORDS.get(category, [])
    if not required_keywords: return f"No specific suggestions for '{category}'."
    resume_words = set(cleaned_resume.split())
    missing_keywords = [word for word in required_keywords if word not in resume_words]
    if missing_keywords:
        max_suggestions = 7
        suggestions = f"**For '{category}' roles, consider adding/highlighting:**\n"
        suggestions += "\n".join([f"* `{kw}`" for kw in missing_keywords[:max_suggestions]])
        if len(missing_keywords) > max_suggestions: suggestions += f"\n* ... and {len(missing_keywords) - max_suggestions} more."
        suggestions += "\n\n_Tip: Integrate keywords naturally into achievements._"
    else:
        suggestions = f"✅ Keywords align well for **{category}**. Focus on quantifiable achievements and tailoring."
    return suggestions
# --- End Suggestions ---

# --- Analysis Function ---
def analyze_resume(file, job_desc=None):
    predicted_category, keyword_ats_score, suggestions, match_percentage = "Error", 0.0, "Analysis failed.", None
    if file is None: return None, None, "Please upload a file.", None
    file_name = file.name
    ext = file_name.split(".")[-1].lower()
    text = ""
    if ext == "pdf": text = extract_text_from_pdf(file)
    elif ext == "docx": text = extract_text_from_docx(file)
    else: return None, None, "Unsupported file type.", None
    if not text: return None, None, "Could not extract text.", None
    cleaned_resume = clean_text(text)
    if not cleaned_resume: return None, None, "Resume empty after cleaning.", None
    try:
        if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
            st.error("Internal Error: Vectorizer not ready.")
            return "Error", 0.0, "Internal Error: Vectorizer not ready.", None
        resume_vector = vectorizer.transform([cleaned_resume])
        if resume_vector.nnz == 0: st.warning("Resume matches no known terms.", icon="⚠️")

        prediction = model.predict(resume_vector)[0]
        predicted_category = category_mapping.get(prediction, "Unknown Category")

        if predicted_category not in ["Unknown Category", "Error"]:
            # *** Call the CORRECTED keyword score function ***
            keyword_ats_score = calculate_keyword_ats_score(cleaned_resume, predicted_category)
            suggestions = generate_resume_suggestions(cleaned_resume, predicted_category)
        else:
            suggestions = "Cannot generate suggestions."

        if job_desc and job_desc.strip():
            cleaned_job_desc = clean_text(job_desc)
            if cleaned_job_desc:
                job_vector = vectorizer.transform([cleaned_job_desc])
                if resume_vector.nnz == 0 or job_vector.nnz == 0:
                    match_percentage = 0.0
                    if job_vector.nnz == 0: st.warning("JD matches no known terms.", icon="⚠️")
                else:
                    similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]
                    match_percentage = round(max(0.0, min(1.0, similarity_score)) * 100, 1)
            else:
                st.info("JD empty after cleaning.")
                match_percentage = None
    except Exception as e:
        st.error(f"Analysis Error: {e}")
    return predicted_category, keyword_ats_score, suggestions, match_percentage
# --- End Analysis Function ---


# --- Session State Initialization ---
defaults = {"authenticated": False, "user_name": "", "user_email": "", "auth_action": None}
for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
# --- End Session State ---

# --- Helper Functions ---
def set_auth_action(action):
    st.session_state.auth_action = action

def verify_password(plain_password, stored_password):
    print("SECURITY WARNING: Using plain text password comparison!")
    return plain_password == stored_password

def hash_password(password):
    print("SECURITY WARNING: Storing plain text password!")
    return password
# --- End Helper Functions ---


# --- Login/Signup Page Logic ---
if not st.session_state.authenticated:
    st.title("🚀 Welcome to Resume Checker Pro")
    st.write("Analyze your resume, get keyword scores, and match against job descriptions.")
    st.markdown("---")
    if st.session_state.auth_action is None:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
             st.button("Login", on_click=set_auth_action, args=('login',), use_container_width=True, type="primary")
             st.button("Signup", on_click=set_auth_action, args=('signup',), use_container_width=True)
        st.markdown("---")
    # Login
    if st.session_state.auth_action == 'login':
        st.header("Login")
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email_input")
            login_password = st.text_input("Password", type="password", key="login_password_input")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if not login_email or not login_password: st.warning("Please enter email and password.")
                elif not client: st.error("Database unavailable.")
                else:
                    try:
                        user = users_collection.find_one({"email": login_email})
                        if user and verify_password(login_password, user.get("password")):
                            st.session_state.authenticated = True
                            st.session_state.user_name = user.get("name", "User")
                            st.session_state.user_email = user.get("email")
                            st.session_state.auth_action = None
                            st.toast("Login Successful!", icon="✅")
                            st.rerun()
                        else: st.error("Invalid email or password.")
                    except Exception as e: st.error(f"Login failed: {e}")
        st.button("Don't have an account? Signup", key='switch_to_signup_btn', on_click=set_auth_action, args=('signup',))
    # Signup
    elif st.session_state.auth_action == 'signup':
        st.header("Signup")
        with st.form("signup_form"):
            signup_name = st.text_input("Full Name", key="signup_name")
            signup_email = st.text_input("Email", key="signup_email")
            signup_mobile = st.text_input("Mobile Number (Optional)", key="signup_mobile")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            submitted = st.form_submit_button("Signup", use_container_width=True)
            if submitted:
                if not signup_name or not signup_email or not signup_password: st.warning("Please fill required fields.")
                elif signup_password != confirm_password: st.error("Passwords do not match!")
                elif "@" not in signup_email or "." not in signup_email.split('@')[-1]: st.error("Please enter a valid email.")
                elif not client: st.error("Database unavailable.")
                else:
                    try:
                        if users_collection.count_documents({"email": signup_email}) > 0:
                            st.error("Email already registered. Please Login.")
                        else:
                            hashed_pw = hash_password(signup_password) # Use placeholder hash
                            users_collection.insert_one({"name": signup_name, "email": signup_email, "mobile": signup_mobile, "password": hashed_pw})
                            st.success("Account created! Please Login.")
                            st.session_state.auth_action = 'login'
                            st.rerun()
                    except Exception as e: st.error(f"Signup failed: {e}")
        st.button("Already have an account? Login", key='switch_to_login_btn', on_click=set_auth_action, args=('login',))

# --- Main Application Page (Authenticated User) ---
else:
    st.sidebar.header(f"Welcome {st.session_state.user_name}!")
    st.sidebar.write(f"({st.session_state.user_email})")
    if st.sidebar.button("Logout"):
        keys_to_clear = ["authenticated", "user_name", "user_email", "auth_action"]
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.toast("Logged out.", icon="👋")
        st.rerun()

    st.title("📄 Resume Analyzer")
    st.write("Upload resume (PDF/DOCX) and optionally paste a job description for analysis.")

    uploaded_file = st.file_uploader("1. Upload Your Resume", type=["pdf", "docx"], key="resume_upload")
    job_desc_input = st.text_area("2. Paste Job Description (Optional)", height=200, key="job_desc", placeholder="For Resume-JD matching...")

    analyze_pressed = st.button("3. Analyze Resume", key="analyze_button", type="primary", disabled=(uploaded_file is None), use_container_width=True)

    if analyze_pressed and uploaded_file is not None:
        results_col, suggestions_col = st.columns([1, 1.5])

        with st.spinner("Analyzing..."), results_col:
            uploaded_file.seek(0)
            category, keyword_ats_score, suggestions, match_percent = analyze_resume(uploaded_file, job_desc_input)

            st.markdown("---")
            st.subheader("📊 Analysis Results")
            if category == "Error" or category is None :
                st.error(suggestions if suggestions else "An unknown analysis error occurred.")
            else:
                st.metric(label="Predicted Category", value=category)
                # Display the keyword score (calculation method is now corrected)
                st.metric(label="Keyword ATS Score",
                          value=f"{keyword_ats_score:.1f}%",
                          help=f"Score based on keywords for '{category}'. Boosted if >=5 keywords match.") # Updated help text
                if match_percent is not None:
                    st.metric(label="Resume-JD Match",
                              value=f"{match_percent:.1f}%",
                              help="Content similarity score (TF-IDF Cosine).")
                elif job_desc_input and job_desc_input.strip():
                     st.info("JD match N/A (e.g., no vocabulary match).")

        with suggestions_col:
             if category != "Error" and category is not None :
                 st.markdown("---")
                 st.subheader("💡 Suggestions")
                 st.markdown(suggestions, unsafe_allow_html=True)

        if client and category != "Error" and category is not None:
            try:
                user_email = st.session_state.get("user_email", "unknown")
                resumes_collection.insert_one({ "user_email": user_email, "user_name": st.session_state.user_name, "file_name": uploaded_file.name, "predicted_category": category, "keyword_ats_score": keyword_ats_score, "job_description_match_score": match_percent, "suggestions": suggestions, "analysis_timestamp": pd.Timestamp.utcnow() })
                st.toast("Analysis saved.", icon="💾")
            except Exception as e: st.warning(f"Could not save results: {e}")

    elif analyze_pressed and uploaded_file is None:
         st.warning("Please upload a resume file first.")

    # Footer
    st.markdown("---")
    st.caption("Resume Checker Pro | Cloud Project Demo")
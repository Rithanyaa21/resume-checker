import os
import re
import string
import pandas as pd
import fitz  # PyMuPDF for PDF extraction
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("C:/Rithanyaa/6th SEM/MCA LAB/Resume_screening/UpdatedResumeDataSet.csv")

# ✅ Preprocessing functions
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return " ".join(words)

# ✅ Apply preprocessing
print("🔄 Preprocessing resumes...")
df['cleaned_resume'] = df['Resume'].apply(clean_text)

# ✅ Encode target labels
df['Category'] = df['Category'].astype('category')
df['Category_Code'] = df['Category'].cat.codes  # Numeric labels for categories

# ✅ Save category mapping
category_mapping = dict(enumerate(df['Category'].cat.categories))

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_resume'], df['Category_Code'], test_size=0.2, random_state=42)

# ✅ Convert text to numerical format (TF-IDF)
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)

# ✅ Train a powerful model with cross-validation
print("🤖 Training the model with cross-validation...")
model = RandomForestClassifier(n_estimators=300, random_state=42)
cross_val_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
print(f"✅ Cross-validation Accuracy: {cross_val_scores.mean():.2f} ± {cross_val_scores.std():.2f}")

model.fit(X_train_tfidf, y_train)

# ✅ Test model accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save the trained model and vectorizer
joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(category_mapping, "category_mapping.pkl")
print("✅ Model training complete! Model, vectorizer, and category mapping saved successfully!")

# ✅ Load trained model & vectorizer
print("🔄 Loading trained model for prediction...")
model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
category_mapping = joblib.load("category_mapping.pkl")

# ✅ Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return None

# ✅ Function to classify user-uploaded resume
def classify_resume(file_path):
    if not os.path.exists(file_path):
        print("❌ File not found! Please enter a valid path.")
        return

    print("📂 Extracting text from resume...")
    resume_text = extract_text_from_pdf(file_path)
    if not resume_text:
        print("❌ Could not extract text from resume.")
        return
    
    cleaned_resume = clean_text(resume_text)
    resume_vector = vectorizer.transform([cleaned_resume])
    prediction = model.predict(resume_vector)[0]

    # ✅ Handle unknown predictions
    if prediction in category_mapping:
        predicted_category = category_mapping[prediction]
        print(f"✅ The uploaded resume belongs to category: {predicted_category}")
    else:
        print("⚠️ Warning: Model predicted an unknown category. Please check the dataset and retrain the model.")

# ✅ Get user input for resume file
file_path = input("Enter the full path of the resume file (PDF only): ").strip().replace('"', '').replace("'", "")
classify_resume(file_path)
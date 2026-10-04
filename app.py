import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.15), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.12), transparent 30%),
        #0b1020;
    color: #f8fafc;
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 35px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

.score-card {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
}

.score-label {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 8px;
}

.score-value {
    font-size: 32px;
    font-weight: 800;
}

.skill-box {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 15px;
    padding: 15px;
    margin-top: 10px;
}

.result-box {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(99,102,241,0.25);
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 50px;
    padding-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD NLTK
# =========================================================

stop_words = set(stopwords.words("english"))


# =========================================================
# LOAD MODEL COMPONENTS
# =========================================================

with open("results/tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)

with open("results/classifier.pkl", "rb") as file:
    classifier = pickle.load(file)

with open("results/required_skills.pkl", "rb") as file:
    required_skills = pickle.load(file)


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)


# =========================================================
# SKILL MATCHING
# =========================================================

def calculate_skill_match(resume_text, skills):

    matched_skills = []

    for skill in skills:

        if skill.lower() in resume_text.lower():
            matched_skills.append(skill)

    percentage = (
        len(matched_skills) / len(skills)
    ) * 100

    return percentage, matched_skills


# =========================================================
# RESUME SCREENING
# =========================================================

def screen_resume(resume_text, job_description):

    clean_resume = preprocess_text(resume_text)

    clean_job = preprocess_text(job_description)

    resume_vector = tfidf_vectorizer.transform(
        [clean_resume]
    )

    job_vector = tfidf_vectorizer.transform(
        [clean_job]
    )

    relevance_score = cosine_similarity(
        job_vector,
        resume_vector
    )[0][0] * 100

    skill_percentage, matched_skills = (
        calculate_skill_match(
            clean_resume,
            required_skills
        )
    )

    final_score = (
        0.6 * relevance_score
        + 0.4 * skill_percentage
    )

    predicted_category = classifier.predict(
        resume_vector
    )[0]

    return (
        relevance_score,
        skill_percentage,
        final_score,
        matched_skills,
        predicted_category
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 Intelligent Resume Screening</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered candidate evaluation using NLP, TF-IDF, '
    'skill matching and machine learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUT SECTION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="section-title">💼 Job Description</div>',
        unsafe_allow_html=True
    )

    job_description = st.text_area(
        "Job Description",
        height=300,
        placeholder=(
            "Enter the job requirements, skills, "
            "qualifications and responsibilities..."
        ),
        label_visibility="collapsed"
    )


with col2:

    st.markdown(
        '<div class="section-title">📄 Candidate Resume</div>',
        unsafe_allow_html=True
    )

    resume_text = st.text_area(
        "Candidate Resume",
        height=300,
        placeholder=(
            "Paste the candidate's resume text here..."
        ),
        label_visibility="collapsed"
    )


# =========================================================
# SCREEN BUTTON
# =========================================================

st.write("")

if st.button(
    "🚀 Analyze Candidate",
    use_container_width=True
):

    if not job_description.strip():

        st.warning(
            "Please enter a job description."
        )

    elif not resume_text.strip():

        st.warning(
            "Please enter the candidate resume."
        )

    else:

        (
            relevance,
            skill_match,
            final_score,
            matched_skills,
            predicted_category
        ) = screen_resume(
            resume_text,
            job_description
        )

        st.success(
            "AI screening completed successfully!"
        )

        # =================================================
        # SCORE CARDS
        # =================================================

        st.markdown(
            '<div class="section-title">📊 Candidate Analysis</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        Resume Relevance
                    </div>
                    <div class="score-value">
                        {relevance:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        Skill Match
                    </div>
                    <div class="score-value">
                        {skill_match:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        Final Score
                    </div>
                    <div class="score-value">
                        {final_score:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # FINAL SCORE PROGRESS
        # =================================================

        st.write("")

        st.progress(
            min(final_score / 100, 1.0)
        )


        # =================================================
        # SUITABILITY
        # =================================================

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 🏆 Candidate Suitability"
        )

        if final_score >= 70:

            st.success(
                "Strong Match — Candidate is highly relevant."
            )

        elif final_score >= 40:

            st.info(
                "Moderate Match — Candidate may require further review."
            )

        else:

            st.warning(
                "Low Match — Candidate has limited relevance."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # MATCHED SKILLS
        # =================================================

        st.markdown(
            '<div class="section-title">🧠 Matched Skills</div>',
            unsafe_allow_html=True
        )

        if matched_skills:

            skills_html = ""

            for skill in matched_skills:

                skills_html += (
                    f'<span style="'
                    f'background:#1e293b;'
                    f'border:1px solid #6366f1;'
                    f'border-radius:20px;'
                    f'padding:8px 14px;'
                    f'margin:4px;'
                    f'display:inline-block;'
                    f'color:#e2e8f0;'
                    f'">'
                    f'✓ {skill}'
                    f'</span>'
                )

            st.markdown(
                f'<div class="skill-box">{skills_html}</div>',
                unsafe_allow_html=True
            )

        else:

            st.info(
                "No required skills were detected."
            )


        # =================================================
        # PREDICTED CATEGORY
        # =================================================

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 🎯 Predicted Resume Category"
        )

        st.info(
            predicted_category
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Intelligent Resume Screening System •
        AI & NLP Based Candidate Evaluation
    </div>
    """,
    unsafe_allow_html=True
)

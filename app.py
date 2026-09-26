import streamlit as st
import sqlite3
import os
from pypdf import PdfReader
import re
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# ==========================================
# CV TEXT EXTRACTION FUNCTIONS
# ==========================================

def extract_text_from_pdf(uploaded_file):

    text = ""

    reader = PdfReader(uploaded_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_text_from_docx(uploaded_file):

    document = Document(uploaded_file)

    text = ""

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            text += paragraph.text + "\n"

    return text


def extract_cv_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        return extract_text_from_pdf(uploaded_file)

    elif file_name.endswith(".docx"):

        return extract_text_from_docx(uploaded_file)

    else:

        return ""


# ==========================================
# AI SEMANTIC MODEL
# ==========================================

@st.cache_resource
def load_semantic_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


semantic_model = load_semantic_model()

# ==========================================
# SEMANTIC SIMILARITY FUNCTION
# ==========================================

def calculate_semantic_similarity(text1, text2):

    if not text1.strip() or not text2.strip():

        return 0

    embedding1 = semantic_model.encode(
        text1,
        convert_to_numpy=True
    )

    embedding2 = semantic_model.encode(
        text2,
        convert_to_numpy=True
    )

    similarity = cosine_similarity(
        [embedding1],
        [embedding2]
    )[0][0]

    score = round(
        float(similarity) * 100,
        2
    )

    return score

# ==========================================
# JD CATEGORY EXTRACTION
# ==========================================

def extract_jd_categories(jd_text):

    categories = {
        "education": "",
        "experience": "",
        "skills": "",
        "responsibilities": "",
        "rmg_denim": ""
    }

    if not jd_text:
        return categories

    # --------------------------------------
    # NORMALIZE JD TEXT
    # --------------------------------------

    text = jd_text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = text.splitlines()

    # --------------------------------------
    # SECTION CONTROL
    # --------------------------------------

    current_section = None

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        # Remove markdown / bullet symbols
        clean_line = clean_line.lstrip(
            ">*•●▪◦➢➤✓✔-"
        ).strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        # ==================================
        # STANDARD RESPONSIBILITY HEADING
        # ==================================

        if (
            "job/key responsibilities" in lower_line
            or "job / key responsibilities" in lower_line
            or "job responsibilities" in lower_line
            or "key responsibilities" in lower_line
            or "job responsibility" in lower_line
            or "responsibilities" in lower_line
            or "major responsibilities" in lower_line
        ):

            current_section = "responsibilities"
            continue

        # ==================================
        # STANDARD REQUIREMENTS HEADING
        # ==================================

        if (
            lower_line.startswith("requirements")
            or lower_line == "requirement"
            or lower_line.startswith("job requirements")
            or lower_line.startswith("qualification")
            or lower_line.startswith("qualifications")
        ):

            current_section = "requirements"
            continue

        # ==================================
        # OTHER POSSIBLE HEADINGS
        # ==================================

        if (
            "education" in lower_line
            or "educational qualification" in lower_line
            or "academic qualification" in lower_line
        ) and len(clean_line) < 100:

            current_section = "education"
            continue

        if (
            "experience" in lower_line
            and len(clean_line) < 100
        ):

            current_section = "experience"
            continue

        if (
            lower_line == "skills"
            or lower_line.startswith("skills:")
            or lower_line == "technical skills"
            or lower_line.startswith("technical skills:")
        ):

            current_section = "skills"
            continue

        # ==================================
        # REMOVE NUMBERING
        # ==================================

        clean_line = re.sub(
            r"^\s*\d+\s*[\.\)\:\-]\s*",
            "",
            clean_line
        ).strip()

        # ==================================
        # JOB RESPONSIBILITIES SECTION
        # ==================================

        if current_section == "responsibilities":

            if len(clean_line) >= 8:

                categories["responsibilities"] += (
                    clean_line + " "
                )

            continue

        # ==================================
        # REQUIREMENTS SECTION
        # ==================================

        if current_section == "requirements":

            lower_clean = clean_line.lower()

            # --------------------------------
            # EDUCATION
            # --------------------------------

            if any(keyword in lower_clean for keyword in [
                "bachelor",
                "master",
                "degree",
                "diploma",
                "university",
                "education",
                "educational"
            ]):

                categories["education"] += (
                    clean_line + " "
                )

            # --------------------------------
            # EXPERIENCE
            # --------------------------------

            if (
                "year" in lower_clean
                or "years" in lower_clean
                or "experience" in lower_clean
                or "experienced" in lower_clean
            ):

                categories["experience"] += (
                    clean_line + " "
                )

            # --------------------------------
            # SKILLS / KNOWLEDGE
            # --------------------------------

            if any(keyword in lower_clean for keyword in [
                "skill",
                "skills",
                "knowledge",
                "erp",
                "plm",
                "software",
                "communication",
                "coordination",
                "analytical",
                "leadership",
                "negotiation",
                "planning",
                "problem solving",
                "problem-solving",
                "teamwork",
                "team work",
                "buyer",
                "costing",
                "follow-up",
                "follow up"
            ]):

                categories["skills"] += (
                    clean_line + " "
                )

            continue

        # ==================================
        # EXPLICIT EDUCATION SECTION
        # ==================================

        if current_section == "education":

            categories["education"] += (
                clean_line + " "
            )

            continue

        # ==================================
        # EXPLICIT EXPERIENCE SECTION
        # ==================================

        if current_section == "experience":

            categories["experience"] += (
                clean_line + " "
            )

            continue

        # ==================================
        # EXPLICIT SKILLS SECTION
        # ==================================

        if current_section == "skills":

            categories["skills"] += (
                clean_line + " "
            )

            continue

    # ======================================
    # FALLBACK EDUCATION EXTRACTION
    # ======================================

    if not categories["education"]:

        education_lines = []

        for line in lines:

            lower_line = line.lower()

            if any(keyword in lower_line for keyword in [
                "bachelor",
                "master",
                "degree",
                "diploma",
                "university"
            ]):

                clean_line = line.strip()

                if clean_line:

                    education_lines.append(
                        clean_line
                    )

        categories["education"] = " ".join(
            education_lines
        )

    # ======================================
    # FALLBACK EXPERIENCE EXTRACTION
    # ======================================

    if not categories["experience"]:

        experience_lines = []

        for line in lines:

            lower_line = line.lower()

            if (
                "experience" in lower_line
                or re.search(
                    r"\b\d+\s*[-–]?\s*\d*\s*years?\b",
                    lower_line
                )
            ):

                clean_line = line.strip()

                if clean_line:

                    experience_lines.append(
                        clean_line
                    )

        categories["experience"] = " ".join(
            experience_lines
        )

    # ======================================
    # FALLBACK SKILLS EXTRACTION
    # ======================================

    if not categories["skills"]:

        skill_lines = []

        skill_keywords = [
            "knowledge",
            "skill",
            "erp",
            "plm",
            "communication",
            "coordination",
            "analytical",
            "leadership",
            "negotiation",
            "planning",
            "teamwork",
            "buyer",
            "costing",
            "production",
            "software"
        ]

        for line in lines:

            lower_line = line.lower()

            if any(
                keyword in lower_line
                for keyword in skill_keywords
            ):

                clean_line = line.strip()

                if clean_line:

                    skill_lines.append(
                        clean_line
                    )

        categories["skills"] = " ".join(
            skill_lines
        )

    # ======================================
    # RMG / DENIM / INDUSTRY EXTRACTION
    # ======================================

    rmg_keywords = [
        "rmg",
        "garment",
        "garments",
        "apparel",
        "denim",
        "woven",
        "knit",
        "textile",
        "fashion",
        "washing",
        "merchandising",
        "production",
        "factory",
        "buyer",
        "buying house"
    ]

    rmg_lines = []

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        if any(
            keyword in lower_line
            for keyword in rmg_keywords
        ):

            clean_line = clean_line.lstrip(
                ">*•●▪◦➢➤✓✔-"
            ).strip()

            if clean_line not in rmg_lines:

                rmg_lines.append(
                    clean_line
                )

    categories["rmg_denim"] = " ".join(
        rmg_lines
    )

    # ======================================
    # CLEAN CATEGORY TEXT
    # ======================================

    for category in categories:

        categories[category] = (
            categories[category]
            .strip()
        )

    return categories


# ==========================================
# CATEGORY-WISE AI MATCHING
# ==========================================

def calculate_category_scores(jd_text, cv_text):

    scores = {
        "education": 0.0,
        "experience": 0.0,
        "skills": 0.0,
        "responsibilities": 0.0,
        "rmg_denim": 0.0
    }

    if not jd_text or not cv_text:

        return scores

    # ======================================
    # GET JD CATEGORIES
    # ======================================

    categories = extract_jd_categories(
        jd_text
    )

    # ======================================
    # PREPARE CV LINES
    # ======================================

    cv_lines = []

    for line in cv_text.splitlines():

        clean_line = line.strip()

        if not clean_line:
            continue

        clean_line = clean_line.lstrip(
            ">*•●▪◦➢➤✓✔-"
        ).strip()

        if len(clean_line) >= 8:

            cv_lines.append(
                clean_line
            )

    # ======================================
    # CV FALLBACK
    # ======================================

    if len(cv_lines) < 3:

        cv_lines = re.split(
            r"[.;]\s+",
            cv_text
        )

        cv_lines = [
            line.strip()
            for line in cv_lines
            if len(line.strip()) >= 8
        ]

    # ======================================
    # HELPER FUNCTION
    # ======================================

    def best_category_match(
        jd_category_text,
        candidate_lines
    ):

        if not jd_category_text:
            return 0.0

        if not candidate_lines:
            return 0.0

        best_score = 0.0

        # ----------------------------------
        # BEST CV LINE MATCH
        # ----------------------------------

        for cv_line in candidate_lines:

            try:

                score = float(
                    calculate_semantic_similarity(
                        jd_category_text,
                        cv_line
                    )
                )

                if score > best_score:

                    best_score = score

            except Exception:

                continue

        # ----------------------------------
        # COMPLETE CV MATCH
        # ----------------------------------

        try:

            full_score = float(
                calculate_semantic_similarity(
                    jd_category_text,
                    cv_text
                )
            )

            if full_score > best_score:

                best_score = full_score

        except Exception:

            pass

        return max(
            0.0,
            min(
                100.0,
                best_score
            )
        )

    # ======================================
    # EDUCATION
    # ======================================

    if categories["education"]:

        scores["education"] = round(
            best_category_match(
                categories["education"],
                cv_lines
            ),
            2
        )

    # ======================================
    # EXPERIENCE
    # ======================================

    if categories["experience"]:

        scores["experience"] = round(
            best_category_match(
                categories["experience"],
                cv_lines
            ),
            2
        )

    # ======================================
    # SKILLS
    # ======================================

    if categories["skills"]:

        scores["skills"] = round(
            best_category_match(
                categories["skills"],
                cv_lines
            ),
            2
        )

    # ======================================
    # RESPONSIBILITIES
    # ======================================

    responsibility_text = categories[
        "responsibilities"
    ]

    if responsibility_text:

        # ----------------------------------
        # Split JD into individual tasks
        # ----------------------------------

        jd_tasks = re.split(
            r"(?<=[.!?])\s+",
            responsibility_text
        )

        jd_tasks = [
            task.strip()
            for task in jd_tasks
            if len(task.strip()) >= 15
        ]

        # ----------------------------------
        # If one long block exists,
        # use comma / semicolon segments
        # ----------------------------------

        if len(jd_tasks) <= 1:

            jd_tasks = re.split(
                r"[;,]\s+",
                responsibility_text
            )

            jd_tasks = [
                task.strip()
                for task in jd_tasks
                if len(task.strip()) >= 15
            ]

        if not jd_tasks:

            jd_tasks = [
                responsibility_text
            ]

        # ----------------------------------
        # MATCH EACH JD TASK
        # ----------------------------------

        task_scores = []

        for jd_task in jd_tasks:

            best_task_score = 0.0

            for cv_line in cv_lines:

                try:

                    score = float(
                        calculate_semantic_similarity(
                            jd_task,
                            cv_line
                        )
                    )

                    if score > best_task_score:

                        best_task_score = score

                except Exception:

                    continue

            task_scores.append(
                best_task_score
            )

        # ----------------------------------
        # RESPONSIBILITY FINAL MATCH
        # ----------------------------------

        if task_scores:

            scores["responsibilities"] = round(
                (
                    sum(task_scores)
                    / len(task_scores)
                ),
                2
            )

    # ======================================
    # RMG / DENIM
    # ======================================

    if categories["rmg_denim"]:

        rmg_cv_lines = []

        for line in cv_lines:

            lower_line = line.lower()

            if any(
                keyword in lower_line
                for keyword in [
                    "rmg",
                    "garment",
                    "garments",
                    "apparel",
                    "denim",
                    "woven",
                    "knit",
                    "textile",
                    "fashion",
                    "washing",
                    "merchandising",
                    "buyer",
                    "production"
                ]
            ):

                rmg_cv_lines.append(
                    line
                )

        if not rmg_cv_lines:

            rmg_cv_lines = cv_lines

        scores["rmg_denim"] = round(
            best_category_match(
                categories["rmg_denim"],
                rmg_cv_lines
            ),
            2
        )

    return scores
    # --------------------------------------
    # SPLIT JD RESPONSIBILITIES INTO TASKS
    # --------------------------------------

    jd_lines = responsibility_text.splitlines()

    jd_tasks = []

    for line in jd_lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        # Remove common bullet / numbering
        clean_line = clean_line.lstrip(
            "•-✓✔▪◦➢➤"
        ).strip()

        clean_line = re.sub(
            r"^\s*\d+[\.\)\-:]\s*",
            "",
            clean_line
        ).strip()

        if len(clean_line) < 10:
            continue

        jd_tasks.append(clean_line)

    # --------------------------------------
    # IF JD RESPONSIBILITIES ARE ONE BLOCK
    # --------------------------------------

    if len(jd_tasks) <= 1:

        jd_tasks = re.split(
            r"(?<=[.!?])\s+",
            responsibility_text
        )

        jd_tasks = [
            item.strip()
            for item in jd_tasks
            if len(item.strip()) >= 10
        ]

    # --------------------------------------
    # CV WORK / RESPONSIBILITY TEXT
    # --------------------------------------

    cv_lines = cv_text.splitlines()

    cv_tasks = []

    capture_cv = False

    cv_start_keywords = [
        "work experience",
        "professional experience",
        "employment history",
        "career history",
        "responsibilities",
        "job responsibilities",
        "key responsibilities",
        "major responsibilities",
        "roles and responsibilities",
        "role & responsibilities",
        "role and responsibilities",
        "duties",
        "key duties",
        "major duties",
        "job profile"
    ]

    cv_stop_keywords = [
        "education",
        "educational qualification",
        "academic qualification",
        "academic background",
        "skills",
        "technical skills",
        "professional skills",
        "personal information",
        "personal details",
        "reference",
        "references",
        "training",
        "trainings",
        "certification",
        "certifications",
        "achievement",
        "achievements",
        "language",
        "languages",
        "declaration"
    ]

    task_keywords = [
        # General HR / Business Tasks
        "manage",
        "managed",
        "handling",
        "handle",
        "handled",
        "coordinate",
        "coordinated",
        "follow",
        "followed",
        "following",
        "prepare",
        "prepared",
        "maintain",
        "maintained",
        "develop",
        "developed",
        "support",
        "supported",
        "assist",
        "assisted",
        "monitor",
        "monitored",
        "control",
        "controlled",
        "communicate",
        "communication",
        "liaise",
        "liaison",
        "negotiate",
        "negotiation",
        "source",
        "sourcing",
        "track",
        "tracking",
        "arrange",
        "arranged",
        "review",
        "reviewed",
        "analyze",
        "analysis",
        "process",
        "processed",
        "submit",
        "submitted",
        "ensure",
        "plan",
        "planned",
        "execute",
        "executed",
        "report",
        "reporting",

        # Merchandising / RMG
        "buyer",
        "buying house",
        "production",
        "sample",
        "sampling",
        "sample development",
        "costing",
        "budget",
        "budgeting",
        "erp",
        "tna",
        "trims",
        "accessories",
        "material sourcing",
        "supplier",
        "commercial",
        "merchandising",
        "order",
        "shipment",
        "quality",
        "wash",
        "washing",
        "knitting",
        "dyeing",
        "denim",
        "woven",
        "garments",
        "fabric",
        "lab-dip",
        "lab dip",
        "strike-off",
        "strike off",
        "tech pack",
        "purchase",
        "planning",
        "follow-up",
        "follow up",
        "testing",
        "compliance"
    ]

    for line in cv_lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        # Start CV experience section
        if any(
            keyword in lower_line
            for keyword in cv_start_keywords
        ):

            capture_cv = True
            continue

        # Stop at unrelated sections
        if capture_cv and any(
            keyword in lower_line
            for keyword in cv_stop_keywords
        ):

            capture_cv = False
            continue

        if not capture_cv:
            continue

        clean_line = clean_line.lstrip(
            "•-✓✔▪◦➢➤"
        ).strip()

        clean_line = re.sub(
            r"^\s*\d+[\.\)\-:]\s*",
            "",
            clean_line
        ).strip()

        if len(clean_line) < 8:
            continue

        lower_clean = clean_line.lower()

        # Keep only lines that contain
        # meaningful work/task evidence
        if any(
            keyword in lower_clean
            for keyword in task_keywords
        ):

            cv_tasks.append(clean_line)

    # --------------------------------------
    # REMOVE DUPLICATE CV TASKS
    # --------------------------------------

    unique_cv_tasks = []

    for task in cv_tasks:

        if task not in unique_cv_tasks:

            unique_cv_tasks.append(task)

    cv_tasks = unique_cv_tasks

    # --------------------------------------
    # FALLBACK:
    # IF CV SECTION EXTRACTION IS WEAK
    # SEARCH ENTIRE CV FOR TASK EVIDENCE
    # --------------------------------------

    if len(cv_tasks) < 2:

        cv_tasks = []

        for line in cv_lines:

            clean_line = line.strip()

            if len(clean_line) < 8:
                continue

            clean_line = clean_line.lstrip(
                "•-✓✔▪◦➢➤"
            ).strip()

            lower_clean = clean_line.lower()

            if any(
                keyword in lower_clean
                for keyword in task_keywords
            ):

                if clean_line not in cv_tasks:

                    cv_tasks.append(clean_line)

    # --------------------------------------
    # MATCH JD TASKS WITH CV TASKS
    # --------------------------------------

    task_scores = []

    for jd_task in jd_tasks:

        best_score = 0.0

        for cv_task in cv_tasks:

            try:

                score = float(
                    calculate_semantic_similarity(
                        jd_task,
                        cv_task
                    )
                )

            except Exception:

                score = 0.0

            if score > best_score:

                best_score = score

        task_scores.append(
            best_score
        )

    # --------------------------------------
    # RESPONSIBILITY MATCH SCORE
    # --------------------------------------

    if task_scores:

        responsibility_score = (
            sum(task_scores) /
            len(task_scores)
        )

    else:

        # Fallback to old category matching
        responsibility_score = calculate_semantic_similarity(
            responsibility_text,
            cv_text
        )

    scores["responsibilities"] = round(
        float(responsibility_score),
        2
    )

    return scores
    scores["responsibilities"] = round(
        float(responsibility_score),
        2
    )

    return scores


# ==========================================
# WEIGHTED AI SCORE
# ==========================================

def calculate_weighted_score(category_scores):

    weights = {
        "education": 15,
        "experience": 25,
        "skills": 25,
        "responsibilities": 20,
        "rmg_denim": 15
    }

    total_score = 0

    for category, weight in weights.items():

        category_score = category_scores.get(
            category,
            0
        )

        weighted_score = (
            category_score * weight / 100
        )

        total_score += weighted_score

    return round(total_score, 2)

# ==========================================
# JD TASK / RESPONSIBILITY EXTRACTION
# ==========================================

def extract_jd_tasks(jd_text):

    if not jd_text:
        return []

    lines = jd_text.splitlines()

    tasks = []

    capture = False

    # Sections that usually contain JD responsibilities
    start_keywords = [
        "job description",
        "job responsibilities",
        "key responsibilities",
        "major responsibilities",
        "responsibilities",
        "roles and responsibilities",
        "role & responsibilities",
        "role and responsibilities",
        "duties",
        "key duties",
        "major duties",
        "experience on",
        "major tasks",
        "key tasks"
    ]

    # Sections that usually come after responsibilities
    stop_keywords = [
        "requirements",
        "additional requirements",
        "educational qualification",
        "education qualification",
        "academic qualification",
        "academic background",
        "education",
        "experience required",
        "experience",
        "salary",
        "benefits",
        "gender",
        "location",
        "job location",
        "apply",
        "application",
        "deadline",
        "how to apply"
    ]

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        # Remove common bullet characters
        clean_line = clean_line.lstrip(
            "•-✓✔▪◦➢➤●○"
        ).strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower().strip()

        # --------------------------------------
        # START RESPONSIBILITY SECTION
        # --------------------------------------

        if any(
            keyword in lower_line
            for keyword in start_keywords
        ):

            capture = True

            # Sometimes the task is on the SAME line
            # after the heading.
            for separator in [":", ".", "–", "-"]:

                if separator in clean_line:

                    parts = clean_line.split(
                        separator,
                        1
                    )

                    if len(parts) == 2:

                        possible_task = parts[1].strip()

                        if len(possible_task) >= 10:

                            tasks.append(
                                possible_task
                            )

                    break

            continue

        # --------------------------------------
        # STOP RESPONSIBILITY SECTION
        # --------------------------------------

        if any(
            keyword in lower_line
            for keyword in stop_keywords
        ):

            capture = False
            continue

        # --------------------------------------
        # COLLECT JD TASKS
        # --------------------------------------

        if capture:

            # Remove numbering:
            # 1.
            # 2)
            # 3 -
            # etc.
            import re

            clean_line = re.sub(
                r"^\s*\d+\s*[\.\)\-:]\s*",
                "",
                clean_line
            ).strip()

            if len(clean_line) < 10:
                continue

            # Ignore obvious headings
            if clean_line.isupper() and len(clean_line) < 80:
                continue

            tasks.append(clean_line)

    # --------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------

    unique_tasks = []

    seen = set()

    for task in tasks:

        task_key = task.lower().strip()

        if task_key not in seen:

            seen.add(task_key)

            unique_tasks.append(task)

    return unique_tasks


# ==========================================
# CV TASK / RESPONSIBILITY EXTRACTION
# ==========================================

def extract_cv_tasks(cv_text):

    if not cv_text:
        return []

    lines = cv_text.splitlines()

    tasks = []

    capture = False

    import re

    # --------------------------------------
    # Sections where work responsibilities
    # are normally found
    # --------------------------------------

    start_keywords = [

        "work experience",
        "work experiences",
        "professional experience",
        "professional experiences",
        "employment history",
        "career history",
        "career experience",

        "responsibilities",
        "job responsibilities",
        "key responsibilities",
        "major responsibilities",

        "roles and responsibilities",
        "role & responsibilities",
        "role and responsibilities",

        "duties",
        "key duties",
        "major duties",

        "job description",
        "job profile",

        "work responsibilities"
    ]

    # --------------------------------------
    # Sections that normally end experience
    # --------------------------------------

    stop_keywords = [

        "educational qualification",
        "education qualification",
        "academic qualification",
        "academic background",
        "education",

        "technical skills",
        "professional skills",
        "skills",
        "core skills",
        "key skills",

        "personal information",
        "personal details",

        "reference",
        "references",

        "training",
        "trainings",

        "certification",
        "certifications",

        "achievement",
        "achievements",

        "language",
        "languages",

        "career objective",
        "objective",

        "declaration",

        "contact information",
        "contact details",

        "interests",
        "interest"
    ]

    # --------------------------------------
    # Words that strongly indicate a task
    # --------------------------------------

    action_words = [

        "manage",
        "managed",
        "managing",

        "handle",
        "handled",
        "handling",

        "coordinate",
        "coordinated",
        "coordinating",

        "follow",
        "followed",
        "following",

        "prepare",
        "prepared",
        "preparing",

        "maintain",
        "maintained",
        "maintaining",

        "develop",
        "developed",
        "developing",

        "support",
        "supported",
        "supporting",

        "assist",
        "assisted",
        "assisting",

        "monitor",
        "monitored",
        "monitoring",

        "supervise",
        "supervised",
        "supervising",

        "control",
        "controlled",
        "controlling",

        "coordinate",
        "coordinated",

        "communicate",
        "communicated",
        "communicating",

        "liaise",
        "liaised",
        "liaison",

        "negotiate",
        "negotiated",
        "negotiation",

        "source",
        "sourced",
        "sourcing",

        "track",
        "tracked",
        "tracking",

        "arrange",
        "arranged",
        "arranging",

        "review",
        "reviewed",
        "reviewing",

        "analyze",
        "analysed",
        "analyzed",
        "analysis",

        "process",
        "processed",
        "processing",

        "submit",
        "submitted",
        "submitting",

        "ensure",
        "ensured",
        "ensuring",

        "plan",
        "planned",
        "planning",

        "execute",
        "executed",
        "execution",

        "report",
        "reported",
        "reporting",

        "liaison",

        "cost",
        "costing",

        "budget",
        "budgeting"
    ]

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        # Remove bullet symbols
        clean_line = clean_line.lstrip(
            "•-✓✔▪◦➢➤●○"
        ).strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower().strip()

        # --------------------------------------
        # START EXPERIENCE SECTION
        # --------------------------------------

        if any(
            keyword in lower_line
            for keyword in start_keywords
        ):

            capture = True
            continue

        # --------------------------------------
        # STOP EXPERIENCE SECTION
        # --------------------------------------

        if any(
            keyword in lower_line
            for keyword in stop_keywords
        ):

            capture = False
            continue

        if not capture:
            continue

        # --------------------------------------
        # Remove dates
        # --------------------------------------

        if re.search(
            r"\b(19|20)\d{2}\b",
            clean_line
        ):

            # Do not immediately discard the line
            # because some task lines may contain dates.
            if len(clean_line) < 35:
                continue

        # --------------------------------------
        # Ignore phone/email/web/address lines
        # --------------------------------------

        if "@" in clean_line:
            continue

        if "www." in lower_line:
            continue

        if "http://" in lower_line:
            continue

        if "https://" in lower_line:
            continue

        # --------------------------------------
        # Ignore very short lines
        # --------------------------------------

        if len(clean_line) < 8:
            continue

        # --------------------------------------
        # Remove numbering
        # --------------------------------------

        clean_line = re.sub(
            r"^\s*\d+\s*[\.\)\-:]\s*",
            "",
            clean_line
        ).strip()

        # --------------------------------------
        # Ignore obvious job-title/company lines
        # --------------------------------------

        if len(clean_line) < 10:
            continue

        # --------------------------------------
        # Detect whether line looks like a task
        # --------------------------------------

        line_is_task = False

        # 1. If it contains an action word
        for action_word in action_words:

            if re.search(
                r"\b"
                + re.escape(action_word)
                + r"\b",
                lower_line
            ):

                line_is_task = True
                break

        # --------------------------------------
        # 2. Common merchandising / RMG tasks
        # --------------------------------------

        task_keywords = [

            "buyer",
            "buying house",
            "production",
            "sample",
            "costing",
            "erp",
            "tna",
            "trims",
            "accessories",
            "material sourcing",
            "supplier",
            "commercial",
            "merchandising",
            "merchandiser",
            "order",
            "shipment",
            "quality",
            "wash",
            "washing",
            "knitting",
            "dyeing",
            "denim",
            "woven",
            "garments",
            "fabric",
            "lab-dip",
            "lab dip",
            "strike-off",
            "strike off",
            "tech pack",
            "purchase",
            "planning",
            "follow-up",
            "follow up",
            "communication",
            "negotiation",
            "budget",
            "report",
            "testing",
            "compliance"
        ]

        if any(
            keyword in lower_line
            for keyword in task_keywords
        ):

            line_is_task = True

        # --------------------------------------
        # Add task
        # --------------------------------------

        if line_is_task:

            tasks.append(
                clean_line
            )

    # --------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------

    unique_tasks = []

    seen = set()

    for task in tasks:

        task_key = task.lower().strip()

        if task_key not in seen:

            seen.add(task_key)

            unique_tasks.append(task)

    return unique_tasks


# ==========================================
# JD TASK ↔ CV TASK SEMANTIC MATCHING
# ==========================================

def calculate_task_matching(jd_text, cv_text):

    jd_tasks = extract_jd_tasks(jd_text)

    cv_tasks = extract_cv_tasks(cv_text)
    
    st.write("### 🔍 TASK DEBUG")
    st.write("JD TASKS:", jd_tasks)
    st.write("CV TASKS:", cv_tasks)

    results = []


    # --------------------------------------
    # If JD tasks are unavailable
    # --------------------------------------

    if not jd_tasks:

        return results

    # --------------------------------------
    # Compare every JD task with CV tasks
    # --------------------------------------

    for jd_task in jd_tasks:

        best_score = 0

        best_cv_task = ""

        for cv_task in cv_tasks:

            try:

                score = calculate_semantic_similarity(
                    jd_task,
                    cv_task
                )

                score = float(score)

            except Exception:

                score = 0

            if score > best_score:

                best_score = score

                best_cv_task = cv_task

        # ----------------------------------
        # Determine result status
        # ----------------------------------

        if not best_cv_task:

            status = "No Evidence"

            icon = "🔴"

        elif best_score >= 70:

            status = "Strong Match"

            icon = "🟢"

        elif best_score >= 45:

            status = "Partial Match"

            icon = "🟡"

        else:

            status = "No Evidence"

            icon = "🔴"

        results.append({

            "jd_task": jd_task,

            "cv_task": best_cv_task,

            "score": round(
                best_score,
                2
            ),

            "status": status,

            "icon": icon

        })

    return results


# ==========================================
# TASK MATCH PERCENTAGE
# ==========================================

def calculate_task_match_percentage(
    task_results
):

    if not task_results:

        return 0

    total_score = 0

    valid_results = 0

    for item in task_results:

        # No Evidence should not falsely
        # increase the task match percentage
        if item["status"] == "No Evidence":

            continue

        total_score += float(
            item.get("score", 0)
        )

        valid_results += 1

    if valid_results == 0:

        return 0

    task_match = (
        total_score /
        valid_results
    )

    return round(
        task_match,
        2
    )
# ==========================================
# IMPROVED CANDIDATE NAME EXTRACTION
# ==========================================

def extract_candidate_name(cv_text, file_name=""):

    # ======================================
    # STEP 1: TRY TO GET NAME FROM FILE NAME
    # ======================================

    if file_name:

        file_name_only = file_name

        # Remove file extension
        if "." in file_name_only:

            file_name_only = file_name_only.rsplit(
                ".",
                1
            )[0]

        # Replace separators
        file_name_only = file_name_only.replace(
            "_",
            " "
        ).replace(
            "-",
            " "
        )

        # Remove common CV words
        filename_words = file_name_only.split()

        remove_words = [
            "resume",
            "cv",
            "curriculum",
            "vitae",
            "updated",
            "final",
            "latest",
            "new",
            "copy"
        ]

        filtered_words = []

        for word in filename_words:

            if word.lower() not in remove_words:

                filtered_words.append(word)

        filename_name = " ".join(
            filtered_words
        ).strip()

        # ----------------------------------
        # Check whether filename looks like
        # a person's name
        # ----------------------------------

        if 2 <= len(filename_name.split()) <= 5:

            filename_lower = filename_name.lower()

            invalid_filename_words = [
                "institute",
                "institution",
                "university",
                "college",
                "school",
                "academy",
                "technology",
                "company",
                "limited",
                "ltd",
                "group",
                "factory",
                "hospital",
                "address",
                "dhaka",
                "savar",
                "gazipur",
                "narayanganj",
                "bangladesh"
            ]

            if not any(
                word in filename_lower
                for word in invalid_filename_words
            ):

                return filename_name.title()

    # ======================================
    # STEP 2: SEARCH FOR EXPLICIT NAME FIELD
    # ======================================

    lines = cv_text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        line = line.lstrip(
            "•-✓✔▪◦|:"
        ).strip()

        if line:

            cleaned_lines.append(line)

    # Look for:
    # Name: Muhammad Al Amin
    # Name - Muhammad Al Amin
    # Candidate Name: Muhammad Al Amin

    for line in cleaned_lines[:40]:

        lower_line = line.lower()

        if (
            lower_line.startswith("name:")
            or lower_line.startswith("name -")
            or lower_line.startswith("name–")
            or lower_line.startswith("candidate name:")
            or lower_line.startswith("candidate name -")
        ):

            name_part = line.split(
                ":",
                1
            )

            if len(name_part) == 2:

                possible_name = name_part[1].strip()

                if 2 <= len(
                    possible_name.split()
                ) <= 5:

                    return possible_name

    # ======================================
    # STEP 3: SEARCH CV HEADER
    # ======================================

    excluded_words = [
        "curriculum vitae",
        "resume",
        "cv",
        "curriculum",
        "profile",
        "career objective",
        "professional summary",
        "personal information",
        "contact information",
        "objective",
        "education",
        "educational qualification",
        "experience",
        "work experience",
        "professional experience",
        "skills",
        "technical skills",
        "responsibilities",
        "references",
        "declaration",
        "address",
        "present address",
        "permanent address",
        "mobile",
        "phone",
        "email",
        "date of birth",
        "nationality",
        "religion",
        "gender",
        "marital status"
    ]

    invalid_context_words = [
        "institute",
        "institution",
        "university",
        "college",
        "school",
        "academy",
        "technology",
        "company",
        "limited",
        "ltd",
        "group",
        "factory",
        "hospital",
        "address",
        "road",
        "street",
        "avenue",
        "lane",
        "house",
        "flat",
        "floor",
        "village",
        "post office",
        "postal",
        "dhaka",
        "savar",
        "gazipur",
        "narayanganj",
        "chattogram",
        "chittagong",
        "bangladesh",
        "freeport",
        "linkedin"
    ]

    possible_names = []

    # Only inspect the first 20 lines
    # because candidate name is normally
    # near the top of the CV

    for line in cleaned_lines[:20]:

        lower_line = line.lower()

        # Skip headings
        if lower_line in excluded_words:
            continue

        # Skip email
        if "@" in line:
            continue

        # Skip website
        if "www." in lower_line:
            continue

        # Skip URLs
        if "http" in lower_line:
            continue

        # Skip lines containing numbers
        digit_count = sum(
            char.isdigit()
            for char in line
        )

        if digit_count >= 2:
            continue

        # Skip long sentences
        if len(line) > 45:
            continue

        # Skip institute/company/location lines
        if any(
            word in lower_line
            for word in invalid_context_words
        ):
            continue

        words = line.split()

        # Normal name = 2 to 5 words
        if not 2 <= len(words) <= 5:
            continue

        # Skip obvious professional headings
        if any(
            word in lower_line
            for word in [
                "manager",
                "executive",
                "officer",
                "engineer",
                "developer",
                "designer",
                "analyst",
                "consultant",
                "department",
                "experience",
                "education",
                "skills",
                "objective"
            ]
        ):
            continue

        # ----------------------------------
        # Check that words look like names
        # ----------------------------------

        valid_word_count = 0

        for word in words:

            clean_word = word.strip(
                ".,:;()[]{}"
            )

            if clean_word.replace(
                "-",
                ""
            ).isalpha():

                valid_word_count += 1

        if valid_word_count == len(words):

            possible_names.append(line)

    # ======================================
    # STEP 4: RETURN FIRST VALID CV NAME
    # ======================================

    if possible_names:

        return possible_names[0]

    # ======================================
    # STEP 5: NOTHING FOUND
    # ======================================

    return "Candidate Name Not Found"

# ==========================================
# AI SCREENING SUMMARY
# ==========================================

def generate_screening_summary(
    candidate_name,
    final_score,
    recommendation,
    category_scores,
    requirement_analysis
):

    strong_count = 0
    partial_count = 0
    weak_count = 0

    for item in requirement_analysis:

        if item["status"] == "Strong Match":
            strong_count += 1

        elif item["status"] == "Partial Match":
            partial_count += 1

        else:
            weak_count += 1

    if final_score >= 70:

        summary = (
            f"{candidate_name} shows strong overall alignment "
            f"with the selected job requirements. "
            f"The CV has {strong_count} strong matching requirement(s), "
            f"{partial_count} partial match(es), and "
            f"{weak_count} weak or missing requirement(s). "
            f"Further verification can be done during the interview."
        )

    elif final_score >= 40:

        summary = (
            f"{candidate_name} shows partial alignment "
            f"with the selected job requirements. "
            f"The CV has {strong_count} strong matching requirement(s), "
            f"{partial_count} partial match(es), and "
            f"{weak_count} weak or missing requirement(s). "
            f"Additional review is recommended before proceeding."
        )

    else:

        summary = (
            f"{candidate_name} shows limited alignment "
            f"with the selected job requirements. "
            f"The CV has {strong_count} strong matching requirement(s), "
            f"{partial_count} partial match(es), and "
            f"{weak_count} weak or missing requirement(s). "
            f"The HR team should review the CV carefully before taking action."
        )

    return summary
# ==========================================
# EDUCATION REQUIREMENT EXTRACTION
# ==========================================

def extract_education_requirement(jd_text):

    lines = jd_text.splitlines()

    education_lines = []

    capture = False

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        lower_line = clean_line.lower()

        # Start education section
        if any(word in lower_line for word in [
            "educational qualification",
            "education qualification",
            "academic qualification",
            "educational background"
        ]):

            capture = True
            continue

        # Stop when another major section starts
        if capture and any(word in lower_line for word in [
            "experience on",
            "experience",
            "skills",
            "responsibilities",
            "major tasks",
            "job responsibilities"
        ]):

            break

        if capture:

            clean_line = clean_line.lstrip(
                "•-✓✔▪◦"
            ).strip()

            if clean_line:

                education_lines.append(
                    clean_line
                )

    return " ".join(education_lines)
# ==========================================
# EDUCATION MATCHING
# ==========================================

def calculate_education_match(jd_text, cv_text):

    education_requirement = extract_education_requirement(
        jd_text
    )

    if not education_requirement.strip():
        return 0

    if not cv_text.strip():
        return 0

    education_score = calculate_semantic_similarity(
        education_requirement,
        cv_text
    )

    return round(education_score, 2)

# ==========================================
# AI SCREENING SUMMARY
# ==========================================

def generate_screening_summary(
    candidate_name,
    final_score,
    recommendation,
    category_scores,
    requirement_analysis
):

    strong_count = 0
    partial_count = 0
    weak_count = 0

    for item in requirement_analysis:

        if item["status"] == "Strong Match":
            strong_count += 1

        elif item["status"] == "Partial Match":
            partial_count += 1

        else:
            weak_count += 1

    # Overall summary
    if final_score >= 70:

        summary = (
            f"{candidate_name} shows strong overall alignment "
            f"with the selected job requirements. "
            f"The CV has {strong_count} strong matching requirement(s), "
            f"{partial_count} partial match(es), and "
            f"{weak_count} weak or missing requirement(s). "
            f"Further verification can be done during the interview."
        )

    elif final_score >= 40:

        summary = (
            f"{candidate_name} shows partial alignment "
            f"with the selected job requirements. "
            f"The CV has {strong_count} strong matching requirement(s), "
            f"{partial_count} partial match(es), and "
            f"{weak_count} weak or missing requirement(s). "
            f"Additional review is recommended before proceeding."
        )

    else:

        summary = (
            f"{candidate_name} shows limited alignment "
            f"with the selected job requirements. "
            f"The CV has {strong_count} strong matching requirement(s), "
            f"{partial_count} partial match(es), and "
            f"{weak_count} weak or missing requirement(s). "
            f"The HR team should review the CV carefully before taking action."
        )

    return summary

# ==========================================
# AI SCREENING RECOMMENDATION
# ==========================================

def get_screening_recommendation(score):

    if score >= 70:

        return "🟢 Shortlist"

    elif score >= 40:

        return "🟡 Review"

    else:

        return "🔴 Not Recommended"
    
# ==========================================
# SMART JD VS CV MATCHING
# ==========================================

def smart_cv_match(jd_text, cv_text):

    jd_text = jd_text.lower()
    cv_text = cv_text.lower()

    # Common words বাদ দেওয়ার জন্য
    stop_words = {
        "and", "the", "with", "for", "from",
        "in", "on", "of", "to", "a", "an",
        "is", "are", "be", "or", "as",
        "experience", "years", "year"
    }

    # JD থেকে গুরুত্বপূর্ণ শব্দ বের করা
    words = jd_text.replace(",", " ").replace(".", " ").split()

    keywords = []

    for word in words:

        word = word.strip(":-()[]{}")

        if len(word) >= 4 and word not in stop_words:

            if word not in keywords:

                keywords.append(word)

    # CV-এর সাথে keyword matching
    matched_keywords = []
    unmatched_keywords = []

    for keyword in keywords:

        if keyword in cv_text:

            matched_keywords.append(keyword)

        else:

            unmatched_keywords.append(keyword)

    # Score calculation
    total_keywords = len(keywords)

    if total_keywords > 0:

        score = round(
            (len(matched_keywords) / total_keywords) * 100
        )

    else:

        score = 0

    # Recommendation
    if score >= 70:

        decision = "🟢 Shortlist"

    elif score >= 40:

        decision = "🟡 Review"

    else:

        decision = "🔴 Not Recommended"

    return {
        "score": score,
        "decision": decision,
        "matched_keywords": matched_keywords,
        "unmatched_keywords": unmatched_keywords
    }

# ==========================================
# AI JD REQUIREMENT ANALYSIS
# ==========================================

def analyze_jd_requirements(jd_text, cv_text):

    if not jd_text or not cv_text:
        return []

    # --------------------------------------
    # STEP 1: CLEAN JD TEXT
    # --------------------------------------

    jd_text = jd_text.replace("\r", "\n")

    # --------------------------------------
    # STEP 2: EXTRACT JD REQUIREMENTS
    # --------------------------------------

    requirements = []

    lines = jd_text.splitlines()

    current_section = ""

    for line in lines:

        clean_line = line.strip()

        if not clean_line:
            continue

        # Remove common bullet symbols
        clean_line = clean_line.lstrip(
            "•●▪◦➢➤-✓✔"
        ).strip()

        lower_line = clean_line.lower()

        # ----------------------------------
        # IDENTIFY SECTION HEADINGS
        # ----------------------------------

        if lower_line in [
            "job description",
            "job responsibilities",
            "responsibilities",
            "major responsibilities",
            "duties",
            "major duties",
            "requirements",
            "job requirements",
            "qualification",
            "qualifications"
        ]:

            current_section = lower_line
            continue

        # ----------------------------------
        # REMOVE NUMBERING
        # Example:
        # 1. Coordinate...
        # 2) Handle...
        # 3. Follow up...
        # ----------------------------------

        clean_line = re.sub(
            r"^\s*\d+\s*[\.\)\:\-]\s*",
            "",
            clean_line
        ).strip()

        # Remove simple numbering such as:
        # 1) / 2. / 3-
        clean_line = re.sub(
            r"^\s*\d+\s*[\.\)\-]\s*",
            "",
            clean_line
        ).strip()

        if len(clean_line) < 15:
            continue

        # ----------------------------------
        # ADD MEANINGFUL JD REQUIREMENT
        # ----------------------------------

        requirements.append(clean_line)

    # --------------------------------------
    # STEP 3: REMOVE DUPLICATES
    # --------------------------------------

    unique_requirements = []

    for requirement in requirements:

        if requirement not in unique_requirements:

            unique_requirements.append(requirement)

    requirements = unique_requirements

    # --------------------------------------
    # STEP 4: PREPARE CV LINES
    # --------------------------------------

    cv_lines = []

    for line in cv_text.splitlines():

        clean_line = line.strip()

        if not clean_line:
            continue

        clean_line = clean_line.lstrip(
            "•●▪◦➢➤-✓✔"
        ).strip()

        if len(clean_line) >= 8:

            cv_lines.append(clean_line)

    # --------------------------------------
    # STEP 5: FALLBACK
    # If CV has no line breaks
    # --------------------------------------

    if len(cv_lines) < 3:

        cv_lines = re.split(
            r"[.;]\s+",
            cv_text
        )

        cv_lines = [
            line.strip()
            for line in cv_lines
            if len(line.strip()) >= 8
        ]

    # --------------------------------------
    # STEP 6: MATCH EACH JD REQUIREMENT
    # WITH THE BEST CV EVIDENCE
    # --------------------------------------

    results = []

    for requirement in requirements:

        best_score = 0.0
        best_cv_line = ""

        # ----------------------------------
        # Compare JD requirement against
        # each CV line
        # ----------------------------------

        for cv_line in cv_lines:

            try:

                jd_embedding = model.encode(
                    requirement,
                    convert_to_tensor=True
                )

                cv_embedding = model.encode(
                    cv_line,
                    convert_to_tensor=True
                )

                similarity = util.cos_sim(
                    jd_embedding,
                    cv_embedding
                ).item()

                score = max(
                    0.0,
                    min(
                        100.0,
                        similarity * 100
                    )
                )

                if score > best_score:

                    best_score = score
                    best_cv_line = cv_line

            except Exception:

                continue

        # ----------------------------------
        # MATCH STATUS
        # ----------------------------------

        if best_score >= 65:

            status = "Strong Match"
            icon = "🟢"

        elif best_score >= 40:

            status = "Partial Match"
            icon = "🟡"

        else:

            status = "Weak / Missing"
            icon = "🔴"

        # ----------------------------------
        # SAVE RESULT
        # ----------------------------------

        results.append({

            "requirement": requirement,

            "score": round(
                best_score,
                1
            ),

            "status": status,

            "icon": icon,

            "cv_evidence": best_cv_line

        })

    return results


# ==========================================
# DATABASE CONNECTION
# ==========================================

database_path = r"D:\HR_Recruitment_App\database.db"

conn = sqlite3.connect(database_path)
cursor = conn.cursor()


# ==========================================
# CREATE JOBS TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title TEXT,
    department TEXT,
    location TEXT,
    vacancy INTEGER,
    experience TEXT,
    salary TEXT,
    deadline TEXT
)
""")

conn.commit()


# ==========================================
# ADD STATUS COLUMN IF NOT EXISTS
# ==========================================

cursor.execute("PRAGMA table_info(jobs)")
columns = [column[1] for column in cursor.fetchall()]

if "status" not in columns:

    cursor.execute("""
    ALTER TABLE jobs
    ADD COLUMN status TEXT DEFAULT 'Open'
    """)

    conn.commit()
# ==========================================
# CREATE CANDIDATES TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_name TEXT,
    mobile TEXT,
    email TEXT,
    gender TEXT,
    current_company TEXT,
    current_designation TEXT,
    total_experience TEXT,
    education TEXT,
    location TEXT,
    source TEXT
)
""")

conn.commit()
# ==========================================
# CREATE USERS TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT UNIQUE NOT NULL,

    password TEXT NOT NULL,

    full_name TEXT,

    role TEXT DEFAULT 'HR',

    active INTEGER DEFAULT 1

)
""")

conn.commit()


# ==========================================
# CREATE DEFAULT ADMIN USER
# ==========================================

cursor.execute(
    "SELECT id FROM users WHERE username = ?",
    ("admin",)
)

admin_exists = cursor.fetchone()

if not admin_exists:

    cursor.execute("""
    INSERT INTO users
    (username, password, full_name, role, active)

    VALUES (?, ?, ?, ?, ?)
    """, (
        "admin",
        "Admin@123",
        "System Administrator",
        "Admin",
        1
    ))

    conn.commit()
# ==========================================
# ADD JOB DESCRIPTION COLUMN IF NOT EXISTS
# ==========================================

cursor.execute("PRAGMA table_info(jobs)")
job_columns = [column[1] for column in cursor.fetchall()]

if "job_description" not in job_columns:

    cursor.execute("""
    ALTER TABLE jobs
    ADD COLUMN job_description TEXT
    """)

    conn.commit()
# ==========================================
# CREATE APPLICATIONS TABLE
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER,
    job_id INTEGER,
    application_date TEXT,
    status TEXT DEFAULT 'Applied',
    remarks TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id),
    FOREIGN KEY (job_id) REFERENCES jobs(id)
)
""")

conn.commit()

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HR Recruitment Management System",
    page_icon="👥",
    layout="wide"
)


# ==========================================
# LOGIN SYSTEM
# ==========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "full_name" not in st.session_state:
    st.session_state.full_name = ""

if "user_role" not in st.session_state:
    st.session_state.user_role = ""


# ==========================================
# LOGIN PAGE
# ==========================================

if not st.session_state.logged_in:

    st.markdown(
        """
        <div style="text-align:center; margin-top:60px;">

        <h1>S. F. Denim</h1>

        <h2>HR Recruitment Management System</h2>

        <p style="color:gray;">
        Secure Login
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    login_col1, login_col2, login_col3 = st.columns(
        [1, 2, 1]
    )

    with login_col2:

        st.subheader("🔐 Login")

        login_username = st.text_input(
            "Username",
            key="login_username"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        login_button = st.button(
            "🔐 Login",
            use_container_width=True
        )

        if login_button:

            cursor.execute(
                """
                SELECT
                    username,
                    password,
                    full_name,
                    role
                FROM users
                WHERE username = ?
                AND active = 1
                """,
                (login_username,)
            )

            user = cursor.fetchone()

            if user:

                db_username = user[0]
                db_password = user[1]
                db_full_name = user[2]
                db_role = user[3]

                if login_password == db_password:

                    st.session_state.logged_in = True
                    st.session_state.username = db_username
                    st.session_state.full_name = db_full_name
                    st.session_state.user_role = db_role

                    st.success(
                        "Login successful!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "Incorrect password."
                    )

            else:

                st.error(
                    "Username not found or account is inactive."
                )

    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    # Company Logo
    logo_path = "S. F. Denim Company logo.jpg"

    if os.path.exists(logo_path):

        st.image(
            logo_path,
            width=150
        )

    # Company Name
    st.markdown(
        "<h2 style='text-align:center;'>S. F. Denim</h2>",
        unsafe_allow_html=True
    )

    st.divider()

    # Logged-in User
    st.markdown(
        f"""
        <div style="text-align:center;">

        <small>Logged in as</small><br>

        <strong>{st.session_state.full_name}</strong><br>

        <small>{st.session_state.user_role}</small>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Main Menu
    menu = st.radio(
        "Menu",
        [
            "🏠 Dashboard",
            "💼 Jobs",
            "👤 Candidates",
            "📋 Applications",
            "📄 CV Bank",
            "🎯 Screening",
            "📅 Interviews",
            "📊 Reports"
        ]
    )

    st.divider()

    # Logout
    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.full_name = ""
        st.session_state.user_role = ""

        st.rerun()

    st.divider()

    # Prepared By
    st.markdown(
        """
        <div style="text-align:center;">

        <small>Prepared by:</small><br>

        <strong>Md. Saiful Islam</strong><br>

        <small>Executive - HR</small>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# DASHBOARD
# ==========================================

if menu == "🏠 Dashboard":

    st.title("HR Recruitment Management System")

    st.write("Welcome to your Recruitment Dashboard!")

    cursor.execute("SELECT COUNT(*) FROM jobs")
    total_jobs = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'Open'"
    )
    open_jobs = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'Closed'"
    )
    closed_jobs = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'On Hold'"
    )
    hold_jobs = cursor.fetchone()[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Jobs", total_jobs)

    with col2:
        st.metric("🟢 Open Jobs", open_jobs)

    with col3:
        st.metric("⏸️ On Hold", hold_jobs)

    with col4:
        st.metric("🔴 Closed Jobs", closed_jobs)


# ==========================================
# JOB MANAGEMENT
# ==========================================

elif menu == "💼 Jobs":

    st.title("💼 Job Management")

        # --------------------------------------
    # CREATE NEW JOB
    # --------------------------------------

    st.subheader("➕ Create New Job")

    col1, col2 = st.columns(2)

    with col1:

        job_title = st.text_input(
            "Job Title",
            placeholder="Example: Executive - HR"
        )

        department = st.text_input(
            "Department",
            placeholder="Example: Human Resources"
        )

        location = st.text_input(
            "Location",
            placeholder="Example: Tejgaon, Dhaka"
        )

        vacancy = st.number_input(
            "Number of Vacancies",
            min_value=1,
            step=1
        )

    with col2:

        experience = st.text_input(
            "Experience Required",
            placeholder="Example: 2-4 Years"
        )

        salary = st.text_input(
            "Salary",
            placeholder="Example: Negotiable"
        )

        deadline = st.date_input(
            "Application Deadline"
        )

        status = st.selectbox(
            "Job Status",
            [
                "Open",
                "On Hold",
                "Closed"
            ],
            key="new_job_status"
        )


    # --------------------------------------
    # JOB DESCRIPTION / REQUIREMENTS
    # --------------------------------------

    job_description = st.text_area(
        "Job Description / Requirements",
        placeholder="""Example:

Experience on:
• Talent Acquisition
• Recruitment
• HRIS
• Employee Relations
• Bangladesh Labour Law
• Advanced Excel
• RMG experience preferred

Educational Qualification:
Bachelor/Master in HRM or relevant field.""",
        height=220,
        key="new_job_description"
    )


    # --------------------------------------
    # SAVE JOB
    # --------------------------------------

    if st.button("💾 Save Job"):

        if job_title.strip() == "":
            st.error("Please enter the Job Title.")

        elif department.strip() == "":
            st.error("Please enter the Department.")

        elif location.strip() == "":
            st.error("Please enter the Location.")

        else:

            cursor.execute("""
            INSERT INTO jobs
            (
                job_title,
                department,
                location,
                vacancy,
                experience,
                salary,
                deadline,
                status,
                job_description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_title,
                department,
                location,
                vacancy,
                experience,
                salary,
                str(deadline),
                status,
                job_description
            ))

            conn.commit()

            st.success("✅ Job saved successfully!")

            st.rerun()


    # --------------------------------------
    # SEARCH JOBS
    # --------------------------------------

    st.divider()

    st.subheader("🔍 Search Jobs")

    search_text = st.text_input(
        "Search by Job Title, Department or Location",
        placeholder="Example: HR"
    )


    # --------------------------------------
    # GET JOBS
    # --------------------------------------

    if search_text.strip() == "":

        cursor.execute("""
        SELECT
            id,
            job_title,
            department,
            location,
            vacancy,
            experience,
            salary,
            deadline,
            status,
            job_description
        FROM jobs
        ORDER BY id DESC
        """)

    else:

        search_value = "%" + search_text + "%"

        cursor.execute("""
        SELECT
            id,
            job_title,
            department,
            location,
            vacancy,
            experience,
            salary,
            deadline,
            status,
            job_description
        FROM jobs
        WHERE
            job_title LIKE ?
            OR department LIKE ?
            OR location LIKE ?
        ORDER BY id DESC
        """, (
            search_value,
            search_value,
            search_value
        ))

    jobs = cursor.fetchall()


    # --------------------------------------
    # SAVED JOBS
    # --------------------------------------

    st.subheader("📋 Saved Jobs")

    if jobs:

        for job in jobs:

            job_id = job[0]
            current_status = job[8]
            current_job_description = job[9] or ""

            if current_status == "Open":
                status_icon = "🟢"

            elif current_status == "On Hold":
                status_icon = "⏸️"

            else:
                status_icon = "🔴"


            with st.expander(
                f"{status_icon} {job[1]} | {job[2]} | {job[3]}"
            ):

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write("**Job Title**")
                    st.write(job[1])

                    st.write("**Department**")
                    st.write(job[2])

                    st.write("**Location**")
                    st.write(job[3])

                with col2:

                    st.write("**Vacancy**")
                    st.write(job[4])

                    st.write("**Experience**")
                    st.write(job[5])

                    st.write("**Salary**")
                    st.write(job[6])

                with col3:

                    st.write("**Application Deadline**")
                    st.write(job[7])

                    st.write("**Job ID**")
                    st.write(job_id)

                    st.write("**Current Status**")
                    st.write(
                        f"{status_icon} {current_status}"
                    )


                # ----------------------------------
                # JOB DESCRIPTION
                # ----------------------------------

                st.divider()

                st.write("📄 **Job Description / Requirements**")

                if current_job_description:

                    st.write(current_job_description)

                else:

                    st.info(
                        "No Job Description has been added yet."
                    )


                # ----------------------------------
                # EDIT JOB
                # ----------------------------------

                st.divider()

                st.write("✏️ **Edit Job**")

                edit_col1, edit_col2 = st.columns(2)

                with edit_col1:

                    edit_title = st.text_input(
                        "Job Title",
                        value=job[1],
                        key=f"title_{job_id}"
                    )

                    edit_department = st.text_input(
                        "Department",
                        value=job[2],
                        key=f"department_{job_id}"
                    )

                    edit_location = st.text_input(
                        "Location",
                        value=job[3],
                        key=f"location_{job_id}"
                    )

                    edit_vacancy = st.number_input(
                        "Vacancy",
                        min_value=1,
                        value=job[4],
                        step=1,
                        key=f"vacancy_{job_id}"
                    )

                with edit_col2:

                    edit_experience = st.text_input(
                        "Experience",
                        value=job[5],
                        key=f"experience_{job_id}"
                    )

                    edit_salary = st.text_input(
                        "Salary",
                        value=job[6],
                        key=f"salary_{job_id}"
                    )

                    edit_deadline = st.text_input(
                        "Deadline",
                        value=job[7],
                        key=f"deadline_{job_id}"
                    )

                    edit_status = st.selectbox(
                        "Job Status",
                        [
                            "Open",
                            "On Hold",
                            "Closed"
                        ],
                        index=[
                            "Open",
                            "On Hold",
                            "Closed"
                        ].index(current_status),
                        key=f"status_{job_id}"
                    )


                edit_job_description = st.text_area(
                    "Job Description / Requirements",
                    value=current_job_description,
                    height=220,
                    key=f"job_description_{job_id}"
                )


                # ----------------------------------
                # UPDATE JOB
                # ----------------------------------

                if st.button(
                    "🔄 Update Job",
                    key=f"update_{job_id}"
                ):

                    cursor.execute("""
                    UPDATE jobs
                    SET
                        job_title = ?,
                        department = ?,
                        location = ?,
                        vacancy = ?,
                        experience = ?,
                        salary = ?,
                        deadline = ?,
                        status = ?,
                        job_description = ?
                    WHERE id = ?
                    """, (
                        edit_title,
                        edit_department,
                        edit_location,
                        edit_vacancy,
                        edit_experience,
                        edit_salary,
                        edit_deadline,
                        edit_status,
                        edit_job_description,
                        job_id
                    ))

                    conn.commit()

                    st.success(
                        f"✅ Job ID {job_id} updated successfully!"
                    )

                    st.rerun()

    else:

        st.info("No jobs found.")


# ==========================================
# CANDIDATES
# ==========================================

elif menu == "👤 Candidates":

    st.title("👤 Candidate Management")

    # --------------------------------------
    # ADD NEW CANDIDATE
    # --------------------------------------

    st.subheader("➕ Add New Candidate")

    col1, col2 = st.columns(2)

    with col1:

        candidate_name = st.text_input(
            "Candidate Name",
            placeholder="Example: Md. Rahim Uddin"
        )

        mobile = st.text_input(
            "Mobile Number",
            placeholder="Example: 017XXXXXXXX"
        )

        email = st.text_input(
            "Email",
            placeholder="Example: candidate@email.com"
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

        location = st.text_input(
            "Current Location",
            placeholder="Example: Dhaka"
        )

    with col2:

        current_company = st.text_input(
            "Current / Previous Company",
            placeholder="Example: ABC Garments Ltd."
        )

        current_designation = st.text_input(
            "Current Designation",
            placeholder="Example: HR Executive"
        )

        total_experience = st.text_input(
            "Total Experience",
            placeholder="Example: 4 Years"
        )

        education = st.text_input(
            "Highest Education",
            placeholder="Example: MBA in HRM"
        )

        source = st.selectbox(
            "Candidate Source",
            [
                "LinkedIn",
                "Facebook",
                "Bdjobs",
                "Company Website",
                "Employee Referral",
                "Email",
                "Other"
            ]
        )


    # --------------------------------------
    # SAVE CANDIDATE
    # --------------------------------------

    if st.button("💾 Save Candidate"):

        if candidate_name.strip() == "":
            st.error("Please enter Candidate Name.")

        elif mobile.strip() == "":
            st.error("Please enter Mobile Number.")

        else:

            cursor.execute("""
            INSERT INTO candidates
            (
                candidate_name,
                mobile,
                email,
                gender,
                current_company,
                current_designation,
                total_experience,
                education,
                location,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candidate_name,
                mobile,
                email,
                gender,
                current_company,
                current_designation,
                total_experience,
                education,
                location,
                source
            ))

            conn.commit()

            st.success(
                "✅ Candidate saved successfully!"
            )

            st.rerun()


    # --------------------------------------
    # SEARCH CANDIDATES
    # --------------------------------------

    st.divider()

    st.subheader("🔍 Search Candidates")

    search_candidate = st.text_input(
        "Search by Name, Mobile, Company or Designation",
        placeholder="Example: Rahim"
    )


    # --------------------------------------
    # GET CANDIDATES
    # --------------------------------------

    if search_candidate.strip() == "":

        cursor.execute("""
        SELECT
            id,
            candidate_name,
            mobile,
            email,
            gender,
            current_company,
            current_designation,
            total_experience,
            education,
            location,
            source
        FROM candidates
        ORDER BY id DESC
        """)

    else:

        search_value = "%" + search_candidate + "%"

        cursor.execute("""
        SELECT
            id,
            candidate_name,
            mobile,
            email,
            gender,
            current_company,
            current_designation,
            total_experience,
            education,
            location,
            source
        FROM candidates
        WHERE
            candidate_name LIKE ?
            OR mobile LIKE ?
            OR current_company LIKE ?
            OR current_designation LIKE ?
        ORDER BY id DESC
        """, (
            search_value,
            search_value,
            search_value,
            search_value
        ))

    candidates = cursor.fetchall()


    # --------------------------------------
    # CANDIDATE DATABASE
    # --------------------------------------

    st.subheader("📋 Candidate Database")

    if candidates:

        for candidate in candidates:

            candidate_id = candidate[0]

            with st.expander(
                f"👤 {candidate[1]} | {candidate[6]} | {candidate[5]}"
            ):

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write("**Candidate Name**")
                    st.write(candidate[1])

                    st.write("**Mobile**")
                    st.write(candidate[2])

                    st.write("**Email**")
                    st.write(candidate[3])

                    st.write("**Gender**")
                    st.write(candidate[4])

                with col2:

                    st.write("**Current / Previous Company**")
                    st.write(candidate[5])

                    st.write("**Designation**")
                    st.write(candidate[6])

                    st.write("**Total Experience**")
                    st.write(candidate[7])

                    st.write("**Education**")
                    st.write(candidate[8])

                with col3:

                    st.write("**Location**")
                    st.write(candidate[9])

                    st.write("**Source**")
                    st.write(candidate[10])

                    st.write("**Candidate ID**")
                    st.write(candidate_id)

    else:

        st.info(
            "No candidates have been added yet."
        )
            # ======================================
    # EDIT / UPDATE CANDIDATE
    # ======================================

    st.divider()

    st.subheader("✏️ Edit / Update Candidate")

    candidate_ids = [candidate[0] for candidate in candidates]

    if candidate_ids:

        selected_candidate_id = st.selectbox(
            "Select Candidate ID",
            candidate_ids
        )

        cursor.execute("""
        SELECT
            candidate_name,
            mobile,
            email,
            gender,
            current_company,
            current_designation,
            total_experience,
            education,
            location,
            source
        FROM candidates
        WHERE id = ?
        """, (selected_candidate_id,))

        selected_candidate = cursor.fetchone()

        if selected_candidate:

            col1, col2 = st.columns(2)

            with col1:

                edit_name = st.text_input(
                    "Candidate Name",
                    value=selected_candidate[0]
                )

                edit_mobile = st.text_input(
                    "Mobile Number",
                    value=selected_candidate[1]
                )

                edit_email = st.text_input(
                    "Email",
                    value=selected_candidate[2]
                )

                edit_gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"],
    index=["Male", "Female", "Other"].index(
        selected_candidate[3]
    ),
    key="edit_gender"
)

                edit_location = st.text_input(
                    "Current Location",
                    value=selected_candidate[8]
                )

            with col2:

                edit_company = st.text_input(
                    "Current / Previous Company",
                    value=selected_candidate[4]
                )

                edit_designation = st.text_input(
                    "Current Designation",
                    value=selected_candidate[5]
                )

                edit_experience = st.text_input(
                    "Total Experience",
                    value=selected_candidate[6]
                )

                edit_education = st.text_input(
                    "Highest Education",
                    value=selected_candidate[7]
                )

                source_options = [
                    "LinkedIn",
                    "Facebook",
                    "Bdjobs",
                    "Company Website",
                    "Employee Referral",
                    "Email",
                    "Other"
                ]

                current_source = selected_candidate[9]

                if current_source not in source_options:
                    current_source = "Other"

                edit_source = st.selectbox(
    "Candidate Source",
    source_options,
    index=source_options.index(current_source),
    key="edit_candidate_source"
)

            if st.button("🔄 Update Candidate"):

                if edit_name.strip() == "":
                    st.error("Candidate Name cannot be empty.")

                elif edit_mobile.strip() == "":
                    st.error("Mobile Number cannot be empty.")

                else:

                    cursor.execute("""
                    UPDATE candidates
                    SET
                        candidate_name = ?,
                        mobile = ?,
                        email = ?,
                        gender = ?,
                        current_company = ?,
                        current_designation = ?,
                        total_experience = ?,
                        education = ?,
                        location = ?,
                        source = ?
                    WHERE id = ?
                    """, (
                        edit_name,
                        edit_mobile,
                        edit_email,
                        edit_gender,
                        edit_company,
                        edit_designation,
                        edit_experience,
                        edit_education,
                        edit_location,
                        edit_source,
                        selected_candidate_id
                    ))

                    conn.commit()

                    st.success(
                        "✅ Candidate information updated successfully!"
                    )

                    st.rerun()

    else:

        st.info(
            "No candidate is available for editing."
        )
        # ==========================================
# APPLICATIONS
# ==========================================

elif menu == "📋 Applications":

    st.title("📋 Application Management")

    st.subheader("➕ Apply Candidate to Job")

    # --------------------------------------
    # GET CANDIDATES
    # --------------------------------------

    cursor.execute("""
    SELECT
        id,
        candidate_name,
        current_designation
    FROM candidates
    ORDER BY candidate_name
    """)

    candidate_list = cursor.fetchall()


    # --------------------------------------
    # GET JOBS
    # --------------------------------------

    cursor.execute("""
    SELECT
        id,
        job_title,
        department
    FROM jobs
    WHERE status = 'Open'
    ORDER BY job_title
    """)

    job_list = cursor.fetchall()


    if not candidate_list:

        st.warning(
            "No candidates are available. Please add a candidate first."
        )

    elif not job_list:

        st.warning(
            "No open jobs are available. Please create an open job first."
        )

    else:

        # ----------------------------------
        # CANDIDATE SELECTION
        # ----------------------------------

        candidate_options = {
            f"{candidate[1]} | {candidate[2]} | ID: {candidate[0]}":
            candidate[0]
            for candidate in candidate_list
        }

        selected_candidate = st.selectbox(
            "Select Candidate",
            list(candidate_options.keys()),
            key="application_candidate"
        )

        candidate_id = candidate_options[selected_candidate]


        # ----------------------------------
        # JOB SELECTION
        # ----------------------------------

        job_options = {
            f"{job[1]} | {job[2]} | ID: {job[0]}":
            job[0]
            for job in job_list
        }

        selected_job = st.selectbox(
            "Select Job",
            list(job_options.keys()),
            key="application_job"
        )

        job_id = job_options[selected_job]


        # ----------------------------------
        # APPLICATION DATE
        # ----------------------------------

        application_date = st.date_input(
            "Application Date",
            key="application_date"
        )


        # ----------------------------------
        # STATUS
        # ----------------------------------

        status = st.selectbox(
            "Application Status",
            [
                "Applied",
                "Screening",
                "Shortlisted",
                "Hold",
                "Rejected",
                "Interview",
                "Selected",
                "Offered",
                "Joined",
                "Withdrawn"
            ],
            key="application_status"
        )


        # ----------------------------------
        # REMARKS
        # ----------------------------------

        remarks = st.text_area(
            "Remarks",
            placeholder="Example: CV received through LinkedIn.",
            key="application_remarks"
        )


        # ----------------------------------
        # APPLY
        # ----------------------------------

        if st.button(
            "📌 Create Application",
            key="create_application"
        ):

            cursor.execute("""
            SELECT id
            FROM applications
            WHERE candidate_id = ?
            AND job_id = ?
            """, (
                candidate_id,
                job_id
            ))

            existing_application = cursor.fetchone()


            if existing_application:

                st.warning(
                    "This candidate has already been applied to this job."
                )

            else:

                cursor.execute("""
                INSERT INTO applications
                (
                    candidate_id,
                    job_id,
                    application_date,
                    status,
                    remarks
                )
                VALUES (?, ?, ?, ?, ?)
                """, (
                    candidate_id,
                    job_id,
                    str(application_date),
                    status,
                    remarks
                ))

                conn.commit()

                st.success(
                    "✅ Application created successfully!"
                )

                st.rerun()
                    # ======================================
    # APPLICATION DATABASE
    # ======================================

    st.divider()

    st.subheader("📊 Application Database")

    cursor.execute("""
    SELECT
        applications.id,
        candidates.candidate_name,
        jobs.job_title,
        jobs.department,
        applications.application_date,
        applications.status,
        applications.remarks
    FROM applications
    LEFT JOIN candidates
        ON applications.candidate_id = candidates.id
    LEFT JOIN jobs
        ON applications.job_id = jobs.id
    ORDER BY applications.id DESC
    """)

    applications = cursor.fetchall()


    if applications:

        for application in applications:

            application_id = application[0]
            candidate_name = application[1]
            job_title = application[2]
            department = application[3]
            application_date = application[4]
            application_status = application[5]
            remarks = application[6]

            # Status icon

            if application_status == "Applied":
                status_icon = "🔵"

            elif application_status == "Screening":
                status_icon = "🟡"

            elif application_status == "Shortlisted":
                status_icon = "🟢"

            elif application_status == "Interview":
                status_icon = "🟣"

            elif application_status == "Selected":
                status_icon = "⭐"

            elif application_status == "Offered":
                status_icon = "📨"

            elif application_status == "Joined":
                status_icon = "✅"

            elif application_status == "Rejected":
                status_icon = "🔴"

            elif application_status == "Hold":
                status_icon = "⏸️"

            elif application_status == "Withdrawn":
                status_icon = "⚪"

            else:
                status_icon = "🔵"


            with st.expander(
                f"{status_icon} {candidate_name} → {job_title}"
            ):

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write("**Candidate**")
                    st.write(candidate_name)

                    st.write("**Job**")
                    st.write(job_title)

                with col2:

                    st.write("**Department**")
                    st.write(department)

                    st.write("**Application Date**")
                    st.write(application_date)

                with col3:

                    st.write("**Status**")
                    st.write(
                        f"{status_icon} {application_status}"
                    )

                    st.write("**Application ID**")
                    st.write(application_id)


                if remarks:

                    st.write("**Remarks**")
                    st.write(remarks)

    else:

        st.info(
            "No applications have been created yet."
        )
# ==========================================
# CV BANK
# ==========================================

elif menu == "📄 CV Bank":

    st.title("📄 CV Bank")

    st.write("CV Storage and Management System")

    st.info(
        "CV upload and storage module will be developed next."
    )


# ==========================================
# SCREENING
# ==========================================

elif menu == "🎯 Screening":

    st.title("🎯 AI CV Screening")

    st.write(
        "Upload multiple CVs and screen candidates "
        "against the selected Job Description."
    )

    st.divider()

    # --------------------------------------
    # STEP 1: SELECT JOB
    # --------------------------------------

    st.subheader("1️⃣ Select Position")

    cursor.execute("""
    SELECT id, job_title, department
    FROM jobs
    WHERE status = 'Open'
    ORDER BY id DESC
    """)

    screening_jobs = cursor.fetchall()

    if not screening_jobs:

        st.warning(
            "No Open Job is available. "
            "Please create an Open Job first."
        )

    else:

        job_options = {}

        for job in screening_jobs:

            job_id = job[0]

            job_label = (
                f"{job[1]} | {job[2]} | Job ID: {job_id}"
            )

            job_options[job_label] = job_id

        selected_job_label = st.selectbox(
            "Select Position",
            list(job_options.keys()),
            key="screening_job_select"
        )

        selected_job_id = job_options[
            selected_job_label
        ]

        # --------------------------------------
        # STEP 2: LOAD JOB REQUIREMENTS
        # --------------------------------------

        st.subheader("2️⃣ Job Requirements")

        cursor.execute("""
        SELECT
            job_title,
            department,
            job_description
        FROM jobs
        WHERE id = ?
        """, (selected_job_id,))

        selected_job = cursor.fetchone()

        if selected_job:

            job_title = selected_job[0]
            department = selected_job[1]
            job_description = selected_job[2] or ""

            st.write(
                f"**Position:** {job_title}"
            )

            st.write(
                f"**Department:** {department}"
            )

            if job_description.strip():

                st.text_area(
                    "Job Description / Requirements",
                    value=job_description,
                    height=250,
                    disabled=True,
                    key="screening_job_description"
                )

            else:

                st.warning(
                    "No Job Description / Requirements "
                    "has been added for this job."
                )

        st.divider()

        # --------------------------------------
        # STEP 3: UPLOAD CVs
        # --------------------------------------

        st.subheader("3️⃣ Upload CVs")

        uploaded_cvs = st.file_uploader(
            "Upload multiple CVs",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="screening_cv_upload"
        )

        if uploaded_cvs:

            st.success(
                f"✅ {len(uploaded_cvs)} CV(s) uploaded successfully."
            )

            st.write("### Uploaded CVs")

            for cv in uploaded_cvs:

                st.write(
                    f"📄 {cv.name}"
                )

        else:

            st.info(
                "Please upload one or more PDF/DOCX CVs."
            )

        # --------------------------------------
        # STEP 4: EXTRACT CV TEXT
        # --------------------------------------

        if uploaded_cvs:

            st.divider()

            st.subheader("4️⃣ Extract CV Text")

            if st.button(
                "📄 Extract CV Text",
                type="primary",
                key="extract_cv_text_button"
            ):

                st.subheader("📋 Extracted CV Text")

                for cv in uploaded_cvs:

                    try:

                        cv_text = extract_cv_text(cv)

                        st.session_state[
                            f"cv_text_{cv.name}"
                        ] = cv_text

                        with st.expander(
                            f"📄 {cv.name}",
                            expanded=False
                        ):

                            if cv_text.strip():

                                st.text_area(
                                    "Extracted Text",
                                    value=cv_text,
                                    height=350,
                                    key=f"extracted_{cv.name}"
                                )

                            else:

                                st.warning(
                                    "⚠️ No readable text was found in this CV."
                                )

                    except Exception as e:

                        with st.expander(
                            f"📄 {cv.name}"
                        ):

                            st.error(
                                f"❌ Error reading CV: {str(e)}"
                            )

        # --------------------------------------
        # STEP 5: AI CV SCREENING
        # --------------------------------------

        if uploaded_cvs:

            st.divider()

            st.subheader("5️⃣ AI CV Screening")

            if st.button(
                "🤖 Screen CVs",
                type="primary",
                key="check_cv_match_button"
            ):

                # --------------------------------------
                # CHECK JOB DESCRIPTION
                # --------------------------------------

                if not job_description.strip():

                    st.warning(
                        "⚠️ এই Job-এর কোনো Job Description / Requirements পাওয়া যায়নি।"
                    )

                else:

                    st.write(
                        "### 📊 AI Screening Results"
                    )

                    # --------------------------------------
                    # SCREEN EVERY UPLOADED CV
                    # --------------------------------------

                    for cv in uploaded_cvs:

                        cv_text = st.session_state.get(
                            f"cv_text_{cv.name}",
                            ""
                        )

                        # --------------------------------------
                        # CHECK CV TEXT
                        # --------------------------------------

                        if not cv_text.strip():

                            st.warning(
                                f"⚠️ {cv.name} এর CV text পাওয়া যায়নি। "
                                "প্রথমে 'Extract CV Text' button চাপুন।"
                            )

                            continue

                        # --------------------------------------
                        # CANDIDATE NAME
                        # --------------------------------------

                        candidate_name = extract_candidate_name(
                            cv_text,
                            cv.name
                        )

                        # --------------------------------------
                        # CATEGORY-WISE AI MATCHING
                        # --------------------------------------

                        category_scores = calculate_category_scores(
                            job_description,
                            cv_text
                        )

                        # --------------------------------------
                        # WEIGHTED AI SCORE
                        # --------------------------------------

                        final_score = calculate_weighted_score(
                            category_scores
                        )

                        # --------------------------------------
                        # EDUCATION MATCH
                        # --------------------------------------

                        education_match = calculate_education_match(
                            job_description,
                            cv_text
                        )

                        # --------------------------------------
                        # RECOMMENDATION
                        # --------------------------------------

                        recommendation = get_screening_recommendation(
                            final_score
                        )

                        # --------------------------------------
                        # REQUIREMENT-WISE AI ANALYSIS
                        # --------------------------------------

                        requirement_analysis = analyze_jd_requirements(
                            job_description,
                            cv_text
                        )

                        # --------------------------------------
                        # AI SCREENING SUMMARY
                        # --------------------------------------

                        screening_summary = generate_screening_summary(
                            candidate_name,
                            final_score,
                            recommendation,
                            category_scores,
                            requirement_analysis
                        )

                        # --------------------------------------
                        # CANDIDATE NAME
                        # --------------------------------------

                        st.write(
                            f"### 👤 {candidate_name}"
                        )

                        st.caption(
                            f"CV File: {cv.name}"
                        )

                        # --------------------------------------
                        # SCORE + DECISION
                        # --------------------------------------

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "AI Match Score",
                                f"{final_score}/100"
                            )

                        with col2:

                            st.metric(
                                "Education Match",
                                f"{education_match:.1f}%"
                            )

                        with col3:

                            st.metric(
                                "Recommendation",
                                recommendation
                            )

                        # --------------------------------------
                        # SCORE PROGRESS BAR
                        # --------------------------------------

                        st.progress(
                            min(max(final_score / 100, 0), 1)
                        )

                        # --------------------------------------
                        # CATEGORY BREAKDOWN
                        # --------------------------------------

                        st.write(
                            "#### 📊 Category-wise AI Match"
                        )

                        education_score = round(
                            category_scores.get("education", 0) * 15 / 100,
                            2
                        )

                        experience_score = round(
                            category_scores.get("experience", 0) * 25 / 100,
                            2
                        )

                        skills_score = round(
                            category_scores.get("skills", 0) * 25 / 100,
                            2
                        )

                        responsibilities_score = round(
                            category_scores.get("responsibilities", 0) * 20 / 100,
                            2
                        )

                        rmg_denim_score = round(
                            category_scores.get("rmg_denim", 0) * 15 / 100,
                            2
                        )

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write(
                                "🎓 **Education**"
                            )

                            st.write(
                                f"Match: {category_scores.get('education', 0):.1f}% "
                                f"| Score: {education_score}/15"
                            )

                            st.write(
                                "💼 **Experience**"
                            )

                            st.write(
                                f"Match: {category_scores.get('experience', 0):.1f}% "
                                f"| Score: {experience_score}/25"
                            )

                            st.write(
                                "🛠 **Skills**"
                            )

                            st.write(
                                f"Match: {category_scores.get('skills', 0):.1f}% "
                                f"| Score: {skills_score}/25"
                            )

                        with col2:

                            st.write(
                                "📋 **Responsibilities**"
                            )

                            st.write(
                                f"Match: {category_scores.get('responsibilities', 0):.1f}% "
                                f"| Score: {responsibilities_score}/20"
                            )

                            st.write(
                                "👕 **RMG / Denim**"
                            )

                            st.write(
                                f"Match: {category_scores.get('rmg_denim', 0):.1f}% "
                                f"| Score: {rmg_denim_score}/15"
                            )

                        st.divider()

                        # --------------------------------------
                        # FINAL SCORE SUMMARY
                        # --------------------------------------

                        st.write(
                            "#### 🎯 Final AI Score"
                        )

                        st.success(
                            f"**{final_score}/100**"
                        )

                        # --------------------------------------
                        # AI SCREENING SUMMARY
                        # --------------------------------------

                        st.write(
                            "#### 📝 AI Screening Summary"
                        )

                        st.info(
                            screening_summary
                        )

                        # --------------------------------------
                        # REQUIREMENT-WISE MATCH ANALYSIS
                        # --------------------------------------

                        st.write(
                            "#### 🔍 Requirement-wise AI Analysis"
                        )

                        strong_matches = []
                        partial_matches = []
                        weak_matches = []

                        for item in requirement_analysis:

                            if item["status"] == "Strong Match":

                                strong_matches.append(item)

                            elif item["status"] == "Partial Match":

                                partial_matches.append(item)

                            else:

                                weak_matches.append(item)

                        # --------------------------------------
                        # STRONG MATCH
                        # --------------------------------------

                        if strong_matches:

                            st.write(
                                "### 🟢 Strong Match"
                            )

                            for item in strong_matches:

                                st.write(
                                    f"✓ {item['requirement']} "
                                    f"({item['score']:.1f}%)"
                                )

                        # --------------------------------------
                        # PARTIAL MATCH
                        # --------------------------------------

                        if partial_matches:

                            st.write(
                                "### 🟡 Partial Match"
                            )

                            for item in partial_matches:

                                st.write(
                                    f"△ {item['requirement']} "
                                    f"({item['score']:.1f}%)"
                                )

                        # --------------------------------------
                        # MISSING / WEAK
                        # --------------------------------------

                        if weak_matches:

                            st.write(
                                "### 🔴 Missing / Weak"
                            )

                            for item in weak_matches:

                                st.write(
                                    f"✗ {item['requirement']} "
                                    f"({item['score']:.1f}%)"
                                )

                        st.divider()


# ==========================================
# INTERVIEWS
# ==========================================

elif menu == "📅 Interviews":

    st.title("📅 Interview Management")

    st.write(
        "Interview Scheduling and Evaluation"
    )

    st.info(
        "Interview management module will be developed next."
    )


# ==========================================
# REPORTS
# ==========================================

elif menu == "📊 Reports":

    st.title("📊 Recruitment Reports")

    st.write(
        "Recruitment Analytics and Reports"
    )

    st.info(
        "Recruitment reports will be developed later."
    )


# ==========================================
# CLOSE DATABASE
# ==========================================

conn.close()
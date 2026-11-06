import streamlit as st
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types
import json
import os
import pandas as pd

st.set_page_config(page_title="People System - ATS Screener", layout="wide")
st.title("👥 People System — Enterprise Batch CV Screener")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input("Enter Gemini API Key:", type="password")

# 1. Target Position & Strict Requirements
st.subheader("১. টার্গেট পদের তথ্য ও ক্রাইটেরিয়া (Job Requirements & KRA/KSA)")
col1, col2 = st.columns(2)
with col1:
    target_position = st.text_input("Position Title:", "Senior Executive - Merchandising / HR / IE")
    department = st.text_input("Department:", "Merchandising / HR / Production")
with col2:
    req_exp = st.text_input("Required Experience / Grade:", "ME-10 (4.5 Years Experience)")
    academic_buyer = st.text_input("Target Degree & Buyers/Tools:", "B.Sc. in Textile / BBA | H&M, Zara, Walmart | FastReact, ERP")

jd_criteria = st.text_area(
    "Job Responsibilities (KRA) & Competencies (KSA) - Bullet Points:",
    height=160,
    value="1. End-to-end follow up from sample development to shipment approval.\n2. Buyer correspondence, price quotation, consumption and costing calculation.\n3. Daily coordination with production floor, wash plant, and quality team.\n4. Timely submission of production status reports and compliance adherence."
)

# 2. Multi-File Batch Upload (10-20 CVs simultaneously)
st.subheader("২. একসাথে ১০-২০টি সিভি আপলোড করুন (PDF বা DOCX)")
uploaded_files = st.file_uploader(
    "সিভি ড্র্যাগ এবং ড্রপ করুন:", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)

def extract_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file.name.endswith(".docx"):
        doc = Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    return text.strip()

# 3. Screening & Ranking Logic
if st.button("🚀 Start Screening & Generate Merit Shortlist", type="primary"):
    if not api_key:
        st.error("দয়া করে সাইডবারে আপনার Gemini API Key প্রদান করুন।")
    elif not uploaded_files:
        st.warning("স্ক্রিন করার জন্য কমপক্ষে একটি বা একাধিক সিভি আপলোড করুন।")
    else:
        client = genai.Client(api_key=api_key)
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {file.name}...")
            
            # Robust Text Extraction
            cv_text = extract_text_from_file(file)
            
            if len(cv_text) < 100:
                st.warning(f"⚠️ {file.name}: টেক্সট পড়া যায়নি (স্ক্যান করা ইমেজ ফাইল হতে পারে)।")
                continue

            eval_prompt = f"""
            You are a Chief Talent Acquisition Officer. Evaluate the candidate's CV against the Target Job Requisition.
            
            TARGET JOB DETAILS:
            - Position: {target_position}
            - Department: {department}
            - Grade & Experience: {req_exp}
            - Required Degree/Buyers/Tools: {academic_buyer}
            - Core Responsibilities (KRA/KSA):
            {jd_criteria}
            
            CANDIDATE CV CONTENT:
            \"\"\"{cv_text}\"\"\"
            
            STRICT RULES:
            1. Semantic Meaning: Evaluate the functional purpose of their duties. If past work fulfills the JD's purpose, award positive match.
            2. If tasks are omitted from the CV but designation, department, and company track match the target role, evaluate credentials and award an initial 70-80% score (Do NOT reject as irrelevant).
            3. Disqualify (Score < 40%, 'Irrelevant') ONLY if the candidate's career vertical has zero relation to the position.
            
            Output strictly in JSON format:
            {{
              "candidate_name": "Extracted Full Name",
              "last_designation": "Recent Title",
              "total_exp": "Total Years of Exp",
              "last_company": "Recent Organization",
              "kra_score": 85,
              "specs_score": 80,
              "final_score": 82,
              "verdict": "Perfect Matched / Better / Good / Irrelevant",
              "hr_verdict": "1-sentence decisive advice for HR.",
              "matched_strengths": ["Matched duty 1", "Matched duty 2"],
              "critical_gaps": ["Gap 1 if any"]
            }}
            """
            
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=eval_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                data = json.loads(response.text)
                data["filename"] = file.name
                results.append(data)
            except Exception as e:
                st.error(f"{file.name} প্রসেস করতে এরর হয়েছে: {str(e)}")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.success("✅ সকল সিভির স্ক্রিনিং সম্পন্ন হয়েছে!")

        # মেধা তালিকা সর্টিং (Rank #1, #2...)
        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        st.write("---")
        st.subheader("📋 শর্টলিস্টেড মেধা তালিকা (Merit-Ranked Shortlist)")

        for rank, res in enumerate(results, start=1):
            badge = "🟢" if res.get("final_score", 0) >= 80 else ("🟡" if res.get("final_score", 0) >= 60 else "🔴")
            
            with st.expander(f"Rank #{rank} | {badge} {res.get('candidate_name', 'Unknown')} — Score: {res.get('final_score', 0)}% [{res.get('verdict')}]"):
                st.markdown(f"**পদবী ও ক্যারিয়ার:** {res.get('last_designation')} | {res.get('last_company')} | অভিজ্ঞতা: {res.get('total_exp')}")
                st.markdown(f"**টাস্ক স্কোর (KRA):** {res.get('kra_score')}% | **স্পেসিফিকেশন স্কোর:** {res.get('specs_score')}%")
                st.info(f"💡 **এইচআর ডিসিশন:** {res.get('hr_verdict')}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**মিলে যাওয়া দায়িত্ব ও দক্ষতা:**")
                    for s in res.get("matched_strengths", []):
                        st.write(f"- {s}")
                with col_b:
                    st.write("**ঘাটতি বা মিসিং পয়েন্ট:**")
                    for g in res.get("critical_gaps", []):
                        st.write(f"- {g}")
                st.caption(f"আসল ফাইল: {res.get('filename')}")

        if results:
            export_df = pd.DataFrame([{
                "Rank": idx + 1,
                "Candidate Name": r.get("candidate_name"),
                "Designation": r.get("last_designation"),
                "Company": r.get("last_company"),
                "Experience": r.get("total_exp"),
                "Final Score": r.get("final_score"),
                "Verdict": r.get("verdict"),
                "HR Decision": r.get("hr_verdict"),
                "File": r.get("filename")
            } for idx, r in enumerate(results)])
            
            st.download_button(
                "📥 শর্টলিস্ট এক্সেল ফাইল ডাউনলোড করুন",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f"{target_position}_Shortlist.csv",
                mime="text/csv"
            )

# -----------------------------------
# Imports & Setup
# -----------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from re import sub
import time
from typing import List
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import seaborn as sns
import snowflake.connector
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from PIL import Image
from requests.exceptions import ConnectionError, InvalidURL, RequestException, Timeout
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.graph_objects as go
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from openai import OpenAI
import io

# -----------------------------------
# Styling, Auth, Banner
# -----------------------------------
def add_banner(username=None) -> None:
    with stylable_container(
        key="fixed-logo",
        css_styles="""
            div[data-testid="stImage"] {
                position: fixed;
                top: 2.5px;
                left: 2.5px;
                width: 100px;
                z-index: 2000;
                border-radius: 0px;
                background-color: white;
                padding: 4px;
            }
        """
    ):
        st.image("assets/jadeglobal.png", width=72)

    welcome_html = f"""
        <div style="position: fixed; top: 0px; left: 0px; width: 100px; background-color: white; height: 40px; z-index: 1"></div>        
        <div style="position: fixed; top: 0px; left: 0px; width: 100%; background-color: #175388; height: 40px; z-index: 1; display: flex; align-items: center; justify-content: flex-end; padding-right: 20px; color: white; font-weight: bold;">
            {"Welcome, " + username if username else ""}
        </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)

def hide_basic_streamlit_elements() -> None:
    st.markdown("""
        <style>
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"],
            div[data-testid="stStatusWidget"],
            #MainMenu, header {
                visibility: hidden;
                height: 0%;
            }
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
            }

            /* Sidebar starts below the fixed logo */
            section[data-testid="stSidebar"] {
                margin-top: 80px;
                padding-top: 20px;
                z-index: 999;
                background-color: #ffffff;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)

def add_general_style() -> None:
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                font-family: 'Segoe UI', sans-serif;
                background-color: #f8f9fa;
                color: #212529;
            }

            .title {
                font-size: 2.3rem;
                font-weight: 700;
                color: #175388;
                margin-bottom: 1rem;
            }

            .stBlock {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 2rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                margin-bottom: 2rem;
            }

            .section-card {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 0 8px rgba(0,0,0,0.05);
                margin-bottom: 1.5rem;
            }

            .section-title {
                font-size: 1.2rem;
                font-weight: 600;
                color: #175388;
                margin-bottom: 1rem;
            }

            button[kind="primary"], button[kind="secondary"] {
                background-color: #175388 !important;
                color: white !important;
                border-radius: 5px !important;
                padding: 0.5rem 1.5rem !important;
                font-weight: 600 !important;
                transition: all 0.2s ease-in-out;
            }
            button[kind="primary"]:hover, button[kind="secondary"]:hover {
                background-color: #0f3a60 !important;
                transform: scale(1.02);
            }

            input, textarea, .stTextInput>div>div>input {
                border-radius: 6px !important;
                border: 1px solid #ced4da !important;
                padding: 0.4rem 0.75rem !important;
            }

            .stDownloadButton button {
                background-color: #28a745 !important;
                color: white !important;
                border-radius: 5px !important;
                font-weight: bold;
            }
            .stDownloadButton button:hover {
                background-color: #218838 !important;
            }

            .stSpinner {
                font-size: 1.1rem;
                font-weight: 500;
                color: #175388;
            }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------------
# Authentication
# -----------------------------------
st_UserName = st.secrets["streamlit_username"]
st_Password = st.secrets["streamlit_password"]

def creds_entered():
    if len(st.session_state["streamlit_username"]) > 0 and len(st.session_state["streamlit_password"]) > 0:
        if st.session_state["streamlit_username"].strip() == st_UserName \
                and st.session_state["streamlit_password"].strip() == st_Password:
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
            st.error("Invalid Username/Password")

def authenticate_user():
    if "authenticated" not in st.session_state:
        st.markdown("")
        st.markdown("")
        buff, col, buff2 = st.columns([1, 1, 1])
        col.text_input("Username", key="streamlit_username", on_change=creds_entered)
        col.text_input("Password", type="password", key="streamlit_password", on_change=creds_entered)
        return False
    elif st.session_state["authenticated"]:
        return True
    else:
        buff, col, buff2 = st.columns([1, 1, 1])
        col.text_input("Username", key="streamlit_username", on_change=creds_entered)
        col.text_input("Password", type="password", key="streamlit_password", on_change=creds_entered)
        return False

# -----------------------------------
# App Entry Point
# -----------------------------------
if __name__ == "__main__":
    jadeimage = Image.open("assets/jadeglobal.png")
    st.set_page_config(page_title="Jade Global", page_icon=jadeimage, layout="wide")

    hide_basic_streamlit_elements()
    add_general_style()
    add_banner(st_UserName if "authenticated" in st.session_state else "")

    if authenticate_user():
        #st.markdown('<h1 class="title">🧠 JSkillify : GPT-Based Skill Mapping</h1>', unsafe_allow_html=True)
        st.markdown('<h1 class="title">🧠 JSkillify : Transforming Talent Intelligence</h1>', unsafe_allow_html=True)
        st.write("""
**JSkillify** is an AI-driven platform that precisely maps job descriptions to skill taxonomies at scale. Built for modern HR teams, it automates job requisition analysis and aligns them with your skill frameworks. With advanced language models, JSkillify boosts recruitment, workforce planning, and talent development — matching the right people to the right roles, faster.

**Smarter hiring starts with JSkillify.**
""")

        required_columns_file1 = {"Job Requisition", "Job Description", "Cost Center"}
        required_columns_file2 = {"Bucket", "Skill Category", "Level 1", "Level 2", "Level 3"}

        

        # Function to validate required columns
        def validate_columns(file, required_columns, file_label):
            try:
                df = pd.read_excel(file)
                missing_columns = required_columns - set(df.columns)
                if missing_columns:
                    st.error(f"❌ {file_label} is missing columns: {', '.join(missing_columns)}")
                    return None
                st.success(f"✅ {file_label} uploaded successfully.")
                return df
            except Exception as e:
                st.error(f"⚠️ Failed to read {file_label}: {e}")
                return None
            
        
        # Sidebar for upload section only
        with st.sidebar:
            #st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.write('<div class="section-title">📁 Upload Section</div>', unsafe_allow_html=True)
            uploaded_file1 = st.file_uploader("📄 Upload *Job Data*", type=["xlsx"], key="job_data")
            uploaded_file2 = st.file_uploader("📘 Upload *Skill Taxonomy*", type=["xlsx"], key="taxonomy")
            process_triggered = st.button("🚀 Submit and Process")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file1:
            job_data_df = validate_columns(uploaded_file1, required_columns_file1, "Job Data File")

        if uploaded_file2:           
            skill_taxonomy_df = validate_columns(uploaded_file2, required_columns_file2, "Skill Taxonomy File")
        #if uploaded_file1 and uploaded_file2 and process_triggered:
        if process_triggered:
            st.write("🔄 **Generating a skill map that works for you.**")
            progress_bar = st.progress(0)
            status_text = st.empty()

            client = OpenAI(api_key=st.secrets["OpenAI_Secret_Key"])
            df1 = pd.read_excel(uploaded_file1).head(30)
            df2 = pd.read_excel(uploaded_file2)

            mapping = {
                "Salesforce": "CRM",
                "Infra": "Infrastructure Management",
                "ESM": "ServiceNow"
            }

            def classify_ai_services(job_req, job_desc):
                prompt = f"""
        Job Requisition: {job_req}
        Job Description: {job_desc}

        The cost center is 'AI Services'. Based on the context above, choose the most appropriate category:
        - QA and Testing
        - App Dev
        - Intelligent Automation

        Reply ONLY with the most appropriate category.
        """
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                return response.choices[0].message.content.strip()

            df1["Mapped Cost Center"] = [
                classify_ai_services(row["Job Requisition"], row["Job Description"]) if row["Cost Center"] == "AI Services"
                else mapping.get(row["Cost Center"], row["Cost Center"])
                for _, row in df1.iterrows()
            ]

            def safe_parse_json(content, job_title):
                try:
                    if not content:
                        raise ValueError("Empty response from GPT.")
                    if content.startswith("```"):
                        content = content.strip("` \n")
                        lines = content.splitlines()
                        if lines and lines[0].strip().lower() == "json":
                            lines = lines[1:]
                        content = "\n".join(lines)
                    return json.loads(content)
                except Exception as e:
                    print(f"\n❌ Error parsing JSON for job title: '{job_title}'\nError: {e}")
                    return {
                        "Skill Category": "Unknown",
                        "Level 1": "",
                        "Level 2": "",
                        "Level 3": ""
                    }

            def gpt_full_mapping(row, df2):
                job_title = row["Job Requisition"]
                job_desc = row["Job Description"]
                mapped_cost_center = row["Mapped Cost Center"]
                filtered_df2 = df2[df2["Bucket"].str.lower() == mapped_cost_center.lower()]
                if filtered_df2.empty:
                    filtered_df2 = df2.copy()
                skill_options = filtered_df2.to_dict(orient="records")
                prompt = f"""
        You are an expert HR skill mapping assistant.

        Your task is to assign the correct:
        - Skill Category
        - Level 1
        - Level 2
        - Level 3

        based on the job requisition and job description.

        Only choose from the provided skill options:
        {json.dumps(skill_options, indent=2)}

        --- Job Requisition ---
        Title: "{job_title}"
        Description: "{job_desc}"

        Respond in raw JSON format like:
        {{
        "Skill Category": "...",
        "Level 1": "...",
        "Level 2": "...",
        "Level 3": "..."
        }}
        """
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that maps job roles to predefined skills."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=300
                    )
                    content = response.choices[0].message.content.strip()
                    return safe_parse_json(content, job_title)
                except Exception as e:
                    print(f"Error calling GPT for '{job_title}': {e}")
                    return {
                        "Skill Category": "Unknown",
                        "Level 1": "",
                        "Level 2": "",
                        "Level 3": ""
                    }

            # Apply with progress tracking
            total = len(df1)
            results = []
            for i, (_, row) in enumerate(df1.iterrows()):
                result = gpt_full_mapping(row, df2)
                results.append(result)
                percent_complete = int((i + 1) / total * 100)
                progress_bar.progress(percent_complete)
                status_text.text(f"Processing row {i + 1} of {total} ({percent_complete}%)")

            # Safely convert results to DataFrame
            cleaned_results = [
                res if isinstance(res, dict) else {
                    "Skill Category": "Unknown",
                    "Level 1": "",
                    "Level 2": "",
                    "Level 3": ""
                }
                for res in results
            ]
            results_df = pd.DataFrame(cleaned_results)

            final_df = pd.concat([df1.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False)
            output.seek(0)

            st.success("✅ **Skill mapping complete—insights ready to use.**")
            st.download_button(
                label="📥 Download Mapped Excel",
                data=output,
                file_name="gpt_skill_mapping_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.markdown("### 🔍 Preview of Mapped Roles")
            #AgGrid(final_df.head(10))
            from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

            # Build grid options
            gb = GridOptionsBuilder.from_dataframe(final_df.head(10))
            gb.configure_default_column(filterable=True, sortable=True, resizable=True, editable=False)
            gb.configure_grid_options(domLayout='normal')  # Adjust layout
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()

            # Optional: Add theme or styling
            custom_theme = "streamlit"  # Options: "streamlit", "alpine", "balham", "material"

            AgGrid(
                final_df.head(10),
                gridOptions=gb.build(),
                theme=custom_theme,
                enable_enterprise_modules=False,
                allow_unsafe_jscode=True,
                height=350,
                fit_columns_on_grid_load=True
            )


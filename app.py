import streamlit as st
import polars as pl
import numpy as np
import json
import time
import io
from pathlib import Path
from docx import Document
from datetime import date
from typing import Any, Dict

# Import domain logic and schema from resume_ranker
from resume_ranker.config import config as app_config  # type: ignore
from resume_ranker.domain.schema import Candidate  # type: ignore
from resume_ranker.domain.explanations import RuleReasoner  # type: ignore
from resume_ranker.domain.scoring import (  # type: ignore
    TopNAggregator,
    extract_jd_keywords,
    _WORD_PATTERN
)
from rapidfuzz import fuzz

# Define Paths
ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
FEATURES_PATH = ARTIFACTS_DIR / "candidate_features.parquet"
CANDIDATE_IDS_PATH = ARTIFACTS_DIR / "candidate_ids.npy"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"

# -------------------------------------------------------------
# Caching Functions (Performance Requirement)
# -------------------------------------------------------------
@st.cache_resource
def load_embedding_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer
    try:
        # Try loading locally first (offline-first design)
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except Exception:
        # Fallback to downloading if not cached (useful for cloud deployments like Streamlit Cloud)
        with st.spinner(f"Downloading embedding model '{model_name}' (first-time setup)..."):
            return SentenceTransformer(model_name, device="cpu", local_files_only=False)

@st.cache_resource
def load_faiss_index(path: Path) -> Any:
    import faiss
    return faiss.read_index(str(path))

@st.cache_data
def load_candidate_features(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)

@st.cache_data
def load_candidate_ids(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)  # type: ignore[no-any-return]

@st.cache_data
def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.8rem;
    }
    .metric-value {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #E5E7EB !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
        border-radius: 0.5rem !important;
        margin-bottom: 0.8rem !important;
    }
    .dataframe {
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# Step 1: Verify precomputed artifacts exist
# -------------------------------------------------------------
artifacts_exist = (
    MANIFEST_PATH.exists()
    and FEATURES_PATH.exists()
    and CANDIDATE_IDS_PATH.exists()
    and INDEX_PATH.exists()
)

if not artifacts_exist:
    st.error("Artifacts not found.")
    st.markdown("Please generate artifacts by running the precompute command in your terminal:")
    st.code("uv run precompute --candidates data/input/candidates.jsonl", language="bash")
    st.stop()

# Load manifest first to extract metadata
try:
    manifest = load_manifest(MANIFEST_PATH)
except Exception as e:
    st.error(f"Failed to load manifest.json: {e}")
    st.stop()

# -------------------------------------------------------------
# Step 2: Load the offline embedding model
# -------------------------------------------------------------
model_name = manifest.get("model_name", "BAAI/bge-small-en-v1.5")
model_loaded = False
model = None

try:
    model = load_embedding_model(model_name)
    model_loaded = True
except Exception:
    # Model missing error handling without printing stack trace
    st.error(
        f"Offline embedding model '{model_name}' could not be loaded. "
        "Please ensure the model is pre-cached locally."
    )
    st.info("To cache the model for offline use, run:\n`python scripts/download_model.py`")
    st.stop()

# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ System Info")
    st.markdown(f"**Embedding Model:**\n`{model_name}`")
    st.markdown(f"**Embedding Dim:** `{manifest.get('embedding_dim', 384)}`")
    st.markdown(f"**Indexed Candidates:** `{manifest.get('candidate_count', 0):,}`")
    st.markdown(f"**FAISS Index:** `{manifest.get('faiss_index', 'IndexFlatIP')}`")
    
    st.markdown("---")
    st.markdown("### 🟢 Status")
    if artifacts_exist:
        st.success("Artifacts: Loaded")
    else:
        st.error("Artifacts: Missing")
        
    if model_loaded:
        st.success("Model: Ready (CPU)")
    else:
        st.error("Model: Unavailable")

    # Ranking runtime placeholder
    st.markdown("---")
    st.markdown("### ⚡ Performance")
    if "last_runtime" in st.session_state:
        st.metric("Ranking Runtime", f"{st.session_state['last_runtime']:.3f}s")
    else:
        st.metric("Ranking Runtime", "N/A")

# -------------------------------------------------------------
# Main Panel UI
# -------------------------------------------------------------
st.markdown('<div class="main-title">💼 AI Resume Ranker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production candidate matching & scoring engine powered by FAISS and BGE offline models.</div>', unsafe_allow_html=True)

# File uploader (Step 3)
st.markdown("### 📥 Job Description Upload")
uploaded_file = st.file_uploader(
    "Upload job_description.docx to rank candidates",
    type=["docx"],
    help="Accepts only Microsoft Word (.docx) files"
)

# Initialize container for outputs
if uploaded_file is not None:
    # Extract paragraphs text using python-docx
    try:
        doc_bytes = uploaded_file.read()
        if not doc_bytes:
            st.error("Uploaded file is empty.")
            st.stop()
            
        doc = Document(io.BytesIO(doc_bytes))
        paragraphs = []
        for p in doc.paragraphs:
            text = " ".join(p.text.split())
            if text:
                paragraphs.append(text)
                
        if not paragraphs:
            st.error("Uploaded DOCX contains no extractable paragraph text.")
            st.stop()
            
        jd_text = "\n\n".join(paragraphs)
    except Exception:
        st.error("Uploaded document is invalid or corrupted. Please provide a valid .docx file.")
        st.stop()

    # Display extracted text
    with st.expander("📄 Extracted Job Description Text", expanded=False):
        st.text_area(
            "Job Description",
            jd_text,
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )

    # -------------------------------------------------------------
    # Step 4, 5, 6: Semantic search & Hybrid Re-ranking
    # -------------------------------------------------------------
    try:
        # Load artifacts from cache
        faiss_index = load_faiss_index(INDEX_PATH)
        candidate_ids = load_candidate_ids(CANDIDATE_IDS_PATH)
        candidate_features = load_candidate_features(FEATURES_PATH)
        
        # Start timer
        start_time = time.perf_counter()
        
        # Step 4: Embed the JD text (never candidates)
        jd_embedding = model.encode([jd_text], normalize_embeddings=True)[0]
        
        # Step 5: Perform FAISS search (Top 2000)
        query = np.ascontiguousarray(jd_embedding.reshape(1, -1), dtype=np.float32)
        top_k = min(2000, len(candidate_ids))
        
        scores, indices = faiss_index.search(query, top_k)
        valid = indices[0] >= 0
        search_ids = candidate_ids[indices[0][valid]]
        semantic_scores = scores[0][valid].astype(np.float32)
        
        # Step 6: Load and filter feature table to FAISS candidates
        search_df = pl.DataFrame({
            "candidate_id": search_ids.astype(str),
            "search_idx": np.arange(len(search_ids)),
            "semantic_score": semantic_scores,
        })
        top_features = candidate_features.join(search_df, on="candidate_id").sort("search_idx")
        
        # Apply production scoring calculations matching pipeline.py
        # Vectorized behavioral score
        behavioral_scores = (
            app_config.behavioral_weight_open_to_work * top_features["open_to_work"].to_numpy()
            + app_config.behavioral_weight_notice_period * top_features["notice_period_score"].to_numpy()
            + app_config.behavioral_weight_response_rate * top_features["response_rate"].to_numpy()
            + app_config.behavioral_weight_experience_fit * top_features["experience_fit"].to_numpy()
            + app_config.behavioral_weight_location_fit * top_features["location_fit"].to_numpy()
            + app_config.behavioral_weight_profile_completeness * top_features["profile_completeness"].to_numpy()
            + app_config.behavioral_weight_verified * top_features["verified"].to_numpy()
            + app_config.behavioral_weight_recent_activity * top_features["recent_activity"].to_numpy()
        )
        behavioral_scores = np.clip(behavioral_scores, app_config.score_min, app_config.score_max).astype(np.float32)

        # Vectorized experience score
        years = top_features["years_of_experience"].to_numpy()
        experience_scores: np.ndarray = np.zeros(len(years), dtype=np.float32)
        ideal_mask = (years >= app_config.experience_ideal_min) & (years <= app_config.experience_ideal_max)
        experience_scores[ideal_mask] = app_config.score_max

        below_mask = (years >= app_config.experience_acceptable_min) & (years < app_config.experience_ideal_min)
        span_below = app_config.experience_ideal_min - app_config.experience_acceptable_min
        if span_below > 0:
            experience_scores[below_mask] = (years[below_mask] - app_config.experience_acceptable_min) / span_below

        above_mask = (years > app_config.experience_ideal_max) & (years <= app_config.experience_acceptable_max)
        span_above = app_config.experience_acceptable_max - app_config.experience_ideal_max
        if span_above > 0:
            experience_scores[above_mask] = app_config.score_max - (years[above_mask] - app_config.experience_ideal_max) / span_above

        # Location score
        location_scores = top_features["location_fit"].to_numpy().astype(np.float32)

        # Title & skills overlap heuristics
        title_scores: np.ndarray = np.zeros(len(top_features), dtype=np.float32)
        skill_scores: np.ndarray = np.zeros(len(top_features), dtype=np.float32)

        current_titles = top_features["current_title_text"].to_list()
        skills_jsons = top_features["skills_json"].to_list()
        has_tech_careers = top_features["has_tech_career"].to_numpy()

        jd_keywords = extract_jd_keywords(jd_text)
        jd_keyword_text = " ".join(sorted(jd_keywords))
        
        # Match loop
        for idx in range(len(top_features)):
            # Title overlap
            title_text = current_titles[idx]
            title_tokens = set(_WORD_PATTERN.findall(title_text))
            if not title_tokens:
                title_score = app_config.title_score_no_match
            else:
                overlap = title_tokens & jd_keywords
                overlap_fraction = len(overlap) / len(title_tokens)
                if overlap_fraction >= app_config.score_max:
                    title_score = app_config.title_score_exact_match
                elif overlap_fraction > app_config.score_min:
                    title_score = app_config.title_score_partial_match
                else:
                    title_score = app_config.title_score_no_match

            # Skill overlap
            import orjson
            skills = orjson.loads(skills_jsons[idx])
            if not skills:
                skill_score = app_config.score_min
            else:
                matches = 0.0
                for skill in skills:
                    name = skill["name"].lower()
                    skill_tokens = set(_WORD_PATTERN.findall(name))
                    fuzzy_match = fuzz.partial_ratio(name, jd_keyword_text) >= 85
                    if skill_tokens & jd_keywords or fuzzy_match:
                        if skill["duration_months"] >= app_config.skill_min_duration_months or skill["endorsements"] > 0:
                            matches += 1.0

                skill_score = matches / app_config.skill_match_denominator
                if len(skills) > app_config.skill_stuffing_threshold:
                    penalty = 1.0 - min(
                        (len(skills) - app_config.skill_stuffing_threshold)
                        * app_config.skill_stuffing_penalty_per_skill,
                        app_config.skill_stuffing_max_penalty,
                    )
                    skill_score *= penalty
                skill_score = min(skill_score, app_config.score_max)

            # Career alignment
            if not has_tech_careers[idx]:
                title_score = 0.0
                skill_score = 0.0
                exp_score = 0.0
            else:
                exp_score = float(experience_scores[idx])

            title_scores[idx] = title_score
            skill_scores[idx] = skill_score
            experience_scores[idx] = exp_score

        # Vectorized scores combination matrix
        scores_matrix = np.column_stack([
            top_features["semantic_score"].to_numpy().astype(np.float32),
            skill_scores,
            behavioral_scores,
            experience_scores,
            location_scores,
        ])

        weights = np.array([
            app_config.weight_semantic,
            app_config.weight_skills,
            app_config.weight_behavior,
            app_config.weight_experience,
            app_config.weight_location,
        ], dtype=np.float32)

        final_scores = scores_matrix @ weights
        final_scores = np.clip(final_scores, app_config.score_min, app_config.score_max).astype(np.float32)

        # Lexicographical sort & tie-breaking (Top 100)
        aggregator = TopNAggregator(app_config)
        top_ids, top_scores = aggregator.aggregate_arrays(search_ids, final_scores)
        
        # Stop timer
        runtime = time.perf_counter() - start_time
        st.session_state["last_runtime"] = runtime
        
        # Parse candidate names and construct schema Candidate objects
        candidate_details: Dict[str, Dict[str, Any]] = {}
        for row in top_features.to_dicts():
            c_id = row["candidate_id"]
            cand_obj = Candidate.model_validate_json(row["candidate_json"])
            candidate_details[c_id] = {
                "name": cand_obj.profile.anonymized_name,
                "candidate": cand_obj
            }
            
        # Map IDs to scores for fast lookup
        id_to_scores = {}
        for idx, c_id in enumerate(search_ids):
            id_to_scores[str(c_id)] = {
                "semantic_score": float(semantic_scores[idx]),
                "skill_score": float(skill_scores[idx]),
                "behavioral_score": float(behavioral_scores[idx]),
                "experience_score": float(experience_scores[idx]),
                "location_score": float(location_scores[idx]),
            }

        # Build display rows
        reasoner = RuleReasoner(app_config)
        display_rows = []
        for rank, (c_id, score) in enumerate(zip(top_ids, top_scores), start=1):
            c_id_str = str(c_id)
            details = candidate_details[c_id_str]
            cand = details["candidate"]
            
            # Generate explanation text
            explanation = reasoner.generate(cand, rank, score, date.today())
            sub_scores = id_to_scores[c_id_str]
            
            display_rows.append({
                "Rank": rank,
                "Candidate ID": c_id_str,
                "Candidate Name": details["name"],
                "Final Score": round(float(score), 4),
                "Semantic Score": round(sub_scores["semantic_score"], 4),
                "Behavioral Score": round(sub_scores["behavioral_score"], 4),
                "Experience Score": round(sub_scores["experience_score"], 4),
                "Skills Score": round(sub_scores["skill_score"], 4),
                "Explanation": explanation
            })
            
        # Metrics Calculations
        avg_score = float(np.mean(top_scores))
        highest_score = float(np.max(top_scores))

    except Exception as e:
        st.error(f"Ranking runtime failed to execute: {e}")
        st.stop()

    # -------------------------------------------------------------
    # Step 7: Display Metrics & Results Table
    # -------------------------------------------------------------
    st.markdown("### 📊 Metrics")
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    with col_m1:
        st.metric("Candidates Indexed", f"{manifest.get('candidate_count', 0):,}")
    with col_m2:
        st.metric("Candidates Retrieved", f"{top_k:,}")
    with col_m3:
        st.metric("Candidates Ranked", f"{len(display_rows)}")
    with col_m4:
        st.metric("Ranking Time", f"{runtime:.3f}s")
    with col_m5:
        st.metric("Average Score", f"{avg_score:.4f}")
    with col_m6:
        st.metric("Highest Score", f"{highest_score:.4f}")

    st.markdown("### 🏆 Top Ranked Resumes")
    results_df = pl.DataFrame(display_rows)
    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Final Score": st.column_config.NumberColumn(format="%.4f"),
            "Semantic Score": st.column_config.NumberColumn(format="%.4f"),
            "Behavioral Score": st.column_config.NumberColumn(format="%.4f"),
            "Experience Score": st.column_config.NumberColumn(format="%.4f"),
            "Skills Score": st.column_config.NumberColumn(format="%.4f"),
        }
    )

    # -------------------------------------------------------------
    # Step 8: Detailed Candidate Profile Inspector
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 👤 Detailed Candidate Profile")
    
    display_rows_dict = {row["Candidate ID"]: row for row in display_rows}
    
    selected_cid = st.selectbox(
        "Select a candidate to view their complete profile and score breakdown:",
        options=[row["Candidate ID"] for row in display_rows],
        format_func=lambda cid: f"Rank {display_rows_dict[cid]['Rank']}: {display_rows_dict[cid]['Candidate Name']} ({cid})"
    )

    if selected_cid:
        selected_cid_str = str(selected_cid)
        details = candidate_details[selected_cid_str]
        cand = details["candidate"]
        row_data = display_rows_dict[selected_cid_str]
        
        st.markdown(f"#### Profile for **{row_data['Candidate Name']}** ({selected_cid_str})")
        
        # Tabs for layout structure
        tab_exp, tab_skills, tab_edu, tab_proj, tab_behavior, tab_expl = st.tabs([
            "💼 Experience",
            "🛠️ Skills",
            "🎓 Education",
            "📁 Projects & Other",
            "📊 Behavioral Signals",
            "💬 Explanation"
        ])
        
        with tab_exp:
            st.markdown(f"**Headline:** {cand.profile.headline}")
            st.markdown(f"**Current Role:** {cand.profile.current_title} at *{cand.profile.current_company}*")
            st.markdown("---")
            for entry in cand.career_history:
                end_str = entry.end_date.strftime("%Y-%m-%d") if entry.end_date else "Present"
                st.markdown(f"**{entry.title}** | {entry.company} *({entry.start_date.strftime('%Y-%m-%d')} to {end_str})*")
                st.markdown(f"Industry: `{entry.industry}` | Company Size: `{entry.company_size}`")
                if entry.description:
                    st.caption(entry.description)
                st.markdown("---")
                
        with tab_skills:
            st.markdown("##### Technical & Professional Skills")
            skills_list = []
            for s in cand.skills:
                skills_list.append({
                    "Skill": s.name,
                    "Proficiency": s.proficiency,
                    "Duration (Months)": s.duration_months,
                    "Endorsements": s.endorsements
                })
            if skills_list:
                st.dataframe(pl.DataFrame(skills_list), width="stretch", hide_index=True)
            else:
                st.write("No skills data available.")

        with tab_edu:
            st.markdown("##### Education History")
            for edu in cand.education:
                st.markdown(f"**{edu.degree}** in *{edu.field_of_study}*")
                st.markdown(f"{edu.institution} ({edu.start_year} - {edu.end_year})")
                if edu.grade:
                    st.markdown(f"Grade: `{edu.grade}`")
                if edu.tier:
                    st.markdown(f"Institution Tier: `{edu.tier}`")
                st.markdown("---")

        with tab_proj:
            # Projects field is not directly schema-supported. Check Certifications/Languages.
            st.markdown("##### Projects")
            st.info("No projects field is directly schema-coded in the candidate's JSON profile.")
            
            st.markdown("##### Certifications")
            if cand.certifications:
                for cert in cand.certifications:
                    st.markdown(f"- **{cert.name}** issued by *{cert.issuer}* ({cert.year})")
            else:
                st.write("No certifications listed.")
                
            st.markdown("##### Languages")
            if cand.languages:
                for lang in cand.languages:
                    st.markdown(f"- **{lang.language}** ({lang.proficiency})")
            else:
                st.write("No languages listed.")

        with tab_behavior:
            st.markdown("##### Behavioral Signals & Performance Indicators")
            sig = cand.redrob_signals
            cb1, cb2 = st.columns(2)
            with cb1:
                st.metric("Profile Completeness", f"{sig.profile_completeness_score:.1f}%")
                st.metric("Recruiter Response Rate", f"{sig.recruiter_response_rate:.1%}")
                st.metric("Notice Period (Days)", f"{sig.notice_period_days}")
                st.metric("Applications (30 Days)", f"{sig.applications_submitted_30d}")
            with cb2:
                st.metric("Open to Work", "Yes" if sig.open_to_work_flag else "No")
                st.metric("Willing to Relocate", "Yes" if sig.willing_to_relocate else "No")
                st.metric("GitHub Score", f"{sig.github_activity_score:.1f}" if sig.github_activity_score >= 0 else "Unlinked")
                st.metric("Connections", f"{sig.connection_count:,}")

        with tab_expl:
            st.markdown("##### Explanation & Score Drivers")
            st.info(row_data["Explanation"])

    # -------------------------------------------------------------
    # Step 9: Download submission.csv
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 💾 Export Submission File")
    submission_df = pl.DataFrame({
        "candidate_id": [row["Candidate ID"] for row in display_rows],
        "rank": [row["Rank"] for row in display_rows],
        "score": [row["Final Score"] for row in display_rows],
        "reasoning": [row["Explanation"] for row in display_rows]
    })
    
    csv_data = submission_df.write_csv()
    st.download_button(
        label="📥 Download submission.csv",
        data=csv_data,
        file_name="submission.csv",
        mime="text/csv",
        help="Download ranked list in production submission CSV format"
    )

else:
    # Prompt the user to upload a file to begin ranking
    st.info("👈 Please upload a job description (.docx) to start the ranking process.")
    
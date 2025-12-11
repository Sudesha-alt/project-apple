# streamlit_lead_bucket_app.py (updated)
# Streamlit app: Lead bucketing + system-message generation
# Changes made:
# - When generating system messages, the output CSVs use two fields: lead_id and message_content
# - Removed Excel outputs for system messages; now CSV per-bucket + ZIP of CSVs

import streamlit as st
import pandas as pd
import io
import zipfile
from typing import Dict, List

st.set_page_config(page_title="Ariseall Lead bucketer", layout="wide")
st.title("Ariseall Lead bucketer")
st.write("Upload your lead sheet (CSV or Excel). Map columns if automatic detection isn't perfect. Choose a feature from the sidebar.")

# Helpful variants for automatic column suggestion
COLUMN_VARIANTS = {
    "lead_id": ["lead_id", "lead id", "id", "user_id", "userid"],
    "user_id": ["user_id", "id", "lead_id", "userid"],
    "first_name": ["first_name", "firstname", "name", "first name"],
    "phone_numbers": ["phone", "phone_number", "phone_numbers", "mobile", "mobile_no", "mobile_no."],
    "class_code": ["class_code", "classcode", "class", "class id"],
    "lead_source": ["lead_source", "source", "lead source", "channel"],
    "mandate_date": ["mandate_date", "mandate date", "mandate"],
    "autopay_status": ["autopay_status", "autopay", "auto_pay", "autopay status"],
    "cancelled_date": ["cancelled_date", "cancelled date", "cancel_date", "cancellation_date"],
    "plan_id": ["plan_id", "planid"],
    "monthly_fees": ["monthly_fees", "monthly fees", "fees", "monthly_fee"],
    "user_set": ["user_set", "user set", "userset", "set"],
    "demo_role": ["demo_role", "demo role"],
    "first_attendance_date": ["first_attendance_date", "first attendance date"],
    "first_class_completed": ["first_class_completed", "first class completed"],
    "total_attendances": ["total_attendances", "total attendance"],
    "total_completed_attendances": ["total_completed_attendances", "completed attendances"],
    "started_yesterday": ["started_yesterday", "started yesterday", "started_yday"],
    "completed_yesterday": ["completed_yesterday", "completed yesterday", "completed_yday"]
}


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


def suggest_column(df: pd.DataFrame, variants: List[str]):
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]
    for v in variants:
        if v.lower() in lower_cols:
            return cols[lower_cols.index(v.lower())]
    return None


def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for key, variants in COLUMN_VARIANTS.items():
        suggestion = suggest_column(df, variants)
        mapping[key] = suggestion
    return mapping

@st.cache_data
def classify_df(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Work on a copy
    working = df.copy()

    # Normalize lead_source and user_set to strings for comparison.
    # Treat "facebook" / "fb" as "meta" so they map to Meta buckets.
    if 'lead_source' in working.columns:
        working['lead_source_norm'] = (
            working['lead_source']
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"facebook": "meta", "fb": "meta"})
        )
    else:
        working['lead_source_norm'] = ''

    if 'user_set' in working.columns:
        working['user_set_norm'] = working['user_set'].astype(str).str.strip().str.lower()
    else:
        working['user_set_norm'] = ''

    google_bucket = working[working['lead_source_norm'] == 'google'].drop(columns=['lead_source_norm', 'user_set_norm'])

    meta_main_bucket = working[(working['lead_source_norm'] == 'meta') & (working['user_set_norm'] == 'main_set')].drop(columns=['lead_source_norm', 'user_set_norm'])

    meta_experiment_bucket = working[(working['lead_source_norm'] == 'meta') & (working['user_set_norm'] == 'experiment')].drop(columns=['lead_source_norm', 'user_set_norm'])

    others_bucket = working[~working.index.isin(google_bucket.index) & ~working.index.isin(meta_main_bucket.index) & ~working.index.isin(meta_experiment_bucket.index)].drop(columns=['lead_source_norm', 'user_set_norm'])

    return {
        'google_bucket': google_bucket,
        'meta_main_bucket': meta_main_bucket,
        'meta_experiment_bucket': meta_experiment_bucket,
        'others_bucket': others_bucket
    }


def generate_system_messages_for_bucket(df: pd.DataFrame, lead_id_col: str = 'lead_id') -> pd.DataFrame:
    """
    Produce a DataFrame with exactly two columns: 'lead_id' and 'message_content'.
    If the source df has a column named lead_id_col, use it; otherwise fall back to index string.
    """
    working = df.copy()
    # Ensure presence of status columns
    for col in ['autopay_status', 'started_yesterday', 'completed_yesterday']:
        if col not in working.columns:
            working[col] = ''

    # Determine values for lead_id
    if lead_id_col in working.columns:
        lead_ids = working[lead_id_col].astype(str)
    elif 'lead_id' in working.columns:
        lead_ids = working['lead_id'].astype(str)
    else:
        lead_ids = working.index.astype(str)

    def make_message(row):
        a = str(row.get('autopay_status', '')).strip()
        s = str(row.get('started_yesterday', '')).strip()
        c = str(row.get('completed_yesterday', '')).strip()
        return f"autopay_status={a}, started_yesterday={s}, completed_yesterday={c}"

    messages = working.apply(make_message, axis=1)

    out = pd.DataFrame({
        'lead_id': lead_ids.values,
        'message_content': messages.values
    })

    return out

# Sidebar: choose feature
feature = st.sidebar.selectbox("Choose feature", ["Bucket & download (existing)", "Generate system messages per bucket"])

# File upload
uploaded_file = st.file_uploader("Upload lead sheet (CSV or XLSX)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read the file: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Automatic mapping suggestions
    suggested = auto_map_columns(df)

    st.subheader("Column mapping")
    st.write("Streamlit tried to auto-detect common columns. If a mapping is empty or incorrect, please select the correct column. Ensure you map the columns needed for the selected feature.")

    cols = [''] + list(df.columns)
    # Show key mappings
    col1, col2 = st.columns(2)
    user_mapping = {}
    with col1:
        user_mapping['lead_source'] = st.selectbox("lead_source column", cols, index=cols.index(suggested.get('lead_source')) if suggested.get('lead_source') in cols else 0)
        user_mapping['user_set'] = st.selectbox("user_set column", cols, index=cols.index(suggested.get('user_set')) if suggested.get('user_set') in cols else 0)
        user_mapping['lead_id'] = st.selectbox("lead_id column", cols, index=cols.index(suggested.get('lead_id')) if suggested.get('lead_id') in cols else 0)
    with col2:
        user_mapping['autopay_status'] = st.selectbox("autopay_status column", cols, index=cols.index(suggested.get('autopay_status')) if suggested.get('autopay_status') in cols else 0)
        user_mapping['started_yesterday'] = st.selectbox("started_yesterday column", cols, index=cols.index(suggested.get('started_yesterday')) if suggested.get('started_yesterday') in cols else 0)
        user_mapping['completed_yesterday'] = st.selectbox("completed_yesterday column", cols, index=cols.index(suggested.get('completed_yesterday')) if suggested.get('completed_yesterday') in cols else 0)

    # If user selected columns, rename them to standard names used in classification & system generation
    working_df = df.copy()
    rename_map = {}
    for std_col, chosen in user_mapping.items():
        if chosen and chosen != '':
            if chosen != std_col:
                rename_map[chosen] = std_col
    if rename_map:
        working_df = working_df.rename(columns=rename_map)

    st.markdown("---")
    st.write("### Preview (first 5 rows)")
    st.dataframe(working_df.head())

    # Run according to feature
    if feature == "Bucket & download (existing)":
        if st.button("Classify leads into buckets"):
            buckets = classify_df(working_df)

            st.success("Classification complete.")
            counts = {k: len(v) for k, v in buckets.items()}
            st.write("**Counts:**")
            st.json(counts)

            # Show preview and download buttons
            for key, bdf in buckets.items():
                st.subheader(f"{key} — {len(bdf)} leads")
                st.dataframe(bdf.head(10))

                # CSV download
                csv_bytes = bdf.to_csv(index=False).encode('utf-8')
                st.download_button(label=f"Download {key}.csv", data=csv_bytes, file_name=f"{key}.csv", mime='text/csv')

            # Option: download all as zip
            if st.button("Download all buckets as ZIP"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for key, bdf in buckets.items():
                        csv_data = bdf.to_csv(index=False).encode('utf-8')
                        zf.writestr(f"{key}.csv", csv_data)
                zip_buffer.seek(0)
                st.download_button(label="Download ZIP of all buckets", data=zip_buffer, file_name="lead_buckets.zip", mime='application/zip')

    else:  # Generate system messages per bucket
        if st.button("Generate system messages for all buckets"):
            buckets = classify_df(working_df)
            # Determine actual lead_id column name present in working_df
            lead_id_col = 'lead_id' if 'lead_id' in working_df.columns else (user_mapping.get('lead_id') or 'lead_id')

            outputs = {}
            for key, bdf in buckets.items():
                # Ensure lead_id exists in bucket (we'll handle inside generator)
                system_df = generate_system_messages_for_bucket(bdf, lead_id_col)
                outputs[key] = system_df

            st.success("System messages generated for each bucket.")
            counts = {k: len(v) for k, v in outputs.items()}
            st.write("**Counts:**")
            st.json(counts)

            # Provide download for each bucket as CSV (two columns: lead_id, message_content)
            for key, outdf in outputs.items():
                st.subheader(f"{key} — {len(outdf)} leads")
                st.dataframe(outdf.head(10))

                csv_bytes = outdf.to_csv(index=False).encode('utf-8')
                st.download_button(label=f"Download {key}.csv", data=csv_bytes, file_name=f"{key}.csv", mime='text/csv')

            # Option: download all as a single ZIP of CSVs
            if st.button("Download all system-message CSVs as ZIP"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for key, outdf in outputs.items():
                        csv_data = outdf.to_csv(index=False).encode('utf-8')
                        zf.writestr(f"{key}.csv", csv_data)
                zip_buffer.seek(0)
                st.download_button(label="Download ZIP of all bucket CSVs", data=zip_buffer, file_name="system_messages_buckets.zip", mime='application/zip')

    st.markdown("---")
    st.write("Need extra features? Reply with: 'export to separate sheets in one Excel' or 'filter where monthly_fees > 0' or 'add other rules' and I'll extend the app.")

else:
    st.info("Upload a CSV or Excel file to get started.")

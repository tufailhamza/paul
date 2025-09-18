# import streamlit as st
# import os
# import pandas as pd
# from schemas.normalizers import validate_and_normalize

# # ----------------------------
# # Setup
# # ----------------------------
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# st.set_page_config(page_title="Kihei Seller Scoring", layout="wide")
# st.title("Kihei Seller Scoring - Upload & Validate")

# # ----------------------------
# # File uploader
# # ----------------------------
# uploaded_files = st.file_uploader(
#     "Upload MLS, MAPPS, Batch Leads, or eCourt/BOC files",
#     type=["csv", "xlsx"],
#     accept_multiple_files=True
# )

# # ----------------------------
# # Process each uploaded file
# # ----------------------------
# if uploaded_files:
#     for file in uploaded_files:
#         file_path = os.path.join(UPLOAD_FOLDER, file.name)
#         with open(file_path, "wb") as f:
#             f.write(file.getbuffer())
#         st.success(f"Uploaded {file.name} to {file_path}")

#         # Decide type based on filename
#         fname = file.name.lower()
#         if "mls" in fname:
#             hint = "mls"
#         elif "mapps" in fname or "permit" in fname:
#             hint = "mapps"
#         elif "batch" in fname or "lead" in fname:
#             hint = "batch"
#         elif "court" in fname or "ecourt" in fname or "boc" in fname:
#             hint = "ecourt"
#         else:
#             hint = "mls"  # fallback if unknown

#         # Normalize + validate
#         try:
#             df_norm, messages = validate_and_normalize(file_path, hint)

#             if messages:
#                 st.warning(f"Validation/Normalization messages for {file.name}:")
#                 for msg in messages:
#                     st.write("- " + str(msg))
#             else:
#                 st.success(f"{file.name} validated and normalized successfully.")

#             st.dataframe(df_norm.head(10))

#             # Save normalized CSV
#             norm_path = os.path.join(UPLOAD_FOLDER, f"normalized__{file.name}.csv")
#             df_norm.to_csv(norm_path, index=False)
#             st.info(f"Normalized CSV saved to: {norm_path}")

#         except Exception as e:
#             st.error(f"Failed to process {file.name}: {e}")

import streamlit as st
import os
import pandas as pd
from schemas.normalizers import validate_and_normalize

# ✅ New: DB ingestion functions
from ingest.ingest import upsert_properties, upsert_permits, upsert_legal_events

# ----------------------------
# Setup
# ----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="Kihei Seller Scoring", layout="wide")
st.title("Kihei Seller Scoring - Upload, Validate & Ingest")

# ----------------------------
# File uploader
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload MLS, MAPPS, Batch Leads, or eCourt/BOC files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

# ----------------------------
# Process each uploaded file
# ----------------------------
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Uploaded {file.name} to {file_path}")

        # Decide type based on filename
        fname = file.name.lower()
        if "mls" in fname:
            hint = "mls"
        elif "mapps" in fname or "permit" in fname:
            hint = "mapps"
        elif "batch" in fname or "lead" in fname:
            hint = "batch"
        elif "court" in fname or "ecourt" in fname or "boc" in fname:
            hint = "ecourt"
        else:
            hint = "mls"  # fallback if unknown

        # Normalize + validate
        try:
            df_norm, messages = validate_and_normalize(file_path, hint)

            if messages:
                st.warning(f"Validation/Normalization messages for {file.name}:")
                for msg in messages:
                    st.write("- " + str(msg))
            else:
                st.success(f"{file.name} validated and normalized successfully.")

            st.dataframe(df_norm.head(10))

            # Save normalized CSV
            norm_path = os.path.join(UPLOAD_FOLDER, f"normalized__{file.name}.csv")
            df_norm.to_csv(norm_path, index=False)
            st.info(f"Normalized CSV saved to: {norm_path}")

            print("----")
            # ✅ New: Ingest to DB
            ingest_res = {}
            if hint == "mls" or hint == "batch":
                ingest_res = upsert_properties(df_norm)
            elif hint == "mapps":
                ingest_res = upsert_permits(df_norm)
            else:  # ecourt
                ingest_res = upsert_legal_events(df_norm)

            print("////")
            st.success(f"Ingestion complete for {file.name}: {ingest_res}")

        except Exception as e:
            st.error(f"Failed to process {file.name}: {e}")

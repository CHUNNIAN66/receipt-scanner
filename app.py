# FINAL SaaS-READY RECEIPT SCANNER APP
# app.py

import os
import json
import base64
import requests
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI
from supabase import create_client

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


BASE_PATH = "/tmp/Receipt_Records"


# =========================
# SUPABASE
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================
# HELPERS
# =========================
def get_financial_year(date):
    if date.month >= 7:
        return f"FY{date.year}-{date.year + 1}"
    return f"FY{date.year - 1}-{date.year}"


def safe_name(name):
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


# =========================
# IMAGE COMPRESSION
# =========================
def compress_image_for_processing(image):
    image = ImageOps.exif_transpose(image)
    image.thumbnail((1200, 1200))
    return image


# =========================
# GPT VISION RECEIPT PARSER
# =========================
def parse_receipt_image_with_gpt(image):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    buffer = BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=70,
        optimize=True
    )

    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = """
You are reading an Australian retail receipt image.

Return ONLY valid JSON array.

Each object must have:
[
  {
    "Item": "item description",
    "Qty": "",
    "Unit Price": "",
    "Amount": 0.00
  }
]

Rules:
- Extract all purchased line items.
- Do not include subtotal, total, GST, payment, cash, change.
- Preserve item descriptions accurately.
- Amount must be line item amount.
- Return Amount as number.
- Do not invent items.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    raw = response.output_text.strip()

    try:
        return json.loads(raw)
    except Exception:
        st.error("GPT returned invalid JSON")
        st.text(raw)
        return []


# =========================
# GOOGLE OAUTH
# =========================
def create_google_flow():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": st.secrets["auth"]["client_id"],
                "client_secret": st.secrets["auth"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["auth"]["redirect_uri"]],
            }
        },
        scopes=[
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid"
        ],
        redirect_uri=st.secrets["auth"]["redirect_uri"],
        autogenerate_code_verifier=False
    )

    return flow


# =========================
# GET GOOGLE USER EMAIL
# =========================
def get_google_user_email(credentials):
    response = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={
            "Authorization": f"Bearer {credentials.token}"
        }
    )

    return response.json().get("email")


# =========================
# GOOGLE LOGIN
# =========================
def google_login_section():
    st.subheader("Google Drive Login")

    if "google_credentials" in st.session_state:
        st.success("✅ Google Drive connected")
        return

    query_params = st.query_params

    if "code" in query_params:
        try:
            code = query_params["code"]

            flow = create_google_flow()
            flow.fetch_token(code=code)

            st.session_state["google_credentials"] = flow.credentials
            
            user_email = get_google_user_email(flow.credentials)
            
            allowed = supabase.table("allowed_users") \
                .select("*") \
                .eq("email", user_email) \
                .execute()
            
            if not allowed.data:
                st.error("You are not authorized to use this app.")
                st.stop()
            
            st.session_state["user_email"] = user_email
            
            ensure_user_setup(user_email)
            
            st.query_params.clear()

            st.success(f"✅ Logged in as {user_email}")
            st.rerun()

        except Exception as e:
            st.error("Google login failed")
            st.exception(e)
            return

    flow = create_google_flow()

    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    st.markdown(f"[Login with Google Drive]({auth_url})")


# =========================
# DRIVE SERVICE
# =========================
def get_drive_service_oauth():
    credentials = st.session_state["google_credentials"]
    return build("drive", "v3", credentials=credentials)


# =========================
# CREATE DRIVE FOLDER
# =========================
def create_drive_folder(folder_name, parent_folder_id=None):
    service = get_drive_service_oauth()

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder"
    }

    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    folder = service.files().create(
        body=metadata,
        fields="id"
    ).execute()

    return folder.get("id")


# =========================
# USER SETUP
# =========================
def ensure_user_setup(user_email):
    existing = supabase.table("users") \
        .select("*") \
        .eq("email", user_email) \
        .execute()

    if existing.data:
        return

    root_folder_id = create_drive_folder("Receipt Scanner")

    supabase.table("users").insert({
        "email": user_email,
        "google_drive_folder_id": root_folder_id,
        "google_refresh_token": st.session_state["google_credentials"].refresh_token
    }).execute()


# =========================
# GET USER ROOT FOLDER
# =========================
def get_user_root_folder_id(user_email):
    result = supabase.table("users") \
        .select("google_drive_folder_id") \
        .eq("email", user_email) \
        .execute()

    return result.data[0]["google_drive_folder_id"]


# =========================
# GET OR CREATE SUBFOLDER
# =========================
def get_or_create_drive_folder(folder_name, parent_folder_id):
    service = get_drive_service_oauth()

    query = (
        f"name='{folder_name}' "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and '{parent_folder_id}' in parents "
        f"and trashed=false"
    )

    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])

    if files:
        return files[0]["id"]

    return create_drive_folder(folder_name, parent_folder_id)


# =========================
# RECEIPT IMAGE FOLDER
# =========================
def get_receipt_image_folder_id(root_folder_id, date_obj, store):
    fy = get_financial_year(date_obj)

    images_folder_id = get_or_create_drive_folder(
        "Receipt Images",
        root_folder_id
    )

    fy_folder_id = get_or_create_drive_folder(
        fy,
        images_folder_id
    )

    store_folder_id = get_or_create_drive_folder(
        store,
        fy_folder_id
    )

    return store_folder_id


# =========================
# SAVE IMAGE LOCAL
# =========================
def save_image_local(original_image, store, date, total="unknown"):
    fy = get_financial_year(date)

    folder = os.path.join(BASE_PATH, fy, safe_name(store))
    os.makedirs(folder, exist_ok=True)

    filename = f"{date.strftime('%Y-%m-%d')}_{safe_name(store)}_{total}.jpg"

    path = os.path.join(folder, filename)

    original_image.convert("RGB").save(
        path,
        format="JPEG",
        quality=55,
        optimize=True
    )

    return path


# =========================
# SAVE TO EXCEL
# =========================
def save_to_excel(rows, date):
    fy = get_financial_year(date)

    folder = os.path.join(BASE_PATH, fy, "Excel")
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{fy}_Expense_Receipts.xlsx")

    new_df = pd.DataFrame(rows)

    if os.path.exists(file_path):
        old_df = pd.read_excel(file_path)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_excel(file_path, index=False)

    return file_path


# =========================
# DRIVE UPLOAD
# =========================
def upload_to_drive(file_path, file_name, folder_id):
    service = get_drive_service_oauth()

    file_metadata = {
        "name": file_name,
        "parents": [folder_id],
    }

    media = MediaFileUpload(file_path, resumable=True)

    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
    ).execute()

    return uploaded_file.get("id")


# =========================
# FIND FILE
# =========================
def find_drive_file(file_name, folder_id):
    service = get_drive_service_oauth()

    query = (
        f"name='{file_name}' "
        f"and '{folder_id}' in parents "
        f"and trashed=false"
    )

    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name, modifiedTime)"
    ).execute()

    files = results.get("files", [])

    if files:
        return files[0]["id"]

    return None


# =========================
# UPDATE FILE
# =========================
def update_drive_file(file_id, local_path):
    service = get_drive_service_oauth()

    media = MediaFileUpload(
        local_path,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        resumable=True
    )

    updated_file = service.files().update(
        fileId=file_id,
        media_body=media,
        fields="id"
    ).execute()

    return updated_file.get("id")


# =========================
# UPLOAD OR UPDATE EXCEL
# =========================
def upload_or_update_excel_to_drive(excel_path, file_name, folder_id):
    existing_file_id = find_drive_file(file_name, folder_id)

    if existing_file_id:
        update_drive_file(existing_file_id, excel_path)
        return existing_file_id

    return upload_to_drive(excel_path, file_name, folder_id)


# =========================
# UI
# =========================
st.set_page_config(page_title="Receipt Scanner", layout="wide")

st.title("📸 Receipt Scanner SaaS")


google_login_section()

if "google_credentials" not in st.session_state:
    st.stop()


uploaded_file = st.file_uploader(
    "Take or upload receipt photo",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:
    original_image = Image.open(uploaded_file)

    original_image = compress_image_for_processing(original_image)

    st.image(original_image, use_container_width=True)

    if st.button("🤖 Read Receipt"):
        with st.spinner("Reading receipt..."):
            items = parse_receipt_image_with_gpt(original_image)

        if items:
            st.session_state["items_df"] = pd.DataFrame(items)
        else:
            st.session_state["items_df"] = pd.DataFrame([
                {
                    "Item": "",
                    "Qty": "",
                    "Unit Price": "",
                    "Amount": 0.0
                }
            ])

    if "items_df" in st.session_state:
        st.subheader("Edit Receipt Items")

        edited_df = st.data_editor(
            st.session_state["items_df"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True
        )

        store = st.text_input("Store", value="Bunnings")
        category = st.text_input("Category", value="Materials")
        receipt_date = st.date_input("Receipt Date", value=datetime.today())

        edited_df["Amount"] = pd.to_numeric(
            edited_df["Amount"],
            errors="coerce"
        ).fillna(0)

        total_amount = edited_df["Amount"].sum()

        st.info(f"Total: ${total_amount:.2f}")

        if st.button("✅ Confirm and Save"):
            date_obj = datetime.combine(receipt_date, datetime.min.time())

            image_path = save_image_local(
                original_image,
                store,
                date_obj,
                total_amount
            )

            rows = []

            for _, row in edited_df.iterrows():
                rows.append({
                    "Date": date_obj.strftime("%Y-%m-%d"),
                    "Store": store,
                    "Item": row.get("Item", ""),
                    "Qty": row.get("Qty", ""),
                    "Unit Price": row.get("Unit Price", ""),
                    "Amount": row.get("Amount", ""),
                    "Category": category,
                    "Image Path": os.path.basename(image_path)
                })

            excel_path = save_to_excel(rows, date_obj)

            user_email = st.session_state["user_email"]

            root_folder_id = get_user_root_folder_id(user_email)

            excel_drive_id = upload_or_update_excel_to_drive(
                excel_path,
                os.path.basename(excel_path),
                root_folder_id
            )

            receipt_image_folder_id = get_receipt_image_folder_id(
                root_folder_id,
                date_obj,
                store
            )

            image_drive_id = upload_to_drive(
                image_path,
                os.path.basename(image_path),
                receipt_image_folder_id
            )

            st.success("✅ Receipt saved successfully")

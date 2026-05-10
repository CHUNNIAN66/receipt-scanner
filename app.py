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
from googleapiclient.http import MediaIoBaseDownload


BASE_PATH = "/tmp/Receipt_Records"


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Receipt Scanner",
    layout="wide"
)


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

Return ONLY valid JSON array. No markdown. No explanation.

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
- Do not include ABN, phone number, footer, barcode, rewards text, advertising text.
- Preserve item descriptions accurately.
- Amount must be the line item amount, not the receipt total.
- If quantity and unit price are visible, fill them.
- If quantity or unit price are unclear, keep them empty.
- If amount is unclear, use 0.
- Return Amount as number, not string.
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

    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")
    raw = raw.strip()

    try:
        data = json.loads(raw)

        if isinstance(data, list):
            return data

        return []

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


def get_google_user_email(credentials):
    response = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={
            "Authorization": f"Bearer {credentials.token}"
        }
    )

    return response.json().get("email")


def google_login_section():
    st.subheader("Google Drive Login")

    if "google_credentials" in st.session_state:
        st.success(f"✅ Google Drive connected: {st.session_state.get('user_email', '')}")
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
# GOOGLE DRIVE
# =========================
def get_drive_service_oauth():
    credentials = st.session_state["google_credentials"]
    return build("drive", "v3", credentials=credentials)


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
        files = sorted(files, key=lambda x: x.get("modifiedTime", ""), reverse=True)
        return files[0]["id"]

    return None


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


def upload_or_update_excel_to_drive(excel_path, file_name, folder_id):
    existing_file_id = find_drive_file(file_name, folder_id)

    if existing_file_id:
        update_drive_file(existing_file_id, excel_path)
        return existing_file_id

    return upload_to_drive(excel_path, file_name, folder_id)


# =========================
# SUPABASE USER SETUP
# =========================
def ensure_user_setup(user_email):
    existing = supabase.table("users") \
        .select("*") \
        .eq("email", user_email) \
        .execute()

    if existing.data:
        return

    root_folder_id = create_drive_folder("Receipt Scanner")

    refresh_token = st.session_state["google_credentials"].refresh_token

    supabase.table("users").insert({
        "email": user_email,
        "google_drive_folder_id": root_folder_id,
        "google_refresh_token": refresh_token
    }).execute()


def get_user_root_folder_id(user_email):
    result = supabase.table("users") \
        .select("google_drive_folder_id") \
        .eq("email", user_email) \
        .execute()

    if not result.data:
        ensure_user_setup(user_email)

        result = supabase.table("users") \
            .select("google_drive_folder_id") \
            .eq("email", user_email) \
            .execute()

    return result.data[0]["google_drive_folder_id"]


# =========================
# LOCAL SAVE
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


def save_to_excel(rows, date, root_folder_id):
    fy = get_financial_year(date)

    folder = os.path.join(BASE_PATH, fy, "Excel")
    os.makedirs(folder, exist_ok=True)

    file_name = f"{fy}_Expense_Receipts.xlsx"
    file_path = os.path.join(folder, file_name)

    new_df = pd.DataFrame(rows)

    existing_file_id = find_drive_file(
        file_name,
        root_folder_id
    )

    # =========================
    # DOWNLOAD EXISTING EXCEL
    # =========================
    if existing_file_id:
        service = get_drive_service_oauth()

        request = service.files().get_media(
            fileId=existing_file_id
        )

        with open(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)

            done = False

            while not done:
                status, done = downloader.next_chunk()

        try:
            old_df = pd.read_excel(file_path)

            final_df = pd.concat(
                [old_df, new_df],
                ignore_index=True
            )

        except Exception:
            final_df = new_df

    else:
        final_df = new_df

    # =========================
    # SAVE LOCAL
    # =========================
    final_df.to_excel(file_path, index=False)

    return file_path


# =========================
# UI
# =========================
st.title("📸 Receipt Scanner")

google_login_section()

if "google_credentials" not in st.session_state:
    st.stop()


uploaded_file = st.file_uploader(
    "Take or upload receipt photo",
    type=["jpg", "jpeg", "png"],
    help="On mobile, choose Camera to take a receipt photo directly."
)


if uploaded_file:
    original_image = Image.open(uploaded_file)
    original_image = compress_image_for_processing(original_image)

    st.subheader("Receipt Image")
    st.image(original_image, use_container_width=True)

    if st.button("🤖 Read Receipt", use_container_width=True):
        with st.spinner("Reading receipt with GPT Vision..."):
            items = parse_receipt_image_with_gpt(original_image)

        if items:
            st.session_state["items_df"] = pd.DataFrame(items)
            st.success(f"Found {len(items)} item rows")
        else:
            st.session_state["items_df"] = pd.DataFrame([
                {
                    "Item": "",
                    "Qty": "",
                    "Unit Price": "",
                    "Amount": 0.0
                }
            ])
            st.warning("No items detected. You can manually enter items below.")

    if "items_df" in st.session_state:
        st.subheader("Edit Receipt Items")

        required_columns = ["Item", "Qty", "Unit Price", "Amount"]

        for col in required_columns:
            if col not in st.session_state["items_df"].columns:
                st.session_state["items_df"][col] = ""

        st.session_state["items_df"] = st.session_state["items_df"][required_columns]

        edited_df = st.data_editor(
            st.session_state["items_df"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Item": st.column_config.TextColumn(
                    "Item",
                    width="large"
                ),
                "Qty": st.column_config.TextColumn(
                    "Qty",
                    width="small"
                ),
                "Unit Price": st.column_config.TextColumn(
                    "Unit Price",
                    width="small"
                ),
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    min_value=0.0,
                    step=0.01,
                    format="$%.2f"
                ),
            }
        )

        if edited_df.empty or "Amount" not in edited_df.columns:
            edited_df = pd.DataFrame([
                {
                    "Item": "",
                    "Qty": "",
                    "Unit Price": "",
                    "Amount": 0.0
                }
            ])

        st.subheader("Receipt Details")

        store = st.text_input("Store", value="Bunnings")
        category = st.text_input("Category", value="Materials")
        project = st.text_input("Project / Investor", value="")
        payment_method = st.text_input("Payment Method", value="")
        receipt_date = st.date_input("Receipt Date", value=datetime.today())

        edited_df["Amount"] = pd.to_numeric(
            edited_df["Amount"],
            errors="coerce"
        ).fillna(0)

        total_amount = edited_df["Amount"].sum()

        st.info(f"Total: ${total_amount:.2f}")

        if st.button("✅ Confirm and Save", use_container_width=True):
            date_obj = datetime.combine(receipt_date, datetime.min.time())

            image_path = save_image_local(
                original_image,
                store,
                date_obj,
                round(float(total_amount), 2)
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
                    "Project / Investor": project,
                    "Payment Method": payment_method,
                    "Image Path": os.path.basename(image_path)
                })

            user_email = st.session_state["user_email"]
            root_folder_id = get_user_root_folder_id(user_email)
            
            excel_path = save_to_excel(
                rows,
                date_obj,
                root_folder_id
            )

            try:
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

                if excel_drive_id and image_drive_id:
                    st.success("✅ Receipt saved successfully")

            except Exception as e:
                st.error("Google Drive upload failed")
                st.exception(e)

            with open(excel_path, "rb") as f:
                st.download_button(
                    "Download Excel backup",
                    f,
                    file_name=os.path.basename(excel_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with open(image_path, "rb") as f:
                st.download_button(
                    "Download Receipt image backup",
                    f,
                    file_name=os.path.basename(image_path),
                    mime="image/jpeg",
                    use_container_width=True
                )

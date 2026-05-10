import os
import json
import base64
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


BASE_PATH = "/tmp/Receipt_Records"


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


def ocr_image(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)

    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41, 15
    )

    pil_img = Image.fromarray(processed)

    config = "--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1"

    text = pytesseract.image_to_string(pil_img, config=config)
    return text, pil_img


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
- Do not include subtotal, total, GST, payment, cash, change, ABN, phone number, footer, barcode, rewards text, advertising text.
- Preserve item descriptions as accurately as possible.
- Amount must be the line item amount, not the receipt total.
- If quantity and unit price are visible, fill them.
- If quantity or unit price are unclear, keep them empty.
- If an item amount is unclear, use 0.
- Return Amount as number, not string.
- Do not invent items.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
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
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        st.error("GPT Vision returned invalid JSON")
        st.text(raw)
        return []


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
        scopes=["https://www.googleapis.com/auth/drive.file"],
        redirect_uri=st.secrets["auth"]["redirect_uri"],
        autogenerate_code_verifier=False
    )
    return flow


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
            st.query_params.clear()

            st.success("✅ Google Drive connected")
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


def get_drive_service_oauth():
    if "google_credentials" not in st.session_state:
        return None

    credentials = st.session_state["google_credentials"]
    return build("drive", "v3", credentials=credentials)


def upload_to_drive(file_path, file_name, folder_id):
    service = get_drive_service_oauth()

    if service is None:
        st.error("Please login to Google Drive first")
        return None

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

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }

    folder = service.files().create(
        body=metadata,
        fields="id"
    ).execute()

    return folder.get("id")


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

    st.subheader("Original Image")
    st.image(original_image, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🤖 Read Receipt with GPT Vision", use_container_width=True):
            with st.spinner("Reading receipt image with GPT Vision..."):
                items = parse_receipt_image_with_gpt(original_image)

            if items:
                st.session_state["items_df"] = pd.DataFrame(items)
                st.success(f"GPT Vision found {len(items)} item rows")
            else:
                st.session_state["items_df"] = pd.DataFrame([{
                    "Item": "",
                    "Qty": "",
                    "Unit Price": "",
                    "Amount": 0.0
                }])
                st.warning("No items detected. You can manually enter items below.")

    with col2:
        if st.button("🧪 Debug OCR", use_container_width=True):
            with st.spinner("Running OCR debug..."):
                ocr_text, processed_image = ocr_image(original_image)

            st.session_state["ocr_text"] = ocr_text
            st.session_state["processed_image"] = processed_image

    if "ocr_text" in st.session_state:
        with st.expander("Show processed OCR image"):
            st.image(st.session_state["processed_image"], use_container_width=True)

        with st.expander("Show OCR text"):
            st.text_area("OCR Text", st.session_state["ocr_text"], height=220)

    if "items_df" in st.session_state and st.session_state["items_df"] is not None:
        st.subheader("Confirm / Edit Items")

        if st.session_state["items_df"].empty:
            st.session_state["items_df"] = pd.DataFrame([{
                "Item": "",
                "Qty": "",
                "Unit Price": "",
                "Amount": 0.0
            }])

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
            edited_df = pd.DataFrame([{
                "Item": "",
                "Qty": "",
                "Unit Price": "",
                "Amount": 0.0
            }])

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
        st.warning(f"Please check total amount before saving: ${total_amount:.2f}")

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

            excel_path = save_to_excel(rows, date_obj)

            root_folder_id = st.secrets["DRIVE_FOLDER_ID"]

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
                    st.success("✅ Saved to yearly Excel and compressed receipt image folder in Google Drive")

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
                    "Download Compressed Receipt backup",
                    f,
                    file_name=os.path.basename(image_path),
                    mime="image/jpeg",
                    use_container_width=True
                )

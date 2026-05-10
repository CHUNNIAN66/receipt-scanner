import os
import io
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
from openai import OpenAI

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


BASE_PATH = "/tmp/Receipt_Records"


def get_financial_year(date):
    if date.month >= 7:
        return f"FY{date.year}-{date.year + 1}"
    return f"FY{date.year - 1}-{date.year}"


def preprocess_versions(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    versions = []

    versions.append(gray)

    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    versions.append(resized)

    threshold = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)[1]
    versions.append(threshold)

    adaptive = cv2.adaptiveThreshold(
        resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )
    versions.append(adaptive)

    return versions


def ocr_image(image):
    versions = preprocess_versions(image)

    best_text = ""
    best_img = versions[0]

    for v in versions:
        pil_img = Image.fromarray(v)
        text = pytesseract.image_to_string(pil_img, config="--psm 6")

        if len(text.strip()) > len(best_text.strip()):
            best_text = text
            best_img = pil_img

    return best_text, best_img


def parse_items_with_gpt(text):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = f"""
You are parsing OCR text from an Australian retail receipt.

Return ONLY valid JSON array. No markdown. No explanation.

Each object must have:
[
  {{
    "Item": "item description",
    "Qty": "",
    "Unit Price": "",
    "Amount": 0.00
  }}
]

Rules:
- Only include purchased line items.
- Exclude store name, ABN, phone number, date, subtotal, total, GST, cash, change, barcode footer, flybuys, advertising text.
- Do not invent items that are not clearly supported by OCR text.
- If unsure about item name, keep the OCR-like wording.
- Amount must be close to a visible price in the OCR text.
- Ignore random isolated numbers unless they look like prices.
- Australian receipt prices usually appear like 3.50, 12.99, 120.00.
- If quantity and unit price are visible, calculate Amount = Qty * Unit Price.
- If line total is visible on the right, use that as Amount.
- If no clear amount is visible, use 0.
- Return Amount as number, not string.

OCR text:
{text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    raw = response.output_text.strip()

    try:
        return json.loads(raw)
    except Exception:
        st.error("GPT returned invalid JSON")
        st.text(raw)
        return []


def save_image(original_image, store, date, total="unknown"):
    fy = get_financial_year(date)
    folder = os.path.join(BASE_PATH, fy, store)
    os.makedirs(folder, exist_ok=True)

    safe_store = store.replace("/", "_").replace(" ", "_")
    filename = f"{date.strftime('%Y-%m-%d')}_{safe_store}_{total}.jpg"
    path = os.path.join(folder, filename)

    original_image.convert("RGB").save(path)
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


def download_drive_file(file_id, local_path):
    service = get_drive_service_oauth()

    request = service.files().get_media(fileId=file_id)

    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()


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

    original_image.thumbnail((1600, 1600))

    st.subheader("Original Image")
    st.image(original_image, use_container_width=True)

    if st.button("🔍 Run OCR", use_container_width=True):
        with st.spinner("Preprocessing image and running OCR..."):
            ocr_text, processed_image = ocr_image(original_image)

        st.session_state["ocr_text"] = ocr_text
        st.session_state["processed_image"] = processed_image
        st.session_state["items_df"] = None

    if "ocr_text" in st.session_state:
        with st.expander("Show processed OCR image"):
            st.image(st.session_state["processed_image"], use_container_width=True)

        with st.expander("Show OCR text"):
            st.text_area("OCR Text", st.session_state["ocr_text"], height=220)

        if st.button("🤖 Process Receipt with GPT", use_container_width=True):
            try:
                with st.spinner("Parsing receipt with GPT..."):
                    items = parse_items_with_gpt(st.session_state["ocr_text"])

                st.session_state["items_df"] = pd.DataFrame(items)
                st.success(f"GPT found {len(items)} item rows")

            except Exception as e:
                st.error("GPT parsing failed")
                st.exception(e)
                st.session_state["items_df"] = None

    if "items_df" in st.session_state and st.session_state["items_df"] is not None:
        st.subheader("Confirm / Edit Items")

        edited_rows = []

        for i, row in st.session_state["items_df"].iterrows():
            item_name = str(row.get("Item", ""))

            with st.expander(f"Item {i + 1}: {item_name}", expanded=True):
                item = st.text_input(
                    "Item",
                    value=str(row.get("Item", "")),
                    key=f"item_{i}"
                )

                qty = st.text_input(
                    "Qty",
                    value=str(row.get("Qty", "")),
                    key=f"qty_{i}"
                )

                unit_price = st.text_input(
                    "Unit Price",
                    value=str(row.get("Unit Price", "")),
                    key=f"unit_{i}"
                )

                try:
                    default_amount = float(row.get("Amount", 0) or 0)
                except Exception:
                    default_amount = 0.0

                amount = st.number_input(
                    "Amount",
                    value=default_amount,
                    step=0.01,
                    key=f"amount_{i}"
                )

                edited_rows.append({
                    "Item": item,
                    "Qty": qty,
                    "Unit Price": unit_price,
                    "Amount": amount
                })

        edited_df = pd.DataFrame(edited_rows)

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

            total_amount = edited_df["Amount"].sum()

            image_path = save_image(
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
            folder_id = st.secrets["DRIVE_FOLDER_ID"]

            try:
                excel_drive_id = upload_or_update_excel_to_drive(
                    excel_path,
                    os.path.basename(excel_path),
                    folder_id
                )

                image_drive_id = upload_to_drive(
                    image_path,
                    os.path.basename(image_path),
                    folder_id
                )

                if excel_drive_id and image_drive_id:
                    st.success("✅ Saved to one yearly Excel file in Google Drive")

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
                    "Download Receipt backup",
                    f,
                    file_name=os.path.basename(image_path),
                    mime="image/jpeg",
                    use_container_width=True
                )

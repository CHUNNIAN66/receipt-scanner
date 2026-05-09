import os
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
from googleapiclient.http import MediaFileUpload


BASE_PATH = "/tmp/Receipt_Records"


def get_financial_year(date):
    if date.month >= 7:
        return f"FY{date.year}-{date.year + 1}"
    return f"FY{date.year - 1}-{date.year}"


def preprocess_image(image):
    img = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 20
    coords = np.argwhere(mask)

    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img[y0:y1, x0:x1]

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    return Image.fromarray(gray)


def ocr_image(image):
    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed, config="--psm 6")
    return text, processed


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
- If quantity and unit price are visible, calculate Amount = Qty * Unit Price.
- If line total is visible on the right, use that as Amount.
- If OCR is messy, infer the most likely item rows.
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
    )
    return flow


def google_login_section():
    st.subheader("Google Drive Login")
    st.write("OAuth version: no PKCE")

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


st.title("📸 Receipt Scanner - GPT + Personal Google Drive")

google_login_section()

uploaded_file = st.file_uploader(
    "Upload receipt image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    original_image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(original_image, use_container_width=True)

    with st.spinner("Preprocessing image and running OCR..."):
        ocr_text, processed_image = ocr_image(original_image)

    st.subheader("Processed OCR Image")
    st.image(processed_image, use_container_width=True)

    st.subheader("OCR Text")
    st.text_area("OCR Text", ocr_text, height=250)

    if "items_df" not in st.session_state:
        st.session_state.items_df = None

    if st.button("Process Receipt"):
        try:
            with st.spinner("Parsing receipt with GPT, please wait 5-20 seconds..."):
                items = parse_items_with_gpt(ocr_text)

            st.session_state.items_df = pd.DataFrame(items)
            st.success(f"GPT found {len(items)} item rows")

        except Exception as e:
            st.error("GPT parsing failed")
            st.exception(e)
            st.session_state.items_df = None

    if st.session_state.items_df is not None:
        st.subheader("Confirm / Edit Items Before Saving")

        edited_df = st.data_editor(
            st.session_state.items_df,
            num_rows="dynamic",
            use_container_width=True
        )

        store = st.text_input("Store", value="Bunnings")
        category = st.text_input("Category", value="Materials")
        project = st.text_input("Project / Investor", value="")
        payment_method = st.text_input("Payment Method", value="")
        receipt_date = st.date_input("Receipt Date", value=datetime.today())

        if st.button("Confirm and Save"):
            date_obj = datetime.combine(receipt_date, datetime.min.time())

            edited_df["Amount"] = pd.to_numeric(
                edited_df["Amount"],
                errors="coerce"
            ).fillna(0)

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
                excel_drive_id = upload_to_drive(
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
                    st.success("✅ Saved and uploaded to your Google Drive")
                    st.write("Excel Drive file ID:", excel_drive_id)
                    st.write("Receipt image Drive file ID:", image_drive_id)

            except Exception as e:
                st.error("Google Drive upload failed")
                st.exception(e)

            with open(excel_path, "rb") as f:
                st.download_button(
                    "Download Excel backup",
                    f,
                    file_name=os.path.basename(excel_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with open(image_path, "rb") as f:
                st.download_button(
                    "Download Receipt backup",
                    f,
                    file_name=os.path.basename(image_path),
                    mime="image/jpeg"
                )

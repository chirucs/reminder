import streamlit as st
import requests
from PIL import Image
import io
import re
from datetime import datetime
import urllib.parse
import os
import socket
import cv2
import numpy as np
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import webbrowser
import easyocr
import fitz  # PyMuPDF

def preprocess_image(image_bytes):
    # A more robust, multi-step preprocessing pipeline for IDs
    image = Image.open(io.BytesIO(image_bytes)).convert('L') # Grayscale
    np_img = np.array(image)

    # 1. Denoising
    denoised = cv2.fastNlMeansDenoising(np_img, None, 10, 7, 21)

    # 2. Thresholding
    _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    processed_image = Image.fromarray(thresholded)
    return processed_image

def extract_text_from_image(image_bytes):
    # Use EasyOCR for text extraction
    reader = easyocr.Reader(['en'])
    image = Image.open(io.BytesIO(image_bytes))
    np_img = np.array(image)
    results = reader.readtext(np_img, detail=0)
    return '\n'.join(results)

def convert_pdf_to_images(pdf_bytes):
    # Convert PDF to images using PyMuPDF
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images

def extract_text_from_pdf(pdf_bytes):
    # Convert PDF to images using PyMuPDF
    images = convert_pdf_to_images(pdf_bytes)
    ocr_text = ''
    for i, img in enumerate(images):
        np_img = np.array(img)
        ocr_text += f'--- Page {i+1} ---\n'
        ocr_text += extract_text_from_image(np_img)
        ocr_text += '\n'
    return ocr_text

def normalize_expiry_date(date_str):
    # Convert all to mm/dd/yyyy if possible, else mm/yyyy
    parts = date_str.split('/')
    if len(parts) == 3:
        mm, dd, yyyy = parts
    elif len(parts) == 2:
        mm, yyyy = parts
        dd = None
    else:
        return None
    if len(yyyy) == 2:
        yyyy = str(datetime.now().year)[:2] + yyyy
    if dd:
        return f"{mm.zfill(2)}/{dd.zfill(2)}/{yyyy}"
    else:
        return f"{mm.zfill(2)}/{yyyy}"

def extract_expiry_dates(text):
    expiry_keywords = r'(exp(?:iry|iration)?\s*date|exp\s*date|exp\.|expires|valid\s*thru|valid\s*until|exp|good\s*thru|good\s*until|validity|exp)'  # expanded
    date_patterns = [
        r'(0[1-9]|1[0-2])[\/\-](0[1-9]|[12][0-9]|3[01])[\/\-](\d{4})',  # mm/dd/yyyy or mm-dd-yyyy
        r'(0[1-9]|[12][0-9]|3[01])[\/\-](0[1-9]|1[0-2])[\/\-](\d{4})',  # dd/mm/yyyy or dd-mm-yyyy
        r'(0[1-9]|1[0-2])[\/\-](\d{4})',                                  # mm/yyyy
    ]
    candidates = set()
    current_year = datetime.now().year
    lines = text.splitlines()

    # 1. Look for expiry keyword and date on the same line
    for line in lines:
        if re.search(expiry_keywords, line, re.IGNORECASE):
            for pattern in date_patterns:
                found = re.findall(pattern, line)
                for match in found:
                    if len(match) == 3:
                        mm, dd, yyyy = match
                        if int(yyyy) >= current_year and int(yyyy) <= current_year + 50:  # Validate year range
                            candidates.add(f"{mm}/{dd}/{yyyy}")
                    elif len(match) == 2:
                        mm, yyyy = match
                        if int(yyyy) >= current_year and int(yyyy) <= current_year + 50:  # Validate year range
                            candidates.add(f"{mm}/{yyyy}")

    # 2. If not found, look for a line where the keyword is immediately followed by a date
    if not candidates:
        for i, line in enumerate(lines):
            if re.search(expiry_keywords, line, re.IGNORECASE):
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    for pattern in date_patterns:
                        found = re.findall(pattern, next_line)
                        for match in found:
                            if len(match) == 3:
                                mm, dd, yyyy = match
                                if int(yyyy) >= current_year and int(yyyy) <= current_year + 50:
                                    candidates.add(f"{mm}/{dd}/{yyyy}")
                            elif len(match) == 2:
                                mm, yyyy = match
                                if int(yyyy) >= current_year and int(yyyy) <= current_year + 50:
                                    candidates.add(f"{mm}/{yyyy}")

    # 3. As a last resort, pick the most likely date pattern in the whole text
    if not candidates:
        for pattern in date_patterns:
            found = re.findall(pattern, text)
            for match in found:
                if len(match) == 3:
                    mm, dd, yyyy = match
                    if int(yyyy) >= current_year and int(yyyy) <= current_year + 50:
                        candidates.add(f"{mm}/{dd}/{yyyy}")
                elif len(match) == 2:
                    mm, yyyy = match
                    if int(yyyy) >= current_year and int(yyyy) <= current_year + 50:
                        candidates.add(f"{mm}/{yyyy}")

    # Normalize and pick the latest expiry date, prefer mm/dd/yyyy if available
    normalized = set()
    for c in candidates:
        norm = normalize_expiry_date(c)
        if norm:
            normalized.add(norm)

    if not normalized:
        return []

    def date_key(d):
        parts = d.split('/')
        if len(parts) == 3:
            mm, dd, yyyy = parts
        else:
            mm, yyyy = parts
            dd = '01'  # fallback for sorting
        return int(yyyy) * 10000 + int(mm) * 100 + int(dd)

    sorted_dates = sorted(normalized, key=date_key, reverse=True)
    for d in sorted_dates:
        if len(d.split('/')) == 3:
            return [d]
    return [sorted_dates[0]]

def get_host_url():
    # Get the host URL dynamically
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return f"http://{local_ip}:8000/uploads/"

def save_uploaded_file(uploaded_file):
    # Save the uploaded file locally
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def create_google_calendar_link_with_local_path(expiry_date_str, file_path, reference_location):
    # expiry_date_str: mm/dd/yyyy or mm/yyyy
    from datetime import datetime, timedelta
    try:
        parts = expiry_date_str.split('/')
        if len(parts) == 3:
            # Handles mm/dd/yyyy and dd/mm/yyyy by trying both
            try:
                dt = datetime.strptime(expiry_date_str, "%m/%d/%Y")
            except ValueError:
                dt = datetime.strptime(expiry_date_str, "%d/%m/%Y")
        elif len(parts) == 2:
            # If only mm/yyyy, default to first of the month
            dt = datetime.strptime(f"01/{expiry_date_str}", "%d/%m/%Y")
        else:
            return None
    except (ValueError, IndexError):
        return None # Return None if parsing fails

    reminder_date = dt - timedelta(days=30)
    start = reminder_date.strftime("%Y%m%d")
    end = (reminder_date + timedelta(hours=1)).strftime("%Y%m%d")
    title = urllib.parse.quote("Important Document Reminder")

    # Create a clickable file URL
    file_url = f"file://{os.path.abspath(file_path)}"
    details = urllib.parse.quote(f"Your document expires on {expiry_date_str}.\nDocument Location: {reference_location}\nFile URL: {file_url}")
    link = f"https://calendar.google.com/calendar/render?action=TEMPLATE&text={title}&dates={start}/{end}&details={details}"
    return link

def extract_title_from_text(text):
    # Extract the most likely title or heading from the text
    lines = text.splitlines()
    for line in lines:
        # Look for the first non-empty line in uppercase or with keywords
        if line.strip() and (line.isupper() or re.search(r'(ID|CARD|LICENSE|DOCUMENT|CERTIFICATE)', line, re.IGNORECASE)):
            return line.strip()
    # Fallback to the first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()
    return "Document Reminder"  # Default title if no suitable line is found

def recommend_expiry_date():
    # Recommend an expiry date 1 year from today
    today = datetime.now()
    recommended_date = today + timedelta(days=365)
    return recommended_date.strftime("%m/%d/%Y")

def send_email_with_attachment(to_email, subject, body, file_path):
    # Email configuration
    from_email = "your_email@example.com"  # Replace with your email
    from_password = "your_password"  # Replace with your email password

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    # Attach the file
    attachment = open(file_path, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(file_path)}")
    msg.attach(part)
    attachment.close()

    # Send the email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, from_password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

def open_cloud_app_or_url(source_name):
    # Open the local cloud app if installed, else fallback to the web URL
    cloud_sources = {
        "Google Drive": "https://drive.google.com",
        "Dropbox": "https://www.dropbox.com",
        "OneDrive": "https://onedrive.live.com",
        "iCloud Drive": "https://www.icloud.com/",
        "Mega": "https://mega.nz"
    }
    try:
        # Attempt to open the offline app (updated implementation)
        if source_name == "Google Drive":
            if os.name == 'posix':  # macOS/Linux
                os.system('open -a "Google Drive"')  # macOS example
            else:
                webbrowser.open("google-drive://")
        elif source_name == "Dropbox":
            webbrowser.open("dropbox://")
        elif source_name == "OneDrive":
            webbrowser.open("onedrive://")
        elif source_name == "iCloud Drive":
            webbrowser.open("icloud://")
        elif source_name == "Mega":
            webbrowser.open("mega://")
        else:
            raise Exception("Offline app not found")
    except:
        # Fallback to web link
        webbrowser.open(cloud_sources[source_name])

def open_camera():
    # Open the device's camera (dummy implementation)
    try:
        if os.name == 'posix':  # macOS/Linux
            os.system('open -a "Photo Booth"')  # macOS example
        elif os.name == 'nt':  # Windows
            os.system('start microsoft.windows.camera:')
        else:
            raise Exception("Camera app not supported on this OS")
    except Exception as e:
        st.error(f"Could not open the camera: {e}")

def main():
    st.set_page_config(page_title="ExpiryGuard: Smart Document Reminder")
    st.title("ExpiryGuard: Smart Document Reminder")
    st.write("Upload an image or PDF to extract expiry dates and set reminders easily.")

    uploaded_file = st.file_uploader("Choose an image or PDF file", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)

    if uploaded_file:
        st.success("File uploaded successfully!")
        file_bytes = uploaded_file.read()

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(file_bytes)
        else:
            text = extract_text_from_image(file_bytes)

        expiry_dates = extract_expiry_dates(text)
        st.subheader("Extracted Expiry Dates:")

        if expiry_dates:
            for date in expiry_dates:
                st.write(f"- {date}")
                cal_link = create_google_calendar_link_with_local_path(date, "", "")
                if cal_link:
                    st.markdown(f"[Add 30-day reminder to Google Calendar]({cal_link})", unsafe_allow_html=True)
                else:
                    st.warning(f"Could not generate a calendar link for the date: {date}")
        else:
            st.write("No expiry dates found in the document.")
            st.info("You can set a custom expiry date below.")
            selected_date = st.date_input("Select expiry date:", min_value=datetime.now().date())
            reference_location = st.text_input("Enter a reference location or note for the document:")
            recommended_date_str = selected_date.strftime("%m/%d/%Y")
            st.write(f"Selected expiry date: {recommended_date_str}")

            cal_link = create_google_calendar_link_with_local_path(recommended_date_str, "", reference_location)
            if cal_link:
                st.markdown(f"[Add 30-day reminder to Google Calendar]({cal_link})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()







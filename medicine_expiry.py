
import cv2
import easyocr
import re
import calendar
from datetime import datetime

# 1. Initialize OCR
print("Loading EasyOCR model. Please wait...")
reader = easyocr.Reader(["en"], gpu=False)
print("Medicine Expiry Detection System Started")


# 2. Extract dates from OCR text
def extract_dates(text):

    # Join separate OCR lines so label and date can be matched
    text = re.sub(r"\s+", " ", text).upper()

    # Support MFG 03/2025, MFD: 03/2025, etc.
    mfg_pattern = re.compile(
        r"\bM\s*F\s*[GD]\s*\.?\s*(?:DATE\s*)?[:.]?\s*"
        r"((?:0?[1-9]|1[0-2])\s*[/.-]\s*\d{4})",
        re.IGNORECASE
    )

    # Support EXP 08/2027, EXP DATE: 08/2027, EXPIRY 08/2027
    exp_pattern = re.compile(
        r"\bE\s*X\s*P(?:IRY)?\s*\.?\s*"
        r"(?:DATE\s*)?[:.]?\s*"
        r"((?:0?[1-9]|1[0-2])\s*[/.-]\s*\d{4})",
        re.IGNORECASE
    )

    mfg_match = mfg_pattern.search(text)
    exp_match = exp_pattern.search(text)

    mfg_date = None
    expiry_date = None

    if mfg_match:
        mfg_date = re.sub(r"\s+", "", mfg_match.group(1))

    if exp_match:
        try:
            value = re.sub(r"\s+", "", exp_match.group(1))
            month, year = map(int, re.split(r"[/.-]", value))

            if 1 <= month <= 12 and 1900 <= year <= 2200:
                last_day = calendar.monthrange(year, month)[1]
                expiry_date = datetime(
                    year, month, last_day
                ).date()

        except ValueError:
            expiry_date = None

    return mfg_date, expiry_date


# 3. Process image using multiple OCR methods
def read_label(crop):

    enlarged = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    best_mfg = None
    best_expiry = None
    best_text = ""

    # Try all versions instead of stopping after one date
    for image in [gray, enhanced, thresholded]:

        results = reader.readtext(
            image,
            detail=1,
            paragraph=False
        )

        # Arrange text in reading order
        results.sort(
            key=lambda item: (
                sum(point[1] for point in item[0]) / 4,
                min(point[0] for point in item[0])
            )
        )

        text = " ".join(
            item[1].strip() for item in results
        )

        print("OCR attempt:", text)

        mfg_date, expiry_date = extract_dates(text)

        # Keep any date found in any OCR attempt
        if mfg_date:
            best_mfg = mfg_date

        if expiry_date:
            best_expiry = expiry_date

        if len(text) > len(best_text):
            best_text = text

        # Stop only when both dates have been found
        if best_mfg and best_expiry:
            break

    return best_mfg, best_expiry, best_text


# 4. Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nInstructions:")
print("1. Keep MFG and EXP dates inside the green rectangle.")
print("2. Keep the text clear, upright and well lit.")
print("3. Press SPACE to capture another image.")
print("4. Press Q to quit.")


# 5. Display webcam with guide rectangle
while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read webcam.")
        break

    height, width = frame.shape[:2]

    # Large rectangle for MFG and EXP dates
    box_width = int(width * 0.85)
    box_height = int(height * 0.45)

    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    x2 = x1 + box_width
    y2 = y1 + box_height

    display = frame.copy()

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        display,
        "Place MFG and EXP dates inside box",
        (x1, max(30, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "SPACE: Capture | Q: Quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("Medicine Expiry Detection", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # 6. Capture and process each time SPACE is pressed
    if key == ord(" "):

        crop = frame[y1:y2, x1:x2].copy()

        print("\nProcessing new medicine capture...")

        mfg_date, expiry_date, detected_text = read_label(crop)

        # 7. Print medicine details
        print("\n========== MEDICINE DETAILS ==========")

        print(
            "Manufacturing Date:",
            mfg_date if mfg_date else "Not detected"
        )

        if expiry_date is not None:

            print("Expiry Date:", expiry_date.strftime("%m/%Y"))

            if datetime.now().date() > expiry_date:
                status = "PRODUCT IS EXPIRED"
                color = (0, 0, 255)
            else:
                status = "PRODUCT IS NOT EXPIRED"
                color = (0, 200, 0)

            print("Status:", status)

        else:
            status = "PLEASE VERIFY EXPIRY DATE"
            color = (0, 165, 255)

            print("Expiry Date: Not detected")
            print("Status:", status)

        print("======================================")

        # 8. Display captured label and result
        result = cv2.copyMakeBorder(
            crop,
            110, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(25, 25, 25)
        )

        cv2.putText(
            result,
            "MFG: " + (mfg_date or "Not detected"),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        exp_text = (
            expiry_date.strftime("%m/%Y")
            if expiry_date else "Not detected"
        )

        cv2.putText(
            result,
            "EXP: " + exp_text,
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            result,
            status,
            (10, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        cv2.imshow("Medicine Expiry Result", result)

        # Save every capture to the same file
        cv2.imwrite("captured_medicine_label.jpg", crop)


# 9. Release webcam
camera.release()
cv2.destroyAllWindows()

print("Medicine Expiry Detection System Closed.")

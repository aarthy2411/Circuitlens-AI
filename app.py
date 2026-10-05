import streamlit as st
import pytesseract
from pytesseract import Output
from PIL import Image
import cv2
import numpy as np
import re


# ============================================================
# TESSERACT
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="CircuitLens AI",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 CircuitLens AI")
st.subheader("AI-Powered Circuit Image Analyzer")

st.write(
    "Upload a circuit diagram to detect circuit labels, "
    "components, values and connections."
)


# ============================================================
# OCR PREPROCESSING
# ============================================================

def prepare_ocr_images(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Upscale
    enlarged = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        enlarged
    )

    # Threshold
    otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    return [
        enhanced,
        otsu,
        adaptive
    ]


# ============================================================
# OCR
# ============================================================

def perform_ocr(images):

    detected_words = []

    for img in images:

        for psm in [6, 11, 12]:

            try:

                data = pytesseract.image_to_data(
                    img,
                    config=f"--oem 3 --psm {psm}",
                    output_type=Output.DICT
                )

                for i in range(
                    len(data["text"])
                ):

                    word = data["text"][i].strip()

                    if not word:
                        continue

                    try:
                        confidence = float(
                            data["conf"][i]
                        )
                    except:
                        confidence = 0

                    if confidence < 25:
                        continue

                    detected_words.append({
                        "text": word,
                        "confidence": confidence
                    })

            except:
                pass

    return detected_words


# ============================================================
# OCR NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.upper()

    text = text.replace(
        "µ",
        "U"
    )

    text = text.replace(
        "μ",
        "U"
    )

    text = re.sub(
        r"\s+",
        "",
        text
    )

    return text


# ============================================================
# COMPONENT LABEL DETECTION
# ============================================================

def detect_labels(words):

    components = {
        "LED": set(),
        "RESISTOR": set(),
        "CAPACITOR": set(),
        "DIODE": set(),
        "TRANSISTOR": set(),
        "GROUND": set(),
        "VOLTAGE_SOURCE": set()
    }

    for item in words:

        raw = item["text"]

        confidence = item["confidence"]

        word = normalize_text(
            raw
        )

        # ----------------------------------------------------
        # LED
        # ----------------------------------------------------

        if re.fullmatch(
            r"LED[0-9]*",
            word
        ):

            if confidence >= 30:
                components["LED"].add(
                    word
                )

        # ----------------------------------------------------
        # RESISTOR
        # ----------------------------------------------------

        if re.fullmatch(
            r"R[0-9]+",
            word
        ):

            if confidence >= 30:
                components["RESISTOR"].add(
                    word
                )

        # ----------------------------------------------------
        # CAPACITOR
        # ----------------------------------------------------

        if re.fullmatch(
            r"C[0-9]+",
            word
        ):

            if confidence >= 30:
                components["CAPACITOR"].add(
                    word
                )

        # ----------------------------------------------------
        # DIODE
        # ----------------------------------------------------

        if re.fullmatch(
            r"D[0-9]+",
            word
        ):

            if confidence >= 30:
                components["DIODE"].add(
                    word
                )

        # ----------------------------------------------------
        # TRANSISTOR
        # ----------------------------------------------------

        if re.fullmatch(
            r"Q[0-9]+",
            word
        ):

            if confidence >= 30:
                components["TRANSISTOR"].add(
                    word
                )

        # ----------------------------------------------------
        # GND
        # ----------------------------------------------------

        if word in [
            "GND",
            "GROUND",
            "GNO"
        ]:

            components["GROUND"].add(
                "GND"
            )

        # ----------------------------------------------------
        # VOLTAGE
        # ----------------------------------------------------

        if re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)?V",
            word
        ):

            if confidence >= 30:

                components[
                    "VOLTAGE_SOURCE"
                ].add(
                    word
                )

    return components


# ============================================================
# ELECTRICAL VALUE DETECTION
# ============================================================

def detect_values(words):

    voltage = set()
    resistance = set()
    capacitance = set()

    for item in words:

        word = normalize_text(
            item["text"]
        )

        confidence = item["confidence"]

        if confidence < 30:
            continue

        # ----------------------------------------------------
        # VOLTAGE
        # ----------------------------------------------------

        voltage_matches = re.findall(
            r"[0-9]+(?:\.[0-9]+)?V",
            word
        )

        for value in voltage_matches:

            voltage.add(
                value
            )

        # ----------------------------------------------------
        # RESISTANCE
        # ----------------------------------------------------

        resistance_matches = re.findall(
            r"[0-9]+(?:\.[0-9]+)?"
            r"(?:OHM|OHMS|KOHM|KOHMS|MEGOHM|MEGOHMS|K|M)",
            word
        )

        for value in resistance_matches:

            resistance.add(
                value
            )

        # ----------------------------------------------------
        # CAPACITANCE
        # ----------------------------------------------------

        capacitance_matches = re.findall(
            r"[0-9]+(?:\.[0-9]+)?"
            r"(?:UF|NF|PF)",
            word
        )

        for value in capacitance_matches:

            capacitance.add(
                value
            )

    return (
        sorted(voltage),
        sorted(resistance),
        sorted(capacitance)
    )


# ============================================================
# OPENCV - DETECT CIRCUIT LINES
# ============================================================

def detect_wires(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Slightly blur
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=10
    )

    horizontal = 0
    vertical = 0

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            dx = abs(
                x2 - x1
            )

            dy = abs(
                y2 - y1
            )

            # Horizontal
            if dx > 40 and dy < 8:

                horizontal += 1

            # Vertical
            elif dy > 40 and dx < 8:

                vertical += 1

    return horizontal, vertical


# ============================================================
# OPENCV - SHAPE ANALYSIS
# ============================================================

def analyze_shapes(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Threshold
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Remove tiny noise
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    large_objects = 0
    rectangular_objects = 0
    elongated_objects = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 30:
            continue

        if area > 100:

            large_objects += 1

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w > 5 and h > 5:

            ratio = max(w, h) / min(w, h)

            if ratio > 3:

                elongated_objects += 1

            elif ratio < 3:

                rectangular_objects += 1

    return {
        "large_objects": large_objects,
        "rectangular_objects": rectangular_objects,
        "elongated_objects": elongated_objects
    }


# ============================================================
# SAFE RESISTANCE CONVERSION
# ============================================================

def resistance_to_ohms(value):

    value = value.upper()

    number_match = re.search(
        r"[0-9]+(?:\.[0-9]+)?",
        value
    )

    if not number_match:
        return None

    number = float(
        number_match.group()
    )

    if "MEGOHM" in value:

        number *= 1000000

    elif "KOHM" in value:

        number *= 1000

    elif re.search(
        r"[0-9]+K$",
        value
    ):

        number *= 1000

    elif re.search(
        r"[0-9]+M$",
        value
    ):

        number *= 1000000

    return number


# ============================================================
# MAIN
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Circuit Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ========================================================
    # IMAGE
    # ========================================================

    st.divider()

    st.header(
        "📷 Uploaded Circuit"
    )

    st.image(
        image,
        use_container_width=True
    )


    # ========================================================
    # OCR
    # ========================================================

    with st.spinner(
        "🤖 Reading circuit labels..."
    ):

        ocr_images = prepare_ocr_images(
            image
        )

        ocr_words = perform_ocr(
            ocr_images
        )


    # ========================================================
    # OCR TEXT
    # ========================================================

    unique_text = []

    for item in ocr_words:

        word = item["text"].strip()

        if word and word not in unique_text:

            unique_text.append(
                word
            )

    final_ocr_text = "\n".join(
        unique_text
    )


    st.divider()

    st.header(
        "📄 OCR Result"
    )

    if final_ocr_text:

        st.success(
            "Text detected!"
        )

        st.text(
            final_ocr_text
        )

    else:

        st.warning(
            "No readable text detected."
        )


    # ========================================================
    # COMPONENT LABELS
    # ========================================================

    components = detect_labels(
        ocr_words
    )


    # ========================================================
    # VALUES
    # ========================================================

    (
        voltage_values,
        resistance_values,
        capacitance_values
    ) = detect_values(
        ocr_words
    )


    # ========================================================
    # COMPUTER VISION
    # ========================================================

    with st.spinner(
        "👁️ Analysing circuit structure..."
    ):

        horizontal_wires, vertical_wires = (
            detect_wires(image)
        )

        shape_info = analyze_shapes(
            image
        )


    # ========================================================
    # COMPONENTS
    # ========================================================

    st.divider()

    st.header(
        "🔧 Detected Components"
    )

    found = False


    if components["LED"]:

        found = True

        for item in sorted(
            components["LED"]
        ):

            st.write(
                "💡 LED →",
                item
            )


    if components["RESISTOR"]:

        found = True

        for item in sorted(
            components["RESISTOR"]
        ):

            st.write(
                "🔩 Resistor →",
                item
            )


    if components["CAPACITOR"]:

        found = True

        for item in sorted(
            components["CAPACITOR"]
        ):

            st.write(
                "🔋 Capacitor →",
                item
            )


    if components["DIODE"]:

        found = True

        for item in sorted(
            components["DIODE"]
        ):

            st.write(
                "➡️ Diode →",
                item
            )


    if components["TRANSISTOR"]:

        found = True

        for item in sorted(
            components["TRANSISTOR"]
        ):

            st.write(
                "🔺 Transistor →",
                item
            )


    if components["GROUND"]:

        found = True

        st.write(
            "🌍 Ground → GND"
        )


    if components["VOLTAGE_SOURCE"]:

        found = True

        for item in sorted(
            components["VOLTAGE_SOURCE"]
        ):

            st.write(
                "⚡ Voltage Source →",
                item
            )


    if not found:

        st.warning(
            "No reliable component labels detected."
        )


    # ========================================================
    # VALUES
    # ========================================================

    st.header(
        "📊 Detected Values"
    )


    if voltage_values:

        st.write(
            "⚡ Voltage:",
            ", ".join(
                voltage_values
            )
        )


    if resistance_values:

        st.write(
            "🔩 Resistance:",
            ", ".join(
                resistance_values
            )
        )


    if capacitance_values:

        st.write(
            "🔋 Capacitance:",
            ", ".join(
                capacitance_values
            )
        )


    if not (
        voltage_values
        or resistance_values
        or capacitance_values
    ):

        st.info(
            "No reliable electrical values detected."
        )


    # ========================================================
    # CIRCUIT STRUCTURE
    # ========================================================

    st.divider()

    st.header(
        "🔌 Circuit Structure"
    )


    if horizontal_wires > 0:

        st.write(
            f"➖ Horizontal wire segments: "
            f"{horizontal_wires}"
        )

    if vertical_wires > 0:

        st.write(
            f"│ Vertical wire segments: "
            f"{vertical_wires}"
        )


    if (
        horizontal_wires > 0
        or vertical_wires > 0
    ):

        st.success(
            "Circuit wiring structure detected."
        )

    else:

        st.info(
            "No strong wire structure detected."
        )


    # ========================================================
    # SHAPE INFORMATION
    # ========================================================

    st.header(
        "👁️ Visual Structure Analysis"
    )

    st.write(
        "Large circuit regions detected:",
        shape_info["large_objects"]
    )

    st.write(
        "Rectangular regions:",
        shape_info["rectangular_objects"]
    )

    st.write(
        "Elongated regions:",
        shape_info["elongated_objects"]
    )


    # ========================================================
    # CIRCUIT ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "🧠 Circuit Analysis"
    )


    has_led = bool(
        components["LED"]
    )

    has_resistor = bool(
        components["RESISTOR"]
    )

    has_ground = bool(
        components["GROUND"]
    )

    has_voltage = bool(
        voltage_values
    )


    if has_led and has_resistor:

        st.success(
            "💡 LED + resistor detected. "
            "The resistor can limit the current flowing "
            "through the LED."
        )

    elif has_led:

        st.warning(
            "💡 LED detected, but no reliable resistor "
            "label was found."
        )

    elif has_resistor:

        st.info(
            "🔩 Resistor detected."
        )


    if has_voltage:

        st.info(
            "⚡ Supply voltage detected: "
            + ", ".join(
                voltage_values
            )
        )


    if has_ground:

        st.info(
            "🌍 Ground connection detected."
        )


    # ========================================================
    # OHM'S LAW
    # ========================================================

    st.header(
        "⚡ Electrical Calculation"
    )


    if (
        voltage_values
        and resistance_values
    ):

        try:

            voltage_number = float(
                re.search(
                    r"[0-9]+(?:\.[0-9]+)?",
                    voltage_values[0]
                ).group()
            )

            resistance_number = (
                resistance_to_ohms(
                    resistance_values[0]
                )
            )

            if resistance_number:

                current = (
                    voltage_number /
                    resistance_number
                )

                st.success(
                    f"Voltage = {voltage_number} V\n\n"
                    f"Resistance = {resistance_number} Ω\n\n"
                    f"Current = {current:.6f} A"
                )

            else:

                st.info(
                    "Resistance value could not be interpreted."
                )

        except:

            st.info(
                "Could not calculate current."
            )

    else:

        st.info(
            "Voltage and resistance values are needed "
            "for current calculation."
        )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.divider()

    st.header(
        "💡 Circuit Recommendation"
    )


    if has_led and has_resistor:

        st.success(
            "✅ The circuit contains an LED and a resistor. "
            "The resistor provides current limiting."
        )

    elif has_led:

        st.warning(
            "⚠️ LED detected without a reliably detected "
            "resistor label. Check the circuit manually."
        )

    elif has_resistor:

        st.info(
            "ℹ️ Resistor detected. Verify its resistance value."
        )

    else:

        st.info(
            "Upload a clearer circuit diagram "
            "for stronger analysis."
        )


    # ========================================================
    # FINAL
    # ========================================================

    st.divider()

    st.success(
        "🎯 CircuitLens AI analysis completed!"
    )

    st.caption(
        "CircuitLens AI combines OCR and OpenCV-based "
        "visual analysis. OCR identifies readable labels "
        "and values, while OpenCV analyses circuit structure "
        "such as lines and shapes."
    )

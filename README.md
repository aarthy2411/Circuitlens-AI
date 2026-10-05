# Circuitlens-AI

**CircuitLens AI** is an AI-based circuit image analysis application that helps analyze electronic circuit diagrams using **OCR and Computer Vision**.

The application takes a circuit image as input and identifies readable circuit labels, electrical values, components, wires, and basic circuit structure. It also performs simple electrical calculations such as current calculation using Ohm's Law when suitable voltage and resistance values are detected.Try the live application here:https://circuitlens-ai-hpiayu84bnza53i5tv2jfy.streamlit.app/

##  Objectives

* Analyze circuit diagrams from uploaded images.
* Extract readable text from circuit diagrams using OCR.
* Detect common electronic components and their labels.
* Identify electrical values such as voltage, resistance, and capacitance.
* Detect horizontal and vertical wire structures.
* Perform basic circuit analysis.
* Calculate current using Ohm's Law when voltage and resistance are available.
* Provide simple recommendations based on the detected circuit components.

##  Features

###  Circuit Image Upload

Users can upload circuit diagrams in PNG, JPG, or JPEG format.

###  OCR Text Detection

The application uses **Tesseract OCR** to extract readable text from circuit images.

It can detect information such as:

* Component labels
* Voltage values
* Resistance values
* Capacitance values
* Ground labels

### Component Detection

CircuitLens AI can identify common component labels such as:

* LED
* Resistor
* Capacitor
* Diode
* Transistor
* Ground
* Voltage Source

###  Wire Detection

The application uses OpenCV image processing techniques to identify:

* Horizontal wire segments
* Vertical wire segments

This helps in understanding the basic structure of the circuit.

###  Electrical Value Detection

The application can detect values such as:

* Voltage → `5V`, `12V`
* Resistance → `220Ω`, `1K`
* Capacitance → `10uF`, `100nF`

###  Ohm's Law Calculation

When both voltage and resistance values are detected, CircuitLens AI calculates current using:

**I = V / R**

Where:

* `I` = Current
* `V` = Voltage
* `R` = Resistance

### Circuit Analysis

Based on the detected components and values, the application provides a simple analysis of the uploaded circuit.

For example, if an LED and resistor are detected, the application can identify that the resistor can be used for current limiting.

##  Technologies Used

| Technology    | Purpose                              |
| ------------- | ------------------------------------ |
| Python        | Main programming language            |
| Streamlit     | Web application interface            |
| OpenCV        | Image processing and computer vision |
| Tesseract OCR | Text recognition                     |
| Pytesseract   | Python interface for Tesseract OCR   |
| NumPy         | Numerical and image array processing |
| Pillow        | Image handling                       |

##  How It Works

The application follows these main steps:

```text
Upload Circuit Image
        ↓
Image Preprocessing
        ↓
OCR Text Detection
        ↓
Component & Value Detection
        ↓
Wire Detection
        ↓
Circuit Structure Analysis
        ↓
Electrical Calculation
        ↓
Circuit Recommendation
```
### `app.py`

Contains the complete Streamlit application, including:

* Image upload
* OCR processing
* Component detection
* Electrical value detection
* Wire detection
* Circuit analysis
* Ohm's Law calculation

### `requirements.txt`

Contains the Python libraries required to run the application.

### `packages.txt`

Contains the system package required for Tesseract OCR.

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aarthy2411/Circuitlens-AI.git
```

### 2. Open the Project Folder

```bash
cd Circuitlens-AI
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in the browser.

##  Deployment

CircuitLens AI can also be deployed using **Streamlit Cloud**.

The project uses:

```text
opencv-python-headless
```

instead of the normal OpenCV package so that it can run properly in a cloud environment.

Tesseract OCR is installed through:

```text
packages.txt
```

##  Output

After uploading a circuit image, the application provides:

* Uploaded circuit image
* OCR result
* Detected components
* Detected electrical values
* Wire structure
* Visual structure analysis
* Circuit analysis
* Current calculation
* Circuit recommendations

##  Example Analysis

For a circuit containing an LED, resistor, and voltage source, the application may detect:

```text
LED → LED1
Resistor → R1
Voltage → 5V
Resistance → 220Ω
```

It can then calculate the approximate current using:

```text
I = V / R
```

##  Future Enhancements

The project can be improved further by adding:

* Automatic circuit diagram reconstruction
* More accurate component recognition
* Series and parallel circuit detection
* Automatic circuit simulation
* Short-circuit detection
* More advanced electrical calculations
* AI-based circuit fault detection
* Support for more electronic components
* Interactive circuit visualization
* Improved OCR accuracy for handwritten circuit diagrams

##  Project Purpose

CircuitLens AI was developed as a learning project to understand the practical use of:

* Python
* Computer Vision
* OCR
* Image Processing
* Streamlit
* Basic Electrical Circuit Analysis

The project combines these technologies into a single application that can analyze circuit diagrams from images.

##  Author

**Aarthy V**

B.Sc. Computer Science with AI



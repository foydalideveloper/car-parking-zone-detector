
This project is a complete, end-to-end computer vision application that detects and counts occupied and empty spaces in a parking lot. It uses a custom-trained YOLOv8 model deployed in a sophisticated and interactive Streamlit web application.

A user can upload an image of a parking lot, and the application will return a full analysis dashboard, including an annotated image, occupancy metrics, a visual chart, and a downloadable PDF report.

 
https://imgur.com/a/zhWCUds

---

##  Features

-   **Advanced Object Detection:** Utilizes a custom-trained YOLOv8n model to identify parking spaces with high accuracy.
-   **Intelligent Classification:** Classifies each space as either **"occupied"** or **"empty"**.
-   **Interactive Dashboard:** A professional user interface built with Streamlit, featuring a sidebar for controls and a main panel for results.
-   **Dynamic Controls:** Allows users to adjust the model's **confidence threshold** in real-time via a slider.
-   **Data Visualization:** Displays a clean, modern **pie chart** visualizing the ratio of occupied vs. empty spots.
-   **PDF Reporting:** Generates and allows users to download a full PDF report of the analysis, including the image, counts, and chart.
-   **Persistent Logging:** Automatically saves the results of every detection to a local `detection_log.csv` file for data persistence.
-   **Upload History:** The sidebar keeps track of the last 5 analyses for easy reference (within the current session).

##  Tech Stack & Tools

-   **Model:** YOLOv8n (from Ultralytics)
-   **Training Environment:** Google Colab (with T4 GPU)
-   **Data Preparation:** Roboflow (for annotation conversion and augmentation)
-   **Application Framework:** Streamlit
-   **Core Libraries:** PyTorch, OpenCV, NumPy, Pandas, Matplotlib
-   **Reporting Library:** FPDF2 (`fpdf2`)
-   **Development Environment:** Visual Studio Code
-   **Environment Management:** Conda

##  Project Structure

```
streamlit_parking_app/
├── app.py              # The main Streamlit application script
├── best.pt             # The custom-trained YOLOv8 model weights
├── requirements.txt    # List of all Python dependencies for the project
├── detection_log.csv   # (Auto-generated) Log file for detection results
├── .gitignore          # Specifies which files/folders for Git to ignore
└── README.md           # This documentation file
```

## How to Run This Project Locally

Follow these steps to set up and run the application on your own machine (tested on macOS).

### 1. Prerequisites

-   You have **Conda** installed on your system.
-   You have **Git** installed. If not, on macOS you may need to run `xcode-select --install`.

### 2. Setup

**a. Get the Project Files**

Clone this repository to your local machine:
```bash
git clone https://github.com/foydalideveloper/car-parking-zone-detector.git
cd car-parking-zone-detector
```

**b. Create and Activate the Conda Environment**

I used a Conda environment to keep all project dependencies isolated. This is the recommended way to avoid library conflicts.

```bash
# Create a new environment named 'parking_app_env' with Python 3.10
conda create --name parking_app_env python=3.10 -y

# Activate the new environment
conda activate parking_app_env
```
Your terminal prompt should now start with `(parking_app_env)`.

**c. Install Dependencies**

All required libraries are listed in the `requirements.txt` file. Install them with a single command:
```bash
pip install -r requirements.txt
```

### 3. Run the Application

With the environment activated and dependencies installed, you can launch the Streamlit app. I recommend using the explicit command to ensure it uses the Python from our environment.

```bash
# Run the app
python -m streamlit run app.py
```

A new tab should automatically open in your web browser at `http://localhost:8501`. You can now upload a parking lot image to see the model in action!


# Import necessary libraries for the application
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# --- PAGE CONFIGURATION ---
# Setting up the basic configuration for the Streamlit page.
# This includes the title, icon, layout, and initial state of the sidebar.
st.set_page_config(
    page_title="Pro Parking Lot Counter", 
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
# using Streamlit's session_state to store the history of uploaded images.
# This makes the app feel more interactive and stateful.
# initialize it here to prevent it from being reset on every script rerun.
if 'history' not in st.session_state:
    st.session_state.history = []

# --- MODEL LOADING ---
# To improve performance, caching the model resource.
# This means the heavy YOLO model is loaded only once, not every time a widget is used.
@st.cache_resource
def load_yolo_model(model_path):
    """
    Loads the YOLOv8 model from the specified path.
    Returns the model object or None if loading fails.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the YOLO model: {e}")
        return None

# Load  custom-trained model. 'best.pt' must be in the same directory.
model = load_yolo_model('best.pt')

# --- SIDEBAR ---
# organized all user controls into a sidebar for a clean layout.
with st.sidebar:
    st.title("🅿️ Pro Parking Counter")
    st.write("---")
    
    st.header("⚙️ Controls")
    
    # added a slider to let the user control the model's confidence threshold.
    # This allows for dynamically filtering out less certain predictions.
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.60, # A reasonable default value.
        step=0.05,
        help="Only detections with confidence above this threshold will be shown."
    )

    st.write("---")

    # The main user interaction: the file uploader.
    uploaded_file = st.file_uploader(
        "Upload a parking lot image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    st.write("---")
    
    # Here, display the history of the last few uploads.
    st.header("📜 Upload History")
    if not st.session_state.history:
        st.info("Your recent analyses will appear here.")
    else:
        # Loop through the history records stored in the session state.
        for record in st.session_state.history:
            st.image(record['image'], caption=f"Analyzed at {record['timestamp']}", width=150)
            st.text(f"Occupied: {record['occupied']}, Empty: {record['empty']}")
            st.markdown("---")

# --- MAIN PAGE ---
# The main content area of the dashboard.
st.title("Parking Lot Analysis Dashboard")

# Check if a file has been uploaded and if the model loaded correctly.
if uploaded_file is not None and model is not None:
    # 1. Image Preparation
    # Reading the uploaded file bytes into a format that OpenCV can use.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # need to convert the image from BGR (OpenCV's default) to RGB for correct color display in Streamlit.
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # 2. Model Inference
    # Display a spinner while the model is processing the image.
    with st.spinner('Analyzing image and detecting spots...'):
        results = model(opencv_image)

    # 3. Process Detection Results
    occupied_count = 0
    empty_count = 0
    annotated_image = rgb_image.copy() # Make a copy to draw the detection boxes on.

    # Loop through all detections found by the model.
    for result in results:
        for box in result.boxes:
            # Apply the user-defined confidence threshold.
            if box.conf[0] > confidence_threshold:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Set the box color and increment the appropriate counter.
                if class_name == 'occupied':
                    occupied_count += 1
                    color = (255, 0, 0) # Red for occupied
                else: # 'empty'
                    empty_count += 1
                    color = (0, 255, 0) # Green for empty
                
                # Get the box coordinates and draw the rectangle and label.
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_image, class_name.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    total_spots = occupied_count + empty_count
    
    st.success(f"Detection Complete! Found {total_spots} valid spots.")
    st.write("---")

    # 4. Display Results in a Dashboard Layout
    # using columns to create a side-by-side dashboard view.
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Occupancy Breakdown")
        
        # Only attempt to draw the chart if spots were actually found.
        if total_spots > 0:
            labels = 'Occupied', 'Empty'
            sizes = [occupied_count, empty_count]
            colors = ['#ff6666', '#99ff99'] # Using softer colors for the chart.
            explode = (0.1, 0)  # "Explode" the first slice (Occupied) for emphasis.

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Ensures the pie chart is a circle.
            st.pyplot(fig1)
        else:
            st.warning("No parking spots detected with the current confidence threshold.")
            
        # Display the final counts as clear metrics.
        st.metric(label="Occupied Spots", value=occupied_count)
        st.metric(label="Empty Spots", value=empty_count)

    with col2:
        st.subheader("🖼️ Annotated Image")
        st.image(annotated_image, use_container_width=True)

    # 5. Update the Session State History
    # Get the current time for the timestamp.
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # need to use the original uploaded file to create a PIL Image for the history.
    pil_image_for_history = Image.open(uploaded_file)
    
    # Add the latest analysis to the beginning of the history list.
    st.session_state.history.insert(0, {
        "image": pil_image_for_history,
        "occupied": occupied_count,
        "empty": empty_count,
        "timestamp": current_time
    })
    
    # To keep the app from getting too cluttered, limit the history to the last 5 uploads.
    if len(st.session_state.history) > 5:
        st.session_state.history.pop()

else:
    # This message is shown before any image is uploaded.
    st.info("Awaiting image upload via the sidebar...")
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- PAGE CONFIGURATION ---
# This is a good practice to set up the page at the start.
st.set_page_config(
    page_title="Pro Parking Lot Counter", 
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
# Session state is Streamlit's way of "remembering" things.
# We initialize it here to store our upload history.
if 'history' not in st.session_state:
    st.session_state.history = []

# --- MODEL LOADING ---
# We cache the model loading so it doesn't reload on every interaction.
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model('best.pt')

# --- SIDEBAR ---
# All our controls will go here.
with st.sidebar:
    st.title("🅿️ Pro Parking Counter")
    st.write("---")
    
    st.header("⚙️ Controls")
    
    # Create a slider for the confidence threshold.
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.60, # Default value
        step=0.05
    )

    st.write("---")

    # The image uploader is now in the sidebar.
    uploaded_file = st.file_uploader(
        "Upload a parking lot image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    st.write("---")
    
    # Display the upload history.
    st.header("📜 Upload History")
    if not st.session_state.history:
        st.info("Your uploaded images will appear here.")
    else:
        for i, record in enumerate(st.session_state.history):
            # Show a small thumbnail of the previous image.
            st.image(record['image'], caption=f"Uploaded: {record['timestamp']}", width=150)
            st.text(f"Occupied: {record['occupied']}, Empty: {record['empty']}")
            st.markdown("---")


# --- MAIN PAGE ---
st.title("Parking Lot Analysis Dashboard")
st.write("Upload an image via the sidebar to begin analysis.")

if uploaded_file is not None and model is not None:
    # 1. Read and prepare the image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # 2. Run the YOLO model.
    with st.spinner('Running detection...'):
        results = model(opencv_image)

    # 3. Process results and draw on the image.
    occupied_count = 0
    empty_count = 0
    annotated_image = rgb_image.copy()

    for result in results:
        for box in result.boxes:
            if box.conf[0] > confidence_threshold:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                if class_name == 'occupied':
                    occupied_count += 1
                    color = (255, 0, 0) # Red
                else: # 'empty'
                    empty_count += 1
                    color = (0, 255, 0) # Green
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_image, class_name.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    total_spots = occupied_count + empty_count
    
    st.success(f"Detection Complete! Found {total_spots} valid spots.")
    st.write("---")

    # 4. Display results in a professional layout.
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Occupancy Breakdown")
        
        # Create the pie chart if spots were detected.
        if total_spots > 0:
            labels = 'Occupied', 'Empty'
            sizes = [occupied_count, empty_count]
            colors = ['#ff6666', '#99ff99'] # Light red, light green
            explode = (0.1, 0)  # Explode the 'Occupied' slice

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig1)
        else:
            st.warning("No parking spots detected with the current confidence threshold.")
            
        # Display the metrics.
        st.metric(label="Occupied Spots", value=occupied_count)
        st.metric(label="Empty Spots", value=empty_count)

    with col2:
        st.subheader("🖼️ Annotated Image")
        st.image(annotated_image, use_container_width=True)

    # 5. Add the current result to the history.
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Convert uploaded image to a format that can be stored in session state
    pil_image_for_history = Image.open(uploaded_file)
    
    st.session_state.history.insert(0, {
        "image": pil_image_for_history,
        "occupied": occupied_count,
        "empty": empty_count,
        "timestamp": current_time
    })
    
    # Keep history to a reasonable size (e.g., last 5 uploads)
    if len(st.session_state.history) > 5:
        st.session_state.history.pop()

else:
    st.info("Awaiting image upload...")
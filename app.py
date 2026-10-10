import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️")

# Must be the same order as printed in Colab
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

INFO = {
    "cardboard": ("Recyclable", "Flatten the box, keep it dry and put it in the dry-waste/recycling bin."),
    "glass": ("Recyclable", "Rinse it and keep it separate from other waste. Wrap broken glass before disposal."),
    "metal": ("Recyclable", "Rinse cans and put them in the dry-waste/recycling bin or give them to a scrap collector."),
    "paper": ("Recyclable", "Keep it clean and dry and put it in the dry-waste/recycling bin. Greasy or wet paper is not recyclable."),
    "plastic": ("Recyclable", "Rinse it, dry it and put it in the dry-waste bin. Avoid single-use plastic where you can."),
    "trash": ("Non-recyclable", "Put it in the general waste bin. Try to reduce this kind of waste."),
}


@st.cache_resource
def get_model():
    return load_model("model.keras")


model = get_model()

st.title("♻️ AI-Based Waste Classification System")
st.write("Upload a photo of a waste item and the AI will tell you what it is and how to dispose of it.")

file = st.file_uploader("Upload a waste photo", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Your image", use_container_width=True)

    x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), axis=0)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    label = CLASSES[idx]
    conf = float(probs[idx]) * 100

    category, tip = INFO[label]
    st.success(f"Prediction: **{label.upper()}** ({conf:.1f}% confident)")
    st.write(f"**Category:** {category}")
    st.info(f"**How to dispose:** {tip}")

    if conf < 60:
        st.warning("The model is not very sure. Try a clearer photo with a plain background.")

    st.subheader("All probabilities")
    for name, p in zip(CLASSES, probs):
        st.progress(float(p), text=f"{name}: {p * 100:.1f}%")
@st.cache_resource
def load_my_model():
    # This prevents the app from hanging on initial browser load
    from tensorflow.keras.models import load_model
    return load_model("model.keras")

# Load the model cleanly
model = load_my_model()

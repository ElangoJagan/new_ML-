import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import load_config, load_model, predict

## Page Config 

st.set_page_config('SMS SPAM DETECTION', layout= 'centered')

# load model Once
@st.cache_resource
def get_model():
    config = load_config()
    model, vectorizer = load_model(config)
    return model, vectorizer, config

model, vectorizer, config = get_model()

# Header
st.title('SMS SPAM DECTECTOR')
st.markdown('Detect whether mail  is spam or ham ')
st.divider()


# input 
message = st.text_area('Enter your Messages Here!', height=150, placeholder= 'Type or paste you ping ')

#predict button
if st.button('Check Message ', use_container_width=True):
    if message.strip() =="":
        st.warning('Please enter a messae first!')
    else:
        with st.spinner('Analyzinf messages.....'):
            result, confidence = predict(message,model, vectorizer)
            
        st.divider()
        
        # Show rsutl
        if result == "Spam":
            st.error(f"🚨 **SPAM DETECTED!**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            st.markdown("⚠️ This message looks like spam. Be careful!")
        else:
            st.success(f"✅ **HAM - Not Spam!**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            st.markdown("👍 This message looks safe!")
        
        st.divider()
        
        # Show details
        with st.expander("📊 See Details"):
            st.write(f"**Original Message:** {message}")
            st.write(f"**Prediction:** {result}")
            st.write(f"**Confidence:** {confidence:.2f}%")
# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("""
    - **Model:** Naive Bayes
    - **Accuracy:** 97.67%
    - **Precision:** 100%
    - **Recall:** 82.67%
    - **F1 Score:** 90.51%
    """)
    
    st.divider()
    
    st.header("📝 Try These Examples")
    
    examples = [
        "FREE entry! Win £1000 cash prize now!",
        "Hey, are we meeting for lunch tomorrow?",
        "URGENT! Your account has been compromised!",
        "Can you send me the project files please?",
    ]
    
    for example in examples:
        if st.button(example[:40] + "...", use_container_width=True):
            st.session_state['example'] = example

    st.divider()
    st.markdown("Built with ❤️ using Streamlit & Scikit-learn")
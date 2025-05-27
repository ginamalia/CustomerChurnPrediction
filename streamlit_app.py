import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import export_text
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model dan preprocessing objects
@st.cache_resource
def load_model_artifacts():
    """Load model dan semua preprocessing objects"""
    try:
        artifacts = joblib.load('churn_prediction_model.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return artifacts, metadata
    except FileNotFoundError:
        st.error("❌ Model files tidak ditemukan! Pastikan Anda sudah menjalankan 'train_and_save_model.py' terlebih dahulu.")
        st.stop()

# Load artifacts
model_artifacts, metadata = load_model_artifacts()

# Extract components
model = model_artifacts['model']
scaler = model_artifacts['scaler']
label_encoders = model_artifacts['label_encoders']
selected_features = model_artifacts['selected_features']
feature_names = model_artifacts['feature_names']
categorical_columns = model_artifacts['categorical_columns']

# Header
st.title("🎯 Customer Churn Prediction System")
st.markdown("**Sistem Prediksi Churn Pelanggan menggunakan Decision Tree**")

# Sidebar untuk navigasi
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox("Pilih Halaman", [
    "🏠 Home", 
    "🔍 Single Prediction", 
    "📊 Batch Prediction", 
    "📈 Model Info",
    "🌳 Decision Tree Rules"
])

if page == "🏠 Home":
    st.header("Selamat Datang di Customer Churn Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Tentang Sistem")
        st.write("""
        Sistem ini menggunakan algoritma **Decision Tree** untuk memprediksi kemungkinan 
        pelanggan akan churn (berhenti menggunakan layanan). 
        
        **Fitur Utama:**
        - Prediksi individual pelanggan
        - Prediksi batch menggunakan file CSV
        - Visualisasi hasil prediksi
        - Analisis decision tree rules
        """)
        
        st.subheader("🎯 Model Performance")
        accuracy = metadata['performance']['accuracy']
        st.metric("Accuracy", f"{accuracy:.2%}")
        
    with col2:
        st.subheader("📋 Fitur yang Digunakan")
        features_df = pd.DataFrame({
            'Fitur': selected_features,
            'Deskripsi': [
                'Skor kredit pelanggan',
                'Usia pelanggan', 
                'Saldo rekening',
                'Jumlah produk yang dimiliki',
                'Apakah pelanggan aktif'
            ][:len(selected_features)]
        })
        st.dataframe(features_df, hide_index=True)

elif page == "🔍 Single Prediction":
    st.header("🔍 Prediksi Individual")
    st.write("Masukkan data pelanggan untuk mendapatkan prediksi churn")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Inputs berdasarkan fitur yang tersedia
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 40)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            
        with col2:
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
            num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card = st.selectbox("Has Credit Card", [0, 1])
            is_active_member = st.selectbox("Is Active Member", [0, 1])
            estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
        
        submit_button = st.form_submit_button("🔮 Prediksi Churn")
        
        if submit_button:
            # Persiapkan data input
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Geography': [geography],
                'Gender': [gender],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })
            
            # Preprocessing
            for col in categorical_columns:
                if col in input_data.columns:
                    input_data[col] = label_encoders[col].transform(input_data[col])
            
            # Seleksi fitur dan scaling
            input_selected = input_data[selected_features]
            input_scaled = scaler.transform(input_selected)
            
            # Prediksi
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Tampilkan hasil
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 HIGH RISK - Pelanggan berpotensi CHURN")
                else:
                    st.success("✅ LOW RISK - Pelanggan tidak berpotensi churn")
            
            with col2:
                st.metric("Probabilitas Churn", f"{probability[1]:.2%}")
            
            with col3:
                st.metric("Probabilitas Stay", f"{probability[0]:.2%}")
            
            # Visualisasi probabilitas
            fig = go.Figure(data=[
                go.Bar(x=['Stay', 'Churn'], 
                       y=[probability[0], probability[1]],
                       marker_color=['green', 'red'])
            ])
            fig.update_layout(title="Probabilitas Prediksi", yaxis_title="Probabilitas")
            st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Batch Prediction":
    st.header("📊 Prediksi Batch")
    st.write("Upload file CSV untuk prediksi multiple pelanggan")
    
    # Template download
    st.subheader("📥 Download Template")
    template_data = pd.DataFrame({
        'CreditScore': [650, 700, 580],
        'Geography': ['France', 'Germany', 'Spain'], 
        'Gender': ['Male', 'Female', 'Male'],
        'Age': [42, 35, 28],
        'Tenure': [2, 5, 3],
        'Balance': [125000, 0, 150000],
        'NumOfProducts': [1, 2, 1],
        'HasCrCard': [1, 0, 1],
        'IsActiveMember': [1, 1, 0],
        'EstimatedSalary': [79000, 60000, 85000]
    })
    
    st.download_button(
        label="📥 Download Template CSV",
        data=template_data.to_csv(index=False),
        file_name="customer_template.csv",
        mime="text/csv"
    )
    
    # File upload
    st.subheader("📤 Upload File")
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read data
            batch_data = pd.read_csv(uploaded_file)
            st.write("✅ File berhasil diupload!")
            st.write(f"Jumlah data: {len(batch_data)} pelanggan")
            
            # Preview data
            st.subheader("👀 Preview Data")
            st.dataframe(batch_data.head())
            
            if st.button("🔮 Jalankan Prediksi Batch"):
                # Preprocessing
                batch_processed = batch_data.copy()
                
                for col in categorical_columns:
                    if col in batch_processed.columns:
                        batch_processed[col] = label_encoders[col].transform(batch_processed[col])
                
                # Seleksi fitur dan scaling
                batch_selected = batch_processed[selected_features]
                batch_scaled = scaler.transform(batch_selected)
                
                # Prediksi
                predictions = model.predict(batch_scaled)
                probabilities = model.predict_proba(batch_scaled)
                
                # Gabungkan hasil
                results = batch_data.copy()
                results['Churn_Prediction'] = predictions
                results['Churn_Probability'] = probabilities[:, 1]
                results['Risk_Level'] = ['HIGH' if p > 0.5 else 'LOW' for p in probabilities[:, 1]]
                
                # Tampilkan hasil
                st.subheader("📊 Hasil Prediksi")
                st.dataframe(results)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Pelanggan", len(results))
                with col2:
                    high_risk = sum(results['Churn_Prediction'])
                    st.metric("High Risk Customers", high_risk)
                with col3:
                    churn_rate = high_risk / len(results) * 100
                    st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                
                # Visualisasi
                fig = px.histogram(results, x='Risk_Level', color='Risk_Level',
                                 title="Distribusi Risk Level")
                st.plotly_chart(fig, use_container_width=True)
                
                # Download hasil
                st.download_button(
                    label="📥 Download Hasil Prediksi",
                    data=results.to_csv(index=False),
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error memproses file: {str(e)}")

elif page == "📈 Model Info":
    st.header("📈 Informasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Details")
        st.write(f"**Model Type:** {metadata['model_type']}")
        st.write(f"**Accuracy:** {metadata['performance']['accuracy']:.2%}")
        st.write(f"**Training Samples:** {metadata['performance']['training_samples']:,}")
        st.write(f"**Test Samples:** {metadata['performance']['test_samples']:,}")
        
        st.subheader("🔧 Preprocessing")
        st.write(f"**Scaler:** {metadata['preprocessing']['scaler']}")
        st.write(f"**Categorical Encoding:** {metadata['preprocessing']['categorical_encoding']}")
        st.write(f"**Feature Selection:** {metadata['preprocessing']['feature_selection']}")
    
    with col2:
        st.subheader("📊 Selected Features")
        features_df = pd.DataFrame({
            'Feature': selected_features,
            'Index': range(len(selected_features))
        })
        st.dataframe(features_df, hide_index=True)
        
        # Feature importance visualization (if available)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

elif page == "🌳 Decision Tree Rules":
    st.header("🌳 Decision Tree Rules")
    st.write("Aturan keputusan yang digunakan model untuk prediksi")
    
    # Export tree rules
    tree_rules = export_text(model, feature_names=selected_features, max_depth=4)
    
    st.subheader("📋 Tree Rules")
    st.code(tree_rules, language='text')
    
    # Interpretasi rules
    st.subheader("💡 Interpretasi Rules")
    st.write("""
    Decision Tree menggunakan serangkaian kondisi if-else untuk membuat prediksi. 
    Setiap node dalam tree merepresentasikan keputusan berdasarkan satu fitur.
    
    **Cara membaca:**
    - Setiap baris menunjukkan kondisi (misal: Age <= 43.5)
    - Angka di akhir menunjukkan jumlah sampel di setiap class [tidak churn, churn]
    - Prediksi dibuat berdasarkan mayoritas class di leaf node
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Customer Churn Prediction**
Dikembangkan menggunakan CRISP-DM methodology

🔧 Tech Stack:
- Streamlit
- Scikit-learn  
- Plotly
- Joblib
""")
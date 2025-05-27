# -*- coding: utf-8 -*-
"""
Customer Churn Prediction - Model Training and Saving
CRISP-DM Implementation with Model Persistence
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    """
    Complete CRISP-DM pipeline: Data Understanding, Preparation, Modeling, and Deployment preparation
    """
    
    print("=== CRISP-DM: Customer Churn Prediction ===")
    print("\n1. BUSINESS UNDERSTANDING")
    print("Tujuan: Memprediksi potensi churn pelanggan untuk meningkatkan retensi")
    
    # 2. DATA UNDERSTANDING
    print("\n2. DATA UNDERSTANDING")
    try:
        data = pd.read_csv('Churn.csv')
        print(f"Dataset berhasil dimuat dengan shape: {data.shape}")
        print(f"Missing values: {data.isnull().sum().sum()}")
    except FileNotFoundError:
        print("Error: File 'Churn.csv' tidak ditemukan!")
        return None
    
    # 3. DATA PREPARATION
    print("\n3. DATA PREPARATION")
    
    # Hapus kolom yang tidak relevan
    data_clean = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    
    # Simpan label encoders untuk deployment
    label_encoders = {}
    categorical_columns = ['Geography', 'Gender']
    
    for column in categorical_columns:
        le = LabelEncoder()
        data_clean[column] = le.fit_transform(data_clean[column])
        label_encoders[column] = le
    
    # Feature Selection berdasarkan importance
    X = data_clean.drop(columns=['Exited'])
    y = data_clean['Exited']
    
    # Feature Importance menggunakan Random Forest
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Seleksi fitur berdasarkan threshold
    importance_threshold = 0.1
    selected_features = feature_importance[
        feature_importance['importance'] > importance_threshold
    ]['feature'].tolist()
    
    print(f"Selected features: {selected_features}")
    
    X_selected = X[selected_features]
    
    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
    
    # 4. MODELING
    print("\n4. MODELING")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Decision Tree Model
    dt_model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=4,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    
    dt_model.fit(X_train, y_train)
    
    # 5. EVALUATION
    print("\n5. EVALUATION")
    y_pred = dt_model.predict(X_test)
    
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # 6. DEPLOYMENT PREPARATION
    print("\n6. DEPLOYMENT PREPARATION")
    
    # Simpan semua komponen yang diperlukan untuk deployment
    model_artifacts = {
        'model': dt_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'selected_features': selected_features,
        'feature_names': list(X.columns),
        'categorical_columns': categorical_columns
    }
    
    # Simpan menggunakan joblib
    joblib.dump(model_artifacts, 'churn_prediction_model.pkl')
    print("Model dan preprocessing objects berhasil disimpan ke 'churn_prediction_model.pkl'")
    
    # Simpan juga metadata untuk referensi
    metadata = {
        'model_type': 'Decision Tree (Entropy)',
        'selected_features': selected_features,
        'performance': {
            'accuracy': dt_model.score(X_test, y_test),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        },
        'preprocessing': {
            'scaler': 'MinMaxScaler',
            'categorical_encoding': 'LabelEncoder',
            'feature_selection': f'Random Forest Importance > {importance_threshold}'
        }
    }
    
    joblib.dump(metadata, 'model_metadata.pkl')
    print("Metadata model berhasil disimpan ke 'model_metadata.pkl'")
    
    return model_artifacts, metadata

if __name__ == "__main__":
    artifacts, metadata = train_and_save_model()
    
    if artifacts:
        print("\n" + "="*50)
        print("MODEL TRAINING SELESAI!")
        print("File yang dihasilkan:")
        print("- churn_prediction_model.pkl (model + preprocessing)")
        print("- model_metadata.pkl (metadata)")
        print("="*50)
        
        print("\nLangkah selanjutnya:")
        print("1. Jalankan: streamlit run streamlit_app.py")
        print("2. Buka browser di http://localhost:8501")
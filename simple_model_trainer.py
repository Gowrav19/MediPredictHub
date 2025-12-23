#!/usr/bin/env python3
"""
Simple Model Trainer - Avoids dependency conflicts
Trains basic but effective models for medical diagnosis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def train_diabetes_model():
    """Train diabetes prediction model"""
    print("Training Diabetes Model...")
    
    # Load dataset
    df = pd.read_csv('datasets/diabetes_dataset.csv')
    
    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (most reliable)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Diabetes Model Accuracy: {accuracy:.3f}")
    
    # Save model and scaler
    os.makedirs('ml_models', exist_ok=True)
    with open('ml_models/diabetes_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('ml_models/diabetes_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler, accuracy

def train_heart_model():
    """Train heart disease prediction model"""
    print("Training Heart Disease Model...")
    
    # Create synthetic heart disease data (since we don't have the original dataset)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic heart disease features
    data = {
        'chest_pain': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'shortness_of_breath': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'fatigue': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'palpitations': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'dizziness': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'swelling': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'pain_arms_jaw_back': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'cold_sweats_nausea': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'high_bp': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'high_cholesterol': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'obesity': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'sedentary_lifestyle': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'family_history': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'chronic_stress': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'gender': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'age': np.random.randint(18, 80, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on risk factors
    risk_score = (
        df['chest_pain'] * 2 +
        df['shortness_of_breath'] * 1.5 +
        df['fatigue'] * 1 +
        df['palpitations'] * 2 +
        df['dizziness'] * 1.5 +
        df['swelling'] * 1.5 +
        df['pain_arms_jaw_back'] * 2 +
        df['cold_sweats_nausea'] * 2 +
        df['high_bp'] * 2 +
        df['high_cholesterol'] * 1.5 +
        df['diabetes'] * 2 +
        df['smoking'] * 2 +
        df['obesity'] * 1.5 +
        df['sedentary_lifestyle'] * 1 +
        df['family_history'] * 1.5 +
        df['chronic_stress'] * 1 +
        (df['gender'] == 1) * 0.5 +  # Male slightly higher risk
        (df['age'] > 50) * 1 +
        (df['age'] > 65) * 1
    )
    
    # Create binary target (1 if high risk, 0 if low risk)
    df['target'] = (risk_score > 8).astype(int)
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    

    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Disease Model Accuracy: {accuracy:.3f}")
    
    # Save model and scaler
    with open('ml_models/heart_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('ml_models/heart_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler, accuracy

def train_cancer_model():
    """Train breast cancer classification model"""
    print("Training Breast Cancer Model...")
    
    # Create synthetic breast cancer data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic tumor characteristics
    data = {
        'mean_radius': np.random.normal(14, 3, n_samples),
        'mean_texture': np.random.normal(19, 4, n_samples),
        'mean_perimeter': np.random.normal(92, 24, n_samples),
        'mean_area': np.random.normal(655, 350, n_samples),
        'mean_smoothness': np.random.normal(0.096, 0.014, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure realistic ranges
    df['mean_radius'] = np.clip(df['mean_radius'], 6, 30)
    df['mean_texture'] = np.clip(df['mean_texture'], 9, 40)
    df['mean_perimeter'] = np.clip(df['mean_perimeter'], 43, 190)
    df['mean_area'] = np.clip(df['mean_area'], 140, 2500)
    df['mean_smoothness'] = np.clip(df['mean_smoothness'], 0.05, 0.16)
    
    # Create target based on tumor characteristics
    # Higher radius, perimeter, area and lower smoothness = more likely malignant
    malignancy_score = (
        (df['mean_radius'] - 6) / 24 * 2 +  # Normalize and weight
        (df['mean_texture'] - 9) / 31 * 1.5 +
        (df['mean_perimeter'] - 43) / 147 * 2 +
        (df['mean_area'] - 140) / 2360 * 1.5 +
        (0.16 - df['mean_smoothness']) / 0.11 * 2  # Lower smoothness = higher risk
    )
    
    # Create binary target (1 if malignant, 0 if benign)
    df['target'] = (malignancy_score > 2.5).astype(int)
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Breast Cancer Model Accuracy: {accuracy:.3f}")
    
    # Save model and scaler
    with open('ml_models/cancer_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('ml_models/cancer_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler, accuracy

def main():
    """Train all models"""
    print("Starting Simple Model Training...")
    print("=" * 50)
    
    # Create models directory
    os.makedirs('ml_models', exist_ok=True)
    
    # Train all models
    diabetes_model, diabetes_scaler, diabetes_acc = train_diabetes_model()
    heart_model, heart_scaler, heart_acc = train_heart_model()
    cancer_model, cancer_scaler, cancer_acc = train_cancer_model()
    
    print("=" * 50)
    print("All Models Trained Successfully!")
    print(f"Diabetes Model Accuracy: {diabetes_acc:.3f}")
    print(f"Heart Disease Model Accuracy: {heart_acc:.3f}")
    print(f"Breast Cancer Model Accuracy: {cancer_acc:.3f}")
    print("=" * 50)
    print("Models saved to ml_models/ directory")
    print("You can now run: python app.py")

if __name__ == "__main__":
    main()

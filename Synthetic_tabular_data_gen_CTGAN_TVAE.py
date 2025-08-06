import pandas as pd
import numpy as np
from ctgan import CTGANSynthesizer
from sdv.metrics.tabular import CSTest, KSTest
from sdv.evaluation import evaluate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# 1. Data Preparation
# Synthetic medical records dataset
data = """patient_id,age,gender,blood_pressure,cholesterol,bmi,diabetes,hypertension,heart_disease,medication_cost
P001,45,M,120/80,190,26.5,0,0,0,350.50
P002,67,F,140/90,240,31.2,1,1,0,820.75
P003,52,M,130/85,210,28.7,0,1,0,420.30
P004,38,F,110/70,180,24.1,0,0,0,280.90
P005,71,M,150/95,260,33.5,1,1,1,1250.40
P006,29,F,115/75,170,22.8,0,0,0,190.25
P007,63,M,145/92,230,30.1,1,1,0,950.60
P008,41,F,125/82,195,27.3,0,0,0,380.45
P009,56,M,135/88,220,29.5,0,1,1,780.30
P010,34,F,118/76,185,23.7,0,0,0,310.75
P011,49,M,142/91,235,31.8,1,1,0,890.20
P012,72,F,155/98,270,34.2,1,1,1,1420.50
P013,31,M,122/78,175,25.0,0,0,0,270.40
P014,58,F,138/87,225,30.5,0,1,0,670.80
P015,43,M,128/83,200,27.9,0,0,0,410.60
P016,65,F,148/93,250,32.7,1,1,1,1100.25
P017,37,M,116/74,182,24.5,0,0,0,230.90
P018,54,F,132/86,215,28.9,0,1,0,590.40
P019,69,M,152/96,265,33.9,1,1,1,1350.75
P020,26,F,112/72,168,21.9,0,0,0,180.30
P021,47,M,124/81,205,26.8,0,0,0,370.50
P022,60,F,143/90,245,31.5,1,1,0,1020.90
P023,33,M,119/77,178,24.8,0,0,0,250.60
P024,51,F,129/84,212,29.1,0,1,0,540.70
P025,74,M,158/100,280,35.0,1,1,1,1580.40
P026,28,F,114/73,172,22.5,0,0,0,200.80
P027,62,M,141/89,238,30.8,1,1,0,920.30
P028,39,F,123/80,192,26.2,0,0,0,330.40
P029,55,M,134/87,218,29.8,0,1,1,710.50
P030,70,F,149/94,255,33.1,1,1,1,1180.60
P031,30,M,121/79,176,25.3,0,0,0,260.90
P032,57,F,136/88,228,30.2,0,1,0,630.20
P033,44,M,126/82,198,27.6,0,0,0,390.70
P034,68,F,147/92,253,32.5,1,1,1,1070.80
P035,35,M,117/75,181,24.3,0,0,0,240.50
P036,53,F,131/85,208,28.5,0,1,0,510.90
P037,66,M,144/91,242,31.9,1,1,0,980.40
P038,42,F,127/83,197,27.1,0,0,0,360.80
P039,59,M,137/89,232,30.0,0,1,1,750.60
P040,73,F,154/97,275,34.5,1,1,1,1480.20
P041,27,M,113/71,169,22.1,0,0,0,190.40
P042,64,F,139/89,235,31.0,1,1,0,870.50
P043,36,M,118/76,183,24.7,0,0,0,230.70
P044,50,F,130/84,210,28.3,0,1,0,480.30
P045,75,M,156/99,278,34.8,1,1,1,1520.90
P046,32,F,120/78,174,25.5,0,0,0,270.80
P047,61,M,140/88,236,30.6,1,1,0,940.70
P048,48,F,125/81,202,27.4,0,0,0,350.20
P049,56,M,133/86,222,29.6,0,1,1,690.40
P050,71,F,151/95,258,33.3,1,1,1,1220.50"""

# Load data into DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# 2. Data Preprocessing
# Split blood pressure into systolic and diastolic
df[['systolic_bp', 'diastolic_bp']] = df['blood_pressure'].str.split('/', expand=True).astype(int)
df.drop('blood_pressure', axis=1, inplace=True)

# Convert categoricals
df['gender'] = df['gender'].map({'M': 0, 'F': 1})

# Define discrete columns
discrete_columns = ['gender', 'diabetes', 'hypertension', 'heart_disease']

# 3. CTGAN Model Training
def train_ctgan(data, discrete_columns, dp=False, epsilon=None):
    """Train CTGAN model with optional differential privacy"""
    ctgan = CTGANSynthesizer(
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        batch_size=500,
        epochs=100,
        verbose=True,
        dp=dp,
        epsilon=epsilon if dp else None
    )
    
    ctgan.fit(data, discrete_columns)
    return ctgan

# Train regular CTGAN
print("Training standard CTGAN model...")
ctgan = train_ctgan(df, discrete_columns)

# Train DP-CTGAN
print("\nTraining DP-CTGAN model (ε=1.0)...")
dp_ctgan = train_ctgan(df, discrete_columns, dp=True, epsilon=1.0)

# 4. Synthetic Data Generation
def generate_samples(model, num_samples):
    """Generate synthetic samples"""
    synthetic_data = model.sample(num_samples)
    
    # Convert gender back to categorical
    synthetic_data['gender'] = synthetic_data['gender'].map({0: 'M', 1: 'F'})
    
    # Recreate blood pressure column
    synthetic_data['blood_pressure'] = (
        synthetic_data['systolic_bp'].astype(str) + '/' + 
        synthetic_data['diastolic_bp'].astype(str)
    synthetic_data.drop(['systolic_bp', 'diastolic_bp'], axis=1, inplace=True)
    
    return synthetic_data

# Generate samples
synth_df = generate_samples(ctgan, len(df))
dp_synth_df = generate_samples(dp_ctgan, len(df))

# 5. Data Quality Evaluation
def evaluate_synthetic(real_data, synthetic_data):
    """Evaluate synthetic data quality"""
    # Statistical tests
    ks_results = {}
    cs_results = {}
    
    for column in real_data.select_dtypes(include=np.number).columns:
        ks_results[column] = KSTest.compute(real_data[column], synthetic_data[column])
        cs_results[column] = CSTest.compute(real_data[column], synthetic_data[column])
    
    # SDV comprehensive evaluation
    evaluation = evaluate(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata={
            'fields': {
                'patient_id': {'type': 'id'},
                'age': {'type': 'numerical'},
                'gender': {'type': 'categorical'},
                'cholesterol': {'type': 'numerical'},
                'bmi': {'type': 'numerical'},
                'diabetes': {'type': 'boolean'},
                'hypertension': {'type': 'boolean'},
                'heart_disease': {'type': 'boolean'},
                'medication_cost': {'type': 'numerical'},
                'blood_pressure': {'type': 'categorical'}
            }
        }
    )
    
    return {
        'ks_test': ks_results,
        'cs_test': cs_results,
        'sdv_score': evaluation
    }

print("\nEvaluating standard CTGAN samples...")
ctgan_eval = evaluate_synthetic(df, synth_df)

print("\nEvaluating DP-CTGAN samples...")
dp_ctgan_eval = evaluate_synthetic(df, dp_synth_df)

# 6. Visualization
def plot_comparison(real_col, synth_col, dp_synth_col, title):
    """Plot comparison of real vs synthetic distributions"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(real_col, bins=20, alpha=0.7, color='blue')
    plt.title('Real Data')
    
    plt.subplot(1, 3, 2)
    plt.hist(synth_col, bins=20, alpha=0.7, color='green')
    plt.title('CTGAN Synthetic')
    
    plt.subplot(1, 3, 3)
    plt.hist(dp_synth_col, bins=20, alpha=0.7, color='red')
    plt.title('DP-CTGAN Synthetic')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Compare numerical features
numerical_cols = ['age', 'cholesterol', 'bmi', 'medication_cost']
for col in numerical_cols:
    plot_comparison(
        df[col],
        synth_df[col],
        dp_synth_df[col],
        f'Distribution Comparison: {col}'
    )

# 7. Privacy Evaluation
def privacy_evaluation(real_data, synthetic_data, sensitive_columns):
    """Evaluate privacy protection"""
    matches = 0
    for _, real_row in real_data.iterrows():
        for _, synth_row in synthetic_data.iterrows():
            if all(real_row[col] == synth_row[col] for col in sensitive_columns):
                matches += 1
                break
    
    return 1 - (matches / len(real_data))

sensitive_cols = ['age', 'gender', 'blood_pressure']
privacy_score = privacy_evaluation(df, synth_df, sensitive_cols)
dp_privacy_score = privacy_evaluation(df, dp_synth_df, sensitive_cols)

print(f"\nPrivacy Protection Scores:")
print(f"Standard CTGAN: {privacy_score:.2%}")
print(f"DP-CTGAN: {dp_privacy_score:.2%}")

# 8. Save Results
synth_df.to_csv('synthetic_medical_records.csv', index=False)
dp_synth_df.to_csv('dp_synthetic_medical_records.csv', index=False)

print("\nSynthetic data generation complete!")
print(f"Standard CTGAN quality score: {ctgan_eval['sdv_score']:.3f}")
print(f"DP-CTGAN quality score: {dp_ctgan_eval['sdv_score']:.3f}")

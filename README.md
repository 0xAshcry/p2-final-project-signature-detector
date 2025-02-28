# FraudProof: Real-Time Fraud Detection & Signature Authentication System

## Introduction
FraudProof is a fraud detection and signature authentication system designed to safeguard financial transactions. The system consists of two specialized models:

 - Model 1: Predicts fraudulent transactions in real-time using machine learning and Apache Airflow.
 - Model 2: Validates customer signatures using computer vision to enhance security in banking operations.

This dual-layered fraud prevention approach ensures both transactional monitoring and identity validation, providing a comprehensive fraud detection solution.

## Problem Statement
With the rapid advancement of technology, banks face increasing demands for fast and efficient financial transactions, both online and offline. However, this technological progress also heightens the risk of fraud, as cybercriminals continuously develop sophisticated methods to exploit system vulnerabilities. Fraudulent transactions can lead to significant financial losses, erode customer trust, and damage a bank's reputation.

To address this challenge, banks must implement advanced fraud detection systems that leverage machine learning and deep learning to automatically and real-time identify suspicious transaction patterns. A robust fraud detection system should be able to:

 1. Continuously monitor all financial transactions.
 2. Instantly detect anomalies that may indicate fraudulent activities.
 3. Trigger real-time alerts for immediate intervention.
 4. Enhance security while minimizing false positives and customer inconvenience.

Moreover, traditional signature verification processes remain a critical aspect of banking security, particularly for loan approvals, high-value transactions, and account verification. Manual signature verification is time-consuming, prone to human error, and inefficient in handling high transaction volumes. The use of computer vision-based signature verification can:

 1. Automatically validate customer signatures against stored samples.
 2. Improve the accuracy and efficiency of identity verification.
 3. Reduce human workload and operational costs.
 4. Prevent fraudulent document approvals and forged transactions.

By integrating fraud detection and signature authentication, banks can significantly reduce financial risks, enhance customer trust, and ensure operational efficiency. The implementation of these technologies not only strengthens fraud prevention but also enables banks to stay competitive in an increasingly digital and technology-driven financial landscape.

## Machine Learning Model
### Model 1: Transaction Fraud Prediction
#### Purpose:
 1. Monitors incoming transactions.
 2. Predicts whether a transaction is fraudulent.
 3. Triggers alerts for flagged transactions.
 4. Stores predictions for continuous model improvement.

#### Workflow:
 - **Data Query**: Fetch transaction data from PostgreSQL.
 - **Fraud Prediction**: Model predicts fraud probability.
 - **Alert System**: If fraud is detected, an alert is sent to the security team.
 - **Database Logging**: Prediction results are stored for future retraining.

**Model Performance Table**
| **Model**              | **Description**                     | **Key Metrics**                  |
|--------------------|--------------------------------|------------------------------|
| **Transaction Model** | XGBoost | Precision: 99%, Recall: 100% |


## Model 2: Signature Verification using Computer Vision
#### Purpose:
 1. Validates customer signatures against stored samples.
 2. Enhances identity verification for secure transactions.
 3. Prevents fraudulent authorizations in banks.

### Workflow:
 - **Capture Image**: Customer signature is scanned.
 - **Preprocessing**: Convert, normalize, and enhance the image.
 - **CNN Model Prediction**: Compares the scanned signature with reference data.
 - **Decision Output**: Approves or flags the signature for review.

**Model Performance Table**
| **Model**              | **Description**                     | **Key Metrics**                  |
|--------------------|--------------------------------|------------------------------|
| **Signature Verification Model** | CNN, MobileNetV2 | Accuracy: 86%, Loss: 57% |


## System Architecture & Orchestration

FraudProof consists of two distinct workflows orchestrated using **Apache Airflow**:  

1. **Fraud Prediction Model (Model 1)** – Real-time transaction fraud detection.  
2. **Signature Authentication Model (Model 2)** – Continuous re-training for signature verification.  

### Fraud Detection Pipeline (Model 1 - Airflow)  

The **Fraud Prediction Model** continuously monitors financial transactions and detects fraud in real-time. If fraud is suspected, an **email notification is triggered** to alert the IT team. The prediction results are then stored for future **model retraining**.

#### **Pipeline Workflow:**  
 1. **Extract Data** → Transaction data is **queried from the PostgreSQL database**.  
 2. **Predict Data** → The machine learning model processes transactions and **predicts fraud likelihood**.  
 3. **Email Alert** → If fraud is detected, an **email notification** is sent to the IT security team.  
 4. **Load Data** → The prediction results are **stored back into the database** for future training.  

🔹 **Key Technologies**: Apache Airflow, PostgreSQL, XGBoost/Random Forest, Email Alerts  

---

### Signature Verification Pipeline (Model 2 - Airflow)  

The **Signature Verification Model** ensures the authenticity of customer signatures using **computer vision**. To maintain accuracy, the model is **automatically re-trained every 14 business days**.

#### **Pipeline Workflow:**  
 1. **Extract Data** → Customer signature data is retrieved from the database.  
 2. **Re-Train Model** → The model is updated **every 14 business days** using the latest labeled data.  
 3. **Model Verification** → The re-trained model is **verified by the team** before deployment.  
 4. **Model Deployment** → If verified, the new model version is **deployed for production use**.  

**Key Technologies**: Apache Airflow, TensorFlow (CNN), PostgreSQL  

---

By combining **real-time fraud detection** and **continuous signature verification**, **FraudProof** strengthens banking security and minimizes fraudulent activities efficiently.

### Fraud Trends & Insights
Below are key insights derived from our model evaluations:
- **Fraudulent transactions often occur at odd hours**, indicating potential bot-based fraud attempts.
- **High-value transactions** have a greater likelihood of fraud detection.
- **Signature forgery cases are commonly linked to loan approvals**, emphasizing the importance of signature verification.
- **Model retraining every 14 days** improves fraud detection accuracy by adapting to new fraud patterns.

---

## Installation & Usage  

### **Setting up Fraud Detection (Model 1)**
(To Be Made)

### **Using Signature Detection (Model 2)**
(To Be Made)

## Business Impact
FraudProof provides two layers of fraud prevention:

✅ Real-time monitoring of transaction fraud.
✅ Signature authentication to prevent document forgery.
✅ Continuous model improvement for evolving fraud patterns.
✅ Scalable architecture to handle large transaction volumes.

By implementing machine learning-driven fraud detection and computer vision-based signature verification, financial institutions can prevent fraudulent transactions, reduce losses, and increase customer trust.

## Contact & Support
For questions, collaboration, or support, reach out via:<br>
Email: affan.anitya@gmail.com
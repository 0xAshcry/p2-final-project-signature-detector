from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import pickle
import logging

MODEL_PATH = "/opt/airflow/dags/model.pkl"

# Load model 
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Default arguments untuk DAG
default_args = {
    'owner': 'group1',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10)  # Batas waktu eksekusi
}

# Fungsi Extract
def extract_data(**context):
    source_hook = PostgresHook(postgres_conn_id="postgres_airflow")
    source_conn = source_hook.get_conn()

    query = "SELECT * FROM ex_transactions"
    df = pd.read_sql(query, source_conn)

    source_conn.close()

    if df.empty:
        raise ValueError("Data extraction failed or no data returned.")

    context['ti'].xcom_push(key='extracted_data', value=df.to_json())

    logging.info("Data extraction successful.")
    

# Fungsi Predict
def predict_data(**context):
    ti = context['ti']
    df_json = ti.xcom_pull(task_ids='extract', key='extracted_data')

    if df_json is None:
        logging.error("No data extracted!")
        return

    df = pd.read_json(df_json)
    features = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
    
    try:
        df['prediction'] = model.predict(df[features])
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return

    # Tandai data yang terdeteksi fraud (misalnya prediksi = 1)
    fraud_data = df[df['prediction'] == 1]

    if not fraud_data.empty:
        # Simpan data fraud ke XCom untuk digunakan pada task email
        ti.xcom_push(key='fraud_data', value=fraud_data.to_json())
        logging.info(f"Fraud data detected: {fraud_data}")

    ti.xcom_push(key='predicted_data', value=df.to_json())
    logging.info("Prediction successful.")


# Fungsi Load ke PostgreSQL
def load_data(**context):
    # Ambil data prediksi dari XCom
    ti = context['ti']
    df_json = ti.xcom_pull(task_ids='predict', key='predicted_data')

    if df_json is None:
        raise ValueError("No data predicted!")

    # Konversi kembali ke DataFrame
    df = pd.read_json(df_json)

    # Koneksi ke PostgreSQL
    target_hook = PostgresHook(postgres_conn_id="postgres_airflow")
    conn = target_hook.get_conn()
    cursor = conn.cursor()

    # Insert data ke tabel predictions_table dengan menggunakan transaction_id
    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO ex_predictions_table (transaction_id, prediction) 
            VALUES (%s, %s)
            ON CONFLICT (transaction_id) DO UPDATE 
            SET prediction = EXCLUDED.prediction
            """,
            (int(row['transaction_id']), int(row['prediction']))
        )

    # Commit perubahan dan tutup koneksi
    conn.commit()
    cursor.close()
    conn.close()

# Fungsi Email Notification
def email_notification(**context):
    fraud_data_json = context['ti'].xcom_pull(task_ids='predict', key='fraud_data')

    if fraud_data_json is None:
        logging.info("No fraud data to notify.")
        return

    # Mengubah JSON menjadi DataFrame
    fraud_data = pd.read_json(fraud_data_json)

    # # Hapus kolom 'isFraud' dan 'isFlaggedFraud' agar tidak ditampilkan di alert
    # fraud_data = fraud_data.drop(columns=['isFraud', 'isFlaggedFraud'], errors='ignore')

    # Siapkan pesan email
    subject = "🚨 Fraud Detection Alert 🚨"
    body = f"""
    <h3> Alert! Suspicious activity detected. Please conduct an investigation immediately to ensure system security</h3>
    <p>The following transactions were detected as fraud:</p>
    {fraud_data.to_html()} 
    """

    # Kirim email 
    email_task = EmailOperator(
        task_id='send_email',
        to= ['inkadw07@gmail.com', 'liakurniawati507@gmail.com', 'realjokiin@gmail.com', 'affan.anitya@gmail.com', 'aqsal.m@students.amikom.ac.id'], 
        subject=subject,
        html_content=body
    )

    email_task.execute(context)

# Definisi DAG
with DAG(
    'ml_pipeline',
    default_args=default_args,
    schedule_interval='0 12 * * *',  
    #schedule_interval=timedelta(seconds=10),
    catchup=False
) as dag:

    extract_task = PythonOperator(
        task_id='extract',
        python_callable=extract_data
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=predict_data
    )

    load_task = PythonOperator(
        task_id='load',
        python_callable=load_data
    )

    # Task untuk mengirim email
    email_task = PythonOperator(
        task_id='email_notification',
        python_callable=email_notification,
        provide_context=True,
        dag=dag,
    )
    
# urutan task
extract_task >> predict_task >> load_task >> email_task


CREATE TABLE transactions (
    "step" INT NOT NULL,
    "type" VARCHAR(20) NOT NULL,
    "amount" FLOAT NOT NULL,
    "nameOrig" VARCHAR(50) NOT NULL,
    "oldbalanceOrg" FLOAT NOT NULL,
    "newbalanceOrig" FLOAT NOT NULL,
    "nameDest" VARCHAR(50) NOT NULL,
    "oldbalanceDest" FLOAT NOT NULL,
    "newbalanceDest" FLOAT NOT NULL,
    "isFraud" INT NOT NULL,
    "isFlaggedFraud" INT NOT NULL
    );
	
ALTER TABLE transactions
ADD transaction_id SERIAL;

CREATE TABLE predictions_table (
    transaction_id INT PRIMARY KEY,  
    prediction INT
);

-- Ex Data
CREATE TABLE ex_transactions (
    "step" INT NOT NULL,
    "type" VARCHAR(20) NOT NULL,
    "amount" FLOAT NOT NULL,
    "nameOrig" VARCHAR(50) NOT NULL,
    "oldbalanceOrg" FLOAT NOT NULL,
    "newbalanceOrig" FLOAT NOT NULL,
    "nameDest" VARCHAR(50) NOT NULL,
    "oldbalanceDest" FLOAT NOT NULL,
    "newbalanceDest" FLOAT NOT NULL,
    );
	
ALTER TABLE ex_transactions
ADD transaction_id SERIAL;

CREATE TABLE ex_predictions_table (
    transaction_id INT PRIMARY KEY,  
    prediction INT
);
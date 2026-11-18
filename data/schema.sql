CREATE DATABASE IF NOT EXISTS fraud_detection;
USE fraud_detection;

CREATE TABLE IF NOT EXISTS users (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(150) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role ENUM('ADMIN','ANALYST') NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  txn_ref VARCHAR(32) NOT NULL UNIQUE,
  time_offset DOUBLE NOT NULL,
  amount DECIMAL(15,2) NOT NULL,
  actual_class TINYINT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_txn_ref (txn_ref)
);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  transaction_id BIGINT NOT NULL UNIQUE,
  fraud_probability DOUBLE NOT NULL,
  predicted_label VARCHAR(20) NOT NULL,
  threshold DOUBLE NOT NULL,
  model_version VARCHAR(50) NOT NULL,
  predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (transaction_id) REFERENCES transactions(id),
  INDEX idx_predicted_label (predicted_label)
);

CREATE TABLE IF NOT EXISTS alerts (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  transaction_id BIGINT NOT NULL UNIQUE,
  status ENUM('NEEDS_REVIEW','CONFIRMED_FRAUD','CLEARED') NOT NULL,
  reviewed_by BIGINT NULL,
  reviewed_at TIMESTAMP NULL,
  note TEXT NULL,
  FOREIGN KEY (transaction_id) REFERENCES transactions(id),
  FOREIGN KEY (reviewed_by) REFERENCES users(id),
  INDEX idx_alert_status (status)
);

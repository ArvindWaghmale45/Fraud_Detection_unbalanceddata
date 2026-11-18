# Banking Fraud Detection Using Big Data Analytics and Machine Learning: Full Project Plan
A phase-by-phase build plan written around the trained notebook results, made to be saved in the repo and followed with Antigravity, one phase at a time.

## 1. How to use this plan with Antigravity

1. Save this document as `docs/PROJECT_PLAN.md` in the project root (`Banking-Fraud-Detection/`).
2. Open the project folder in Antigravity.
3. Paste the master prompt below as your first message, then say: Start with Phase 1.
4. After each phase, check its 'Done when' list, then reply 'confirmed' to move on.
**Master prompt**

```
You are my senior mentor and pair-programmer for my B.Tech project 'Banking Fraud Detection Using Big Data Analytics and Machine Learning'.
First read docs/PROJECT_PLAN.md fully.
Rules:
1. Work on ONE phase at a time, starting with the phase I name.
2. For every step give a simple English explanation, exact file paths, terminal commands with where to run them, and the expected output.
3. Never skip ahead. When a phase is finished, show its 'Done when' checklist and WAIT until I say 'confirmed'.
4. Do not change the tech stack, folder structure or API contracts unless I approve.
5. The ML model is already trained. Do not retrain it. Only use the saved files in ml-service/models and ml-service/outputs.
6. I am a beginner in Big Data, Hadoop, Spark and ML, so explain those parts slowly.
7. My laptop has limited resources: single node only, no multi-node cluster.
```

## 2. Project summary
**Goal:** read bank card transactions and automatically flag the ones that are likely fraud, so an analyst can review them. Because fraud is very rare (about 0.17% of transactions), the system is judged on how many frauds it catches and how few false alarms it raises, not on accuracy.

**What is already finished:** data exploration, imbalance handling, training and comparison of three models, threshold tuning, and saving the final model and metrics (all in `fraud_detection.ipynb`).

**What is still to build:** the Big Data layer (HDFS and Spark), the prediction API (FastAPI), the backend (Spring Boot with JPA/Hibernate and MySQL), the secure login, and the React dashboard.

## 3. Results from the trained notebook
**Dataset (ULB credit card fraud, Kaggle):** 284,807 transactions, 31 columns (Time, V1 to V28, Amount, Class), no missing values. Genuine: 284,315. Fraud: 492 (0.1727%).

**Split:** 80% train and 20% test, stratified, random_state 42. Train: 227,845 rows. Test: 56,962 rows, of which 98 are fraud. Imbalance is handled with `class_weight='balanced'`.

**Model comparison on the test set**

- **Logistic Regression:** precision 0.0610, recall 0.9184, F1 0.1144, ROC-AUC 0.9722, PR-AUC 0.7159. It caught 90 frauds, missed 8, but raised 1,386 false alarms.
- **Decision Tree:** precision 0.0686, recall 0.8265, F1 0.1267, ROC-AUC 0.9075, PR-AUC 0.4748. It caught 81 frauds, missed 17, with 1,100 false alarms.
- **Random Forest (chosen):** accuracy 0.9994, precision 0.8265, recall 0.8265, F1 0.8265, ROC-AUC 0.9833, PR-AUC 0.8547. It caught 81 frauds, missed 17, with only 17 false alarms.
**Why Random Forest wins:** Logistic Regression has the highest recall, but a bank cannot review 1,386 false alarms to catch 9 more frauds. Random Forest keeps recall at 0.83 and cuts false alarms from 1,386 to 17. This precision versus recall trade-off is the main viva talking point.

**Threshold:** tuned on a separate validation set (45,569 rows) and found at 0.4954 (validation precision 0.8714, recall 0.7722, F1 0.8188). On the test set it gives the same result as the default 0.5.

**Top features (global importance):** V14 (0.1839), V10 (0.1184), V12 (0.1000), V17 (0.0947), V4 (0.0916). Amount has a small importance (0.0123).

**Saved files (inside `ml-service/`):**

- `models/random_forest_fraud_model.joblib`
- `outputs/model_metrics.json`
- `outputs/model_comparison.csv`
- `outputs/feature_importance.csv`
- `outputs/test_predictions.csv` (56,962 rows with Actual_Class, Fraud_Probability, Predicted_Class and Prediction_Status)

## 4. System architecture
Flow: React dashboard calls Spring Boot, Spring Boot calls the FastAPI ML service, the ML service loads the Random Forest model and returns a fraud probability, and Spring Boot saves the result in MySQL. Separately, the Big Data pipeline reads the raw dataset from HDFS, processes it with Spark and writes analytics that the dashboard can show.

- **React + Vite + Tailwind + Recharts (port 5173):** screens and charts.
- **Spring Boot (port 8080):** JWT security, business rules, database access with JPA/Hibernate, and the only service the frontend talks to.
- **FastAPI ML service (port 8000):** loads the model once at startup and exposes prediction endpoints.
- **MySQL (port 3306):** users, transactions, predictions, alerts, model metrics.
- **HDFS and Spark (single node):** stores the raw dataset and runs PySpark analytics jobs.

## 5. Folder structure

```
Banking-Fraud-Detection/
  frontend/        React + Vite + Tailwind
  backend/         Spring Boot (Maven)
  ml-service/
    app/           FastAPI code
    data/          creditcard_transactions.csv
    models/        random_forest_fraud_model.joblib
    outputs/       metrics, comparison, feature importance, test predictions
    notebooks/     fraud_detection.ipynb
  bigdata/
    hdfs/          upload and check scripts
    spark/         PySpark jobs
  data/            sample and seed files
  docs/            PROJECT_PLAN.md, report, PPT, screenshots
```

## 6. Data and database design
**Important facts about the data:**

- The dataset has no transaction ID or customer. The backend generates a `txn_ref` (for example TXN-000412).
- `Time` is seconds since the first transaction (about two days of data). Hour of day = (Time / 3600) % 24. Use this for the 'fraud by hour' chart.
- V1 to V28 are anonymized features, so the UI must never pretend to show card numbers or merchant names.
- Statuses: the notebook labels a predicted fraud as 'Suspicious'. Use these statuses: GENUINE, SUSPICIOUS, CONFIRMED_FRAUD, CLEARED.
**Demo data:** seed MySQL from `test_predictions.csv`. It already contains the model output for 56,962 transactions, so the dashboard looks real from day one. The 98 predicted-suspicious rows become the alerts queue.

**Tables (JPA entities):**

- `users`: id, name, email, password_hash, role (ADMIN or ANALYST), created_at.
- `transactions`: id, txn_ref, time_offset, amount, v1 to v28, actual_class (nullable, from the dataset), created_at.
- `predictions`: id, transaction_id (one-to-one), fraud_probability, predicted_label, threshold, model_version, predicted_at.
- `alerts`: id, transaction_id (one-to-one), status (NEEDS_REVIEW, CONFIRMED_FRAUD, CLEARED), reviewed_by (many-to-one users), reviewed_at, note.
- `model_metrics`: id, model_name, accuracy, precision_score, recall_score, f1_score, roc_auc, pr_auc, tp, fp, fn, tn, is_active. Load from `model_comparison.csv` and `model_metrics.json`.
- `audit_log`: id, user_id, action, details, created_at.
Use proper indexes on `txn_ref`, `predicted_label`, and `alerts.status`.

## 7. ML service (FastAPI) contract
**Before starting:** run `pip show scikit-learn` in the notebook environment and pin the same version in `ml-service/requirements.txt`. A joblib model can fail to load on a different scikit-learn version.

**Rules:** load the model once at startup, read the threshold from `model_metrics.json` (0.4954), keep the feature order exactly as in the notebook (Time, V1 to V28, Amount), and validate input with Pydantic.

**Endpoints:**

- `GET /health` returns service status and whether the model is loaded.
- `GET /model-info` returns the contents of `model_metrics.json`.
- `POST /predict` takes `{time, v1 ... v28, amount}` and returns `{fraud_probability, prediction, threshold}` where prediction is 'Suspicious' or 'Genuine'.
- `POST /predict-batch` takes a list of the same objects and returns a list of results.
- `GET /feature-importance` returns the top features from `feature_importance.csv`.

## 8. Backend (Spring Boot) design
**Packages:** config, security, controller, service, repository, entity, dto, exception, client (ML service client).

**Security:** Spring Security with JWT. Roles: ADMIN (everything, manages users) and ANALYST (view data, review alerts, check transactions). Passwords stored with BCrypt.

**Endpoints (all under `/api`):**

- `POST /auth/login` returns a JWT and the user role.
- `GET /dashboard/summary` returns totals, suspicious count, confirmed fraud count, review status counts and model KPIs.
- `GET /dashboard/trend` and `GET /dashboard/by-hour` return chart data.
- `GET /transactions` with page, size, status, minAmount, maxAmount, minProbability and search filters (server-side pagination).
- `GET /transactions/{id}` returns full details.
- `POST /predict` takes manual input, calls the ML service, saves transaction and prediction, and creates an alert if suspicious.
- `POST /transactions/batch` takes a CSV upload and predicts in batches.
- `GET /alerts?status=` and `PATCH /alerts/{id}` with `{decision: CONFIRMED_FRAUD or CLEARED, note}`.
- `GET /models/comparison`, `GET /models/active`, `GET /models/feature-importance`.
- `GET /reports/fraud.csv` and `GET /reports/fraud.pdf`.
**Good practice:** DTOs instead of exposing entities, a global exception handler, validation annotations, CORS for `http://localhost:5173`, ML service URL and JWT secret in `application.properties`, and lazy loading with fetch joins to avoid the N+1 query problem.

## 9. Big Data layer (HDFS and Spark)
Be honest in the viva: the dataset is only about 150 MB, so Spark is not strictly required for speed. It is used to show a real big-data pipeline that would scale to millions of bank transactions.

**HDFS:** single-node pseudo-distributed setup (on Windows prefer WSL2 or Docker). Upload the raw CSV to `/fraud/raw/` and write processed output to `/fraud/processed/`.

**PySpark jobs (in `bigdata/spark/`):**

1. `01_load_and_validate.py`: read the CSV from HDFS with a schema, check nulls and counts.
2. `02_class_analysis.py`: count genuine versus fraud and show the imbalance ratio.
3. `03_time_amount_aggregates.py`: fraud counts and amounts by hour of day, amount statistics per class.
4. `04_save_parquet.py`: write cleaned data as Parquet to `/fraud/processed/`.
5. Optional: `05_spark_mllib_rf.py`, a Spark MLlib Random Forest for comparison with the scikit-learn model.
Export the aggregate results to CSV or MySQL so the dashboard charts can use them.

## 10. Frontend and UI design spec
Goal: a clean, professional banking-analytics look, similar to the preview canvas (dark navy sidebar, light content area, blue accent, red only for fraud).

**Design tokens**

- Font: Inter (400, 500, 600, 700).
- Sidebar navy `#0F1B33`, page background `#F4F6FA`, card white with 1px border `#E4E7EC` and 12px radius.
- Primary blue `#2563EB`. Fraud red `#D92D20` (pill text `#B42318` on `#FEE4E2`). Genuine green `#067647` on `#D1FADF`. Review amber `#93370D` on `#FEF0C7`.
- Spacing on a 4px grid, soft shadows only on hover, 44px minimum height for buttons and inputs.
**Layout:** fixed 220px sidebar with the brand name, top bar with page title and user menu (name, role, logout), content max width about 1440px, responsive down to tablet width.

**Screens**

1. **Login:** split layout with a brand panel and a form, clear error messages, loading state on the button.
2. **Dashboard:** KPI cards (total transactions, flagged suspicious, confirmed fraud, alerts waiting for review); line chart of flagged transactions over time; bar chart of flagged transactions by hour of day; donut chart of alert review status; 'Model health' card (precision, recall, F1, ROC-AUC); table of latest flagged transactions with a link to the alert.
3. **Transactions:** server-side paginated table with status pills, probability bar, search and filters; click a row to open a side drawer with all details and a probability gauge.
4. **Fraud Alerts:** analyst work queue sorted by probability, with 'Confirm fraud' and 'Mark safe' buttons, an optional note, and a toast message after each action.
5. **Check Transaction:** form with Time and Amount, plus a collapsible 'Advanced features' section for V1 to V28. Include buttons 'Load sample fraud' and 'Load sample genuine' (taken from the test predictions) because users cannot invent V1 to V28. The result card shows a gauge, the probability and the verdict.
6. **Model Comparison:** grouped bar chart of precision, recall, F1, ROC-AUC and PR-AUC for the three models; confusion matrix of the active model; horizontal bar chart of the top 15 features; a short note on why Random Forest was chosen.
7. **Reports:** filter by date and status, export CSV or PDF.
**UX details that make it look professional:** skeleton loaders instead of spinners, empty states with a short message, consistent number formatting (284,807 and 0.17%), colour never the only signal (always a text label too), keyboard-friendly focus rings, and an optional 'Simulate live stream' button that replays test transactions every few seconds for demos.

## 11. The 14 phases
Each phase ends with a 'Done when' check. Do not start the next phase before it passes.

**Phase 1: Environment setup.** Install and verify JDK 17, Maven, Node 20 or newer, Python (same version as the notebook), MySQL 8, Git and VS Code. Create the folder structure from section 5. *Done when:* `java -version`, `mvn -v`, `node -v`, `python --version` and `mysql --version` all work and the folders exist.

**Phase 2: Dataset and EDA (already done in the notebook).** Copy the CSV into `ml-service/data/` and `data/`, save the notebook in `ml-service/notebooks/`, and write a one-page EDA summary in `docs/`. *Done when:* the files are in place and the summary is written.

**Phase 3: Trained model (already done).** Confirm all saved files exist and record the scikit-learn version. *Done when:* a 5-line Python script loads the joblib model and predicts one row from `test_predictions.csv` with the same probability.

**Phase 4: HDFS.** Install Hadoop in single-node mode (or WSL2 or Docker), start HDFS and upload the dataset to `/fraud/raw/`. *Done when:* `hdfs dfs -ls /fraud/raw` shows the file and the NameNode web page opens.

**Phase 5: Spark analytics.** Install Spark and run the PySpark jobs from section 9. *Done when:* the class counts match 284,315 and 492 and Parquet files exist in `/fraud/processed/`.

**Phase 6: FastAPI ML service.** Build the endpoints from section 7 with tests. *Done when:* `/docs` opens, and `/predict` returns the same probability as the notebook for a known row.

**Phase 7: Database and Spring Boot skeleton.** Create the MySQL schema, JPA/Hibernate entities and repositories, and the seed script that loads `test_predictions.csv`. *Done when:* the tables exist and contain 56,962 transactions and 98 alerts.

**Phase 8: Security.** Add Spring Security, JWT login and the ADMIN and ANALYST roles. *Done when:* login returns a token, protected endpoints reject missing tokens, and ANALYST cannot use admin endpoints.

**Phase 9: Business APIs.** Build the endpoints from section 8, including the ML service client, alerts workflow and reports. *Done when:* every endpoint works in Postman and a manual `/predict` creates a prediction and an alert.

**Phase 10: Frontend foundation.** Set up Vite, Tailwind, routing, Axios with a token interceptor, the design tokens, the layout and the Login page. *Done when:* login works against the backend and the empty layout matches the design spec.

**Phase 11: Dashboard, Transactions, Fraud Alerts.** Build screens 2 to 4 from section 10. *Done when:* charts show real backend data and an analyst can confirm or clear an alert.

**Phase 12: Check Transaction, Model Comparison, Reports.** Build screens 5 to 7. *Done when:* the sample fraud button returns 'Suspicious', the sample genuine button returns 'Genuine', and a report downloads.

**Phase 13: Integration and testing.** Run the full flow end to end, add unit and API tests, fix bugs, and polish the UI. *Done when:* the test checklist in section 12 passes.

**Phase 14: Documentation and demo.** Write the project report, README, architecture diagram, screenshots, PPT, and a 5-minute demo script. *Done when:* a fresh clone can be set up from the README.

## 12. Testing checklist

- ML service: `pytest` for `/predict` (valid input, missing field, wrong type) and for matching the notebook probability.
- Backend: JUnit and MockMvc for auth, role checks, pagination and the alert workflow; mock the ML client.
- Frontend: manual test of every screen on desktop and tablet widths, plus login failure and expired token.
- End to end: log in, open the dashboard, review an alert, check a sample transaction, export a report.
- Performance: batch predict 1,000 rows and note the time.

## 13. Demo script and viva questions
**5-minute demo:** show the imbalance problem (0.17% fraud), show the model comparison page, log in as analyst, open the dashboard, open a suspicious alert and confirm it, check a sample fraud transaction live, export the report, then show HDFS and Spark output.

**Likely viva questions**

- Why is accuracy a bad metric here? Because predicting everything as genuine gives 99.83% accuracy while catching zero frauds.
- How did you handle the imbalance? Class weights, stratified split, and precision, recall, F1 and PR-AUC as metrics.
- Why Random Forest over Logistic Regression? Similar recall zone, but 17 false alarms instead of 1,386.
- What does the threshold do? A lower threshold catches more frauds and raises more false alarms.
- Why Spark for 150 MB? To demonstrate a scalable pipeline; the same code runs on a cluster.

## 14. Honest limitations

- The model was trained and tested on one public dataset with a random split; a real bank would validate on later time periods.
- The features are anonymized, so per-transaction explanations are not possible; only global feature importance is shown.
- Random Forest missed 17 of 98 frauds in the test set (recall 0.83).
- Spark and HDFS run on a single node, so they show the pipeline, not real cluster scale.

## 15. Resume points (use only after the matching phase is done)

- Built a full-stack fraud detection platform with React, Spring Boot (JPA/Hibernate, Spring Security, JWT) and MySQL.
- Trained and compared Logistic Regression, Decision Tree and Random Forest on 284,807 transactions with 0.17% fraud; the chosen Random Forest reached 0.83 precision and recall (ROC-AUC 0.98, PR-AUC 0.85).
- Served the model through a FastAPI microservice and built an analyst dashboard with alert review workflow and CSV/PDF reports.
- Processed the dataset with a Hadoop HDFS and PySpark pipeline.
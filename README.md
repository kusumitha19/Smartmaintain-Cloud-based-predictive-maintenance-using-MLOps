\# SmartMaintain — Cloud-Based Predictive Maintenance Using MLOps



\## Overview



\*\*SmartMaintain\*\* is a cloud-based predictive maintenance platform that estimates the Remaining Useful Life (RUL) of industrial equipment using machine learning and deep learning. It uses sensor data from the NASA C-MAPSS FD001 dataset to predict how many operating cycles remain before an engine reaches its end of useful life.



The project integrates model development, experiment tracking, API deployment, containerization, and monitoring to demonstrate an end-to-end MLOps workflow.



\## Objectives



\- Estimate the Remaining Useful Life (RUL) of industrial engines from sensor data.

\- Process sequential sensor measurements using an LSTM-based regression model.

\- Compare multiple machine learning and deep learning models.

\- Track model evaluation experiments using MLflow.

\- Expose predictions through a FastAPI REST service.

\- Containerize the prediction service using Docker.

\- Deploy the service on AWS EC2.

\- Monitor API availability, request counts, and latency using Prometheus and Grafana.



\## Technology Stack



| Component | Technology |

|---|---|

| Programming language | Python |

| Dataset | NASA C-MAPSS FD001 |

| Data processing | Pandas, NumPy, Scikit-learn |

| Prediction model | LSTM regression |

| Model comparison | LSTM, GRU, CNN-LSTM, BiLSTM, Random Forest |

| Experiment tracking | MLflow |

| API | FastAPI |

| Containerization | Docker |

| Cloud deployment | AWS EC2 |

| Metrics collection | Prometheus |

| Monitoring dashboard | Grafana |



\## Dataset and Preprocessing



SmartMaintain uses the NASA C-MAPSS FD001 turbofan engine degradation dataset. The dataset contains operating-cycle information, operating settings, and sensor measurements.



The preprocessing pipeline includes:



1\. Loading the training, test, and RUL data files.

2\. Validating the expected columns and data types.

3\. Calculating training RUL labels from engine operating cycles and applying an RUL cap of 125 cycles.

4\. Scaling the sensor and operating-setting features using a StandardScaler fitted on training data.

5\. Sorting observations by engine ID and operating cycle.

6\. Creating sequential input windows of 30 cycles with 25 features per cycle.

7\. Preparing the final 30-cycle window for each test engine.



The resulting model input has the shape \*\*30 × 25\*\*.



The dataset files are not included in this repository. Place the authorized C-MAPSS FD001 files in `data/CMAPSS/` before running the full data-processing pipeline.



Expected files:



\- `train\_FD001.txt`

\- `test\_FD001.txt`

\- `RUL\_FD001.txt`



\## Model Development and Evaluation



The project evaluates multiple models using MAE, RMSE, and R². The following results are the existing evaluation results logged to MLflow.



| Model | MAE | RMSE | R² |

|---|---:|---:|---:|

| LSTM | 11.3076 | 14.7436 | 0.8741 |

| GRU | 11.7537 | 16.5598 | 0.8412 |

| CNN-LSTM | 12.2434 | 17.0503 | 0.8317 |

| Random Forest | 12.6016 | 16.5088 | 0.8303 |

| BiLSTM | 11.6802 | 13.5290 | 0.8860 |



Lower MAE and RMSE indicate smaller prediction errors, while higher R² indicates a better fit to the evaluation targets.



\*\*Note:\*\* BiLSTM has the highest R² and lowest RMSE in this comparison, while LSTM has the lowest MAE. The deployed API uses the saved LSTM model; the comparison results do not mean every model is deployed.



\## System Architecture



The system follows this workflow:



1\. \*\*Dataset ingestion:\*\* Load NASA C-MAPSS FD001 data.

2\. \*\*Preprocessing:\*\* Validate data, scale features, and create sequential windows.

3\. \*\*Model training and evaluation:\*\* Train and evaluate the RUL prediction models.

4\. \*\*Experiment tracking:\*\* Record model parameters and evaluation metrics in MLflow.

5\. \*\*Prediction API:\*\* Use FastAPI to expose the saved LSTM model through REST endpoints.

6\. \*\*Containerization:\*\* Package the prediction application and dependencies in Docker.

7\. \*\*Cloud deployment:\*\* Run the API on an AWS EC2 instance.

8\. \*\*Monitoring:\*\* Expose API metrics to Prometheus and visualize them in Grafana.



\## API Endpoints



The deployed FastAPI application provides these endpoints:



| Endpoint | Purpose |

|---|---|

| `/` | Web interface |

| `/health` | Application health check |

| `/docs` | Interactive API documentation |

| `/predict\_rul` | RUL prediction endpoint |

| `/predict\_engine\_rul` | Engine-based RUL prediction endpoint |

| `/metrics/` | Prometheus metrics endpoint |



Use `/docs` to inspect the request schemas and try the prediction endpoints.



\## MLOps and Monitoring



\### MLflow



MLflow records the existing evaluation metrics for five model experiments under `SmartMaintain-Model-Comparison`. Local tracking data is stored in `mlruns/` and is excluded from Git.



\### Prometheus



Prometheus scrapes the FastAPI application's `/metrics/` endpoint at 15-second intervals. The monitoring configuration uses `host.docker.internal:8080` to reach the API container host on the current EC2 deployment.



\### Grafana



Grafana visualizes the collected API metrics, including request activity and latency. Prometheus and Grafana are configured to bind to localhost on the EC2 host and should be accessed through SSH tunnels rather than exposed publicly.



\## Deployment



The current deployment uses AWS EC2, Docker, FastAPI, Prometheus, Grafana, and a separately launched MLflow UI.



The API is available on port `8080` on the EC2 instance. Monitoring dashboards are configured for private access.



For security, do not commit credentials, `.env` files, private keys, or other secrets. Restrict SSH access to trusted IP addresses and avoid exposing monitoring services directly to the public internet.



\### Start the prediction API



From the project directory on the EC2 instance, with Docker installed and the model artifacts available:



```bash

sudo docker build -t smartmaintain-api:latest .

```



If the dataset is required for engine-based prediction, mount it into the container:



```bash

sudo docker run -d \\

&#x20; --name smartmaintain-api \\

&#x20; --restart unless-stopped \\

&#x20; -p 8080:8080 \\

&#x20; -v "$PWD/data/CMAPSS:/app/data/CMAPSS:ro" \\

&#x20; smartmaintain-api:latest

```



Do not run a second container with the same name if `smartmaintain-api` is already running.



\### Monitoring



The monitoring services are defined in `compose.monitoring.yml`. Configure the required Grafana credentials in a local `.env` file before starting the stack:



```bash

sudo docker compose -f compose.monitoring.yml config -q

sudo docker compose -f compose.monitoring.yml up -d

```



Keep the `.env` file private and do not commit it.



\### MLflow UI



The experiment-tracking script is `track\_mlflow\_experiments.py`. The existing comparison metrics can be registered with MLflow using the project's configured Python environment:



```bash

python track\_mlflow\_experiments.py

```



The MLflow UI should be accessed privately through an SSH tunnel. Do not expose port 5000 publicly.



\## Repository Safety



The repository excludes local MLflow tracking data, raw NASA dataset files, environment files, generated artifacts, and local MLflow UI logs. Do not commit AWS private keys, tokens, passwords, or other secrets.



\## Current Status



The core LSTM prediction service is deployed on AWS EC2. Live predictions have been verified, five model-comparison runs are visible in MLflow, and Prometheus has successfully scraped the API metrics. Grafana has also been accessed for monitoring.



This repository documents the implemented academic project and its current deployment. Further production hardening, such as HTTPS, managed secrets, and automated recovery testing, may be required for a public production service.



\## Author



Developed as an M.Tech Computer Science and Engineering project.




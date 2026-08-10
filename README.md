# DDoS Attack Detection Using Machine Learning on Network Traffic

## 📌 Project Overview
DDoS (Distributed Denial of Service) attacks remain a very common and rapidly evolving cyber threat, fueled by automated tools and cheap DDoS-for-hire services that make them easy for non-technical actors to launch.[cite: 1] This project implements an anomaly-based Machine Learning (ML) Intrusion Detection System (IDS) to identify and classify DDoS traffic in real-time, moving beyond the limitations of legacy signature-based systems.

### The Problem with Legacy Systems
The global DDoS protection and mitigation market size is valued at $6.48 billion and is projected to skyrocket to over $16.66 billion. However, traditional defense mechanisms are failing due to several key vulnerabilities:
* Legacy systems struggle to tell the difference between a malicious DDoS attack and a sudden surge of legitimate viral customer traffic (like a Black Friday sale).
* This leads to catastrophic false positives.
* Modern botnets dynamically change their packet sizes, headers, and frequencies mid-attack.
* Static rules can't catch these variations, but ML anomaly detection can.
* With modern web attacks frequently executing in under 60 seconds to evade human response, systems need autonomous ML models that can detect anomalies and trigger defensive API blocks in milliseconds.

## 🚨 Current Threat Landscape
This model is designed with modern attack vectors in mind, addressing the following current trends:
* **Hyper-Volumetric Surges:** Attacks routinely reach multi-terabit levels per second, targeting network infrastructure with massive data floods.
* **Shorter, Faster Windows:** A significant share of high-impact web attacks now last under 60 seconds, executed programmatically to bypass human detection and manual response runbooks.
* **API and Application Targeting:** Attackers increasingly focus on Layer 7 (application layer) and APIs to disrupt business logic and customer transactions rather than just clogging raw bandwidth.
* **AI-Enhanced Execution:** Threat actors use automated AI tools to map networks and execute "horizontal" multi-destination attacks, hitting multiple weak points at once.

## 🎯 Target Industries and Use Cases
Financial services, telecommunications, gaming, and government sectors are the most heavily targeted industries for DDoS attacks.[cite: 1] Attackers focus on fields where even a few seconds of offline downtime causes massive financial loss, public panic, or strategic disruption.[cite: 1] This model is highly relevant for protecting:

1. **Telecommunications:** Protecting ISPs and DNS servers to prevent "blast radius" outages that knock out internet access for thousands of downstream businesses simultaneously.
2. **Finance & Banking:** Securing payment gateways from extortion and financial sabotage, where DDoS is often used as a distraction to hide background data-theft operations.
3. **Gaming:** Protecting multiplayer servers from sabotage and extortion right before major tournament events.
4. **Government:** Defending public portals and energy grids from geopolitical warfare and hacktivism.[cite: 1]
5. **Generative AI Services:** Preventing application-layer traffic from exhausting AI API resources and driving up massive cloud computing bills.

## 🧠 Technical Architecture & ML Strategy
Academics have plenty of accurate ML models, but the industry is starving for models that can actually run in production under heavy loads. Therefore, this project prioritizes **Feature Selection** and **Inference Time** rather than raw accuracy.

### 1. Data Processing
* **Dataset:** Utilizes standard academic datasets (e.g., CICDDoS2019/CICIDS2017) containing raw telemetry and flow data.
* **Feature Engineering:** Drops identifying features (Source IP, Timestamp) to prevent data leakage. Normalizes continuous variables (e.g., flow duration, packet length).

### 2. Model Selection (Speed vs. Weight)
* In cybersecurity, a model that has 98% accuracy and detects an attack in 0.01 seconds using minimal RAM is worth millions of dollars.
* Conversely, a heavy Deep Learning model that boasts 99.9% accuracy but takes 5 seconds to process traffic is a liability, because the server will crash before the model finishes its calculation.
* **Implementation:** We utilize lightweight edge models (such as optimized Random Forest or XGBoost) to parse stream data rapidly without crashing the routing hardware.

### 3. Evaluation Metrics
Models are evaluated not just on Accuracy, but on:
* **Precision & Recall** (crucial due to massive class imbalance in network traffic).
* **F1-Score & ROC-AUC.**
* **Inference Latency** (measured in milliseconds).

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Scikit-Learn, Pandas, NumPy, XGBoost
* Jupyter Notebook (for exploration)

### Installation
1. Clone the repository: `git clone https://github.com/yourusername/ddos-ml-detection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset and place it in the `/data` directory.

### Execution
Run the preprocessing script to clean the network flow data:
`python src/preprocess.py`

Train the model and evaluate inference times:
`python src/train.py`

## 📚 Future Research Opportunities
* **Lightweight Edge Models:** Build ultra-fast, optimized ML models (like pruned Random Forest or XGBoost) designed to run on low-resource edge routing devices or IoT gateways.
* **Behavioral API Log Analyzers:** Build a model that parses API/Web application logs in real-time, focusing on user behavior patterns rather than packet data.
* **Online/Incremental Learning:** Build an architecture that supports continuous, lightweight learning from new stream data without requiring full retraining from scratch.

---
*Developed as an academic graduation project focusing on applied machine learning in network security.*

# 🧠 Reading Your Mind: EEG Signal Classification

![Project Banner](images/banner_concept.png)

**Author:** Abdallah Khairy Werby (ID: 80708)  
**Supervisor:** Dr. Eslam Abd El-Azeem  
**Status:** Completed (BME 511 Final Project)

---

## 📖 The Story: Why This Matters
Imagine being able to move a robotic arm or type on a computer just by *thinking* about it. For people with severe motor disabilities (like ALS or spinal cord injuries), this isn't science fiction—it's a necessity.

This project, **"Reading Your Mind,"** explores the foundation of Brain-Computer Interfaces (BCIs). By analyzing raw EEG (Electroencephalography) signals—the electrical noise your brain makes—we can predict intended body functions before they physically happen.

**The Practical Goal:**
Translate a biological "thought" (electrical wave) into a digital "command" (0 or 1) that a machine can understand.

---

## ⚙️ Methodology: From Brain to Byte

The project processes raw EEG data collected from 4 different users (14 channels per user) through a rigorous machine learning pipeline.

![Methodology Pipeline](images/pipeline.png)
*Figure: Visual representation of the data processing steps.*

### 1. Data Preprocessing
* **Loading:** Raw `.mat` files containing 14-channel EEG recordings.
* **Cleaning:** Removal of invalid rows and artifacts.
* **Epoching:** Segmenting the continuous signal into 1-second "epochs" to isolate specific brain events.
* **Feature Extraction:** Extracting power bands:
    * **Delta (< 4Hz):** Deep sleep/unconscious.
    * **Theta (4-8Hz):** Drowsiness/meditation.
    * **Alpha (8-13Hz):** Relaxed alertness.
    * **Beta (> 13Hz):** Active thinking/focus.

### 2. Signal Processing
We applied **scaling and normalization** to ensure that high-amplitude artifacts do not skew the machine learning models.

![Scaled vs Unscaled Data](images/scaled_data.png)
*Figure: Comparison of EEG signals before and after normalization.*

### 3. Machine Learning Models
We implemented and compared six different supervised learning algorithms to classify the signals:
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machines (SVM)**
* **Decision Trees**
* **Logistic Regression**
* **Naïve Bayes**
* **Linear Discriminant Analysis (LDA)**

---

## 📊 Results

We tested the models on dataset split into Training (50%) and Testing (50%). The **Decision Tree** and **KNN** models proved to be the most effective for this type of biological data.

| Model | Accuracy | Sensitivity | Specificity |
|-------|----------|-------------|-------------|
| **Decision Tree** | **~70.8%** | High | Moderate |
| **KNN** | **~70.0%** | Moderate | High |
| Logistic Regression | ~60.5% | Low | Moderate |

![Performance Metrics Table](images/results_table.png)
*Figure: Detailed performance metrics output from MATLAB.*

> **Discussion:** Lower sampling rates (20Hz) reduced the fluctuation in accuracy compared to 160Hz, suggesting that for this specific task, reducing data complexity helped the models generalize better, despite the loss of some information.

---

## 💻 How to Run This Project

### Prerequisites
* MATLAB (R2020b or later recommended)
* Statistics and Machine Learning Toolbox

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/EEG-Signal-Classification.git](https://github.com/YourUsername/EEG-Signal-Classification.git)
    ```
2.  Ensure you have the data file `Users_Data.mat` in the main folder.

### Running the Classifier
1.  Open MATLAB.
2.  Navigate to the project folder.
3.  Open the script for a specific user, for example, `Classification_User_a.m`.
4.  **Important:** Ensure the data loading line is set to relative path:
    ```matlab
    % Change this line if it has a hardcoded C:\ path
    load('Users_Data.mat'); 
    ```
5.  Click **Run**. The script will output the accuracy tables and generate performance plots.

---

## 📸 Project Screenshots

### Classification Output
![Command Window Output](images/cmd_output.png)
*The final classification results appearing in the MATLAB Command Window.*

### Confusion Matrices & Plots
![KNN Model Plot](images/knn_plot.png)
*Visualization of the K-Nearest Neighbors classification boundary.*

---

## 🔗 Future Work
To improve accuracy beyond 70%, future iterations will explore:
1.  **Deep Learning:** Implementing CNNs (Convolutional Neural Networks) which are better suited for raw time-series data.
2.  **Advanced Feature Engineering:** Using Wavelet Transforms instead of simple frequency bands.
3.  **Real-time Processing:** connecting live EEG headsets (like Emotiv or OpenBCI) for real-time control.

---

**© 2025 Abdallah Khairy Werby**

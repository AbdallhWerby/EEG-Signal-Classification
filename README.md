# 🧠 Reading Your Mind: EEG Signal Classification

**Author:** Abdallah Khairy Werby (ID: 80708)  
**Supervisor:** Dr. Eslam Abd El-Azeem  
**Context:** BME 511 Final Project (Biomedical Engineering)

---

## 📖 The Story: Turning Noise into Commands
The human brain generates constant electrical activity, appearing as "noise" to the naked eye. This project asks: **Can a computer listen to this noise and predict what a human wants to do?**

This repository contains the MATLAB implementation of a **Brain-Computer Interface (BCI)** pipeline. We analyze EEG signals to classify intended states, creating a foundation for technology that assists individuals with motor disabilities (e.g., controlling a robotic arm via thought).

---

## ⚙️ Methodology

The project processes raw EEG data collected from 4 users (14 channels each) through the following pipeline:

1.  **Input:** Raw `.mat` EEG data (14 channels).
2.  **Preprocessing:**
    * **Cleaning:** Removal of artifacts and invalid rows.
    * **Epoching:** Segmenting continuous signals into 1-second events.
3.  **Feature Extraction:** Separating the signal into brain-wave frequency bands:
    * **Delta (< 4Hz):** Deep sleep/unconscious.
    * **Theta (4-8Hz):** Drowsiness/meditation.
    * **Alpha (8-13Hz):** Relaxed alertness.
    * **Beta (> 13Hz):** Active thinking/focus.
4.  **Classification:** Using Supervised Machine Learning to predict the state.

---

## 📊 Results & Performance

We tested six different models. The data suggests that **Decision Trees** and **KNN** are best suited for this specific type of non-linear biological data.

| Model | Accuracy | Sensitivity | Specificity | Performance Note |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | **~70.8%** | **High** | **Moderate** | Best overall performance. |
| **KNN (K-Nearest)** | **~70.0%** | **Moderate** | **High** | Very consistent results. |
| Logistic Regression | ~60.5% | Low | Moderate | struggled with non-linear data. |
| Naïve Bayes | ~58.0% | Low | Low | Assumed independence too strongly. |

### Key Finding on Sampling Rates
We compared high sampling rates (160Hz) vs. down-sampled rates (20Hz).
* **160Hz:** Higher accuracy potential (up to 83%) but highly unstable (fluctuated down to 16%).
* **20Hz:** More stable, consistent accuracy (50-70%).
* **Conclusion:** Lowering the sampling rate acted as a noise filter, helping the models generalize better despite the loss of some information.

---

## 💻 How to Run

### Prerequisites
* MATLAB (R2020b or later recommended).
* Statistics and Machine Learning Toolbox.

### Steps
1.  **Download** the repository to your local machine.
2.  **Locate** the `Users_Data.mat` file in the main folder.
3.  **Open** the script for a specific user (e.g., `Classification_User_a.m`).
4.  **Important:** Ensure the code looks for the data in the current folder, not a specific C: drive path:
    ```matlab
    load('Users_Data.mat'); % Correct
    % load('C:\subs\Users_Data.mat'); % Incorrect (remove this if seen)
    ```
5.  **Run** the script. The Command Window will output the accuracy matrices for all tested models.

---

## 🔗 Future Scope
To push accuracy beyond 70% and make this viable for real-world medical devices, future work will focus on:
1.  **Deep Learning:** Replacing manual feature extraction with Convolutional Neural Networks (CNNs).
2.  **Real-Time Processing:** Interfacing with live EEG headsets (e.g., Emotiv/OpenBCI) rather than pre-recorded datasets.

---
*© 2025 Abdallah Khairy Werby

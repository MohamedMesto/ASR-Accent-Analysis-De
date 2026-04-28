📌 This repository is part of a larger ASR system developed during my Master's thesis.
# ASR Accent Analysis (German) – Speech AI Evaluation System

This project focuses on analyzing and evaluating **Automatic Speech Recognition (ASR)** performance across different German accents. It was developed as part of my Master’s research at Technische Universität Berlin in collaboration with the German Research Center for Artificial Intelligence (DFKI).

The goal is to build **data-driven evaluation pipelines** that identify performance gaps, bias, and robustness issues in speech recognition systems — with applications in **real-world communication and healthcare contexts**.

---

## 🚀 Project Overview

This repository provides tools and workflows for:

* Analyzing ASR model performance across different accents
* Evaluating transcription quality using **Word Error Rate (WER)** and duration-based metrics
* Exploring bias and variability in speech recognition systems
* Processing and querying speech datasets for targeted analysis

---

## 🧠 System Design

The project follows a modular evaluation pipeline:

1. **Input Data**

   * Speech audio files with associated metadata (accent, speaker, duration)

2. **Processing**

   * Transcription via ASR models (e.g., DeepSpeech)
   * Alignment of predicted vs reference text

3. **Evaluation**

   * WER calculation
   * Duration-based analysis
   * Accent-specific performance comparison

4. **Analysis & Insights**

   * Identification of error patterns
   * Bias detection across accents
   * Data-driven insights for improving model robustness

---

## 📊 Key Features

### 🔹 Accent-based Performance Analysis

Evaluate how ASR models perform across different German accents and identify inconsistencies.

### 🔹 Word / Utterance Search

Locate specific words or phrases across audio datasets and identify their associated accents.

👉 Notebook:
[Word_utterance_audiofiles.ipynb](https://github.com/MohamedMesto/ASR-Accent-Analysis-De/blob/main/Word_utterance_audiofiles.ipynb)

---

### 🔹 Experimental Analysis Framework

Central experimentation and evaluation workflows.

👉 Notebook:
[AttributionAnalysis_De.ipynb](https://github.com/MohamedMesto/ASR-Accent-Analysis-De/blob/main/AttributionAnalysis_De.ipynb)

---

## 🏥 Relevance to Healthcare AI

Speech recognition systems play a key role in:

* clinical documentation
* patient-provider communication
* accessibility for non-native speakers

This project contributes to:

* improving **robustness and inclusivity** of AI systems
* reducing bias in real-world communication scenarios
* enabling more reliable AI-driven healthcare solutions

---

## ⚙️ Technologies

* Python
* NumPy, Pandas
* Speech processing & ASR (DeepSpeech)
* Jupyter Notebooks

---

## 📂 Usage

> Note: Files with `_NB` are intended to be run locally in a Jupyter Notebook environment.

1. Clone the repository
2. Install required dependencies
3. Run notebooks for analysis and experimentation

---

## 🔗 Related Work

* Master Thesis Repository:
  👉 [https://github.com/MohamedMesto/MasterThesis-QU-DFKI-Accented-Speech-Recognition-ASR](https://github.com/MohamedMesto/MasterThesis-QU-DFKI-Accented-Speech-Recognition-ASR)

---

## 📌 Key Takeaways

* Built **evaluation pipelines for ASR systems**
* Applied **data-driven analysis to detect bias and performance gaps**
* Designed workflows that support **real-world AI system improvement**

## 📂 Repository Structure & Key Components

### 🔹 Core Analysis Modules

- **Duration & WER Analysis**
  - Focus: Evaluate ASR performance using Word Error Rate and duration metrics  
  👉 https://github.com/MohamedMesto/ASR-Accent-Analysis-De/tree/main/Duration-WER-Analysis  

- **DeepSpeech Data Processing**
  - Focus: Data preparation and integration with DeepSpeech-based ASR pipelines  
  👉 https://github.com/MohamedMesto/ASR-Accent-Analysis-De/tree/main/DeepSpeech/data  

---

### 🔹 Notebooks

- **Word / Utterance Search**
  - Locate specific words across audio files and identify accent distribution  
  👉 https://github.com/MohamedMesto/ASR-Accent-Analysis-De/blob/main/Word_utterance_audiofiles.ipynb  

- **Attribution & Experimental Analysis**
  - Main experimentation workflows for ASR evaluation  
  👉 https://github.com/MohamedMesto/ASR-Accent-Analysis-De/blob/main/AttributionAnalysis_De.ipynb  

---

### 🔹 Full Thesis Repository

- Complete implementation, datasets, and experiments  
  👉 https://github.com/MohamedMesto/MasterThesis-QU-DFKI-Accented-Speech-Recognition-ASR  

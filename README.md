# ccm-analysis-tool

# 🌐 CCM Analysis Tool

An interactive Streamlit web app for Convergent Cross Mapping (CCM) analysis of time series data.


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/chaseU2/ccm-analysis-tool.git
   cd ccm-analysis-tool

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:
   ```bash
   streamlit run app.py
   ```

## 🌐 Live Demo
This is a Streamlit app running on [Streamlit Cloud](https://ccmwebtool.streamlit.app/).
[Click here to view the app!](https://ccmwebtool.streamlit.app/)


## Test Data

This project uses test data from the file [`ccm_test_data.txt`](./ccm_test_data.txt) for testing the implementation of the Convergent Cross Mapping (CCM) algorithm.

### User-Provided Data

If you want to use your own data for testing or analysis, please ensure that your data file is in the **same format** as [`ccm_test_data.txt`](./ccm_test_data.txt). Importantly, **all missing values (NAs)** in the dataset **must be replaced with zeros** to ensure proper functionality of the algorithm.

You can upload your data file in the same format, with columns and rows matching the original data structure.






## Dependencies and Acknowledgements

Parts of this project are based on the Convergent Cross Mapping (CCM) implementation from the following repository from Prince Javier :

- [Convergent Cross Mapping GitHub Repository Prince Javier ](https://github.com/PrinceJavier/causal_ccm.git)

I have utilized parts of the CCM algorithm from this repository to help analyze causality in time series data in my own project.

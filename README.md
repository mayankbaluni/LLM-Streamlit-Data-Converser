
# LLM-Streamlit-Data-Converser

An interactive tool leveraging LangChain for dynamic question-answering from CSV datasets, featuring a Streamlit interface for ease of use and accessibility.

## Overview
This repository hosts the code for a question-answering system that utilizes large language models (LLMs) to provide answers based on the uploaded CSV data. The system integrates LangChain to leverage the power of LLMs and Streamlit for a user-friendly interface, allowing users to upload data and ask questions dynamically.

## Features
- **CSV File Upload**: Users can upload their own CSV files to be processed.
- **Data Inquiry**: After uploading, users can ask natural language questions regarding their data.
- **LLM Integration**: The system utilizes powerful language models to generate accurate and relevant answers.
- **Streamlit Interface**: A Streamlit-based web interface provides an easy-to-use platform for all interactions.

## Getting Started

### Prerequisites
Before running the application, ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
git clone https://github.com/mayankbaluni/LLM-Streamlit-Data-Converser.git
cd LLM-Streamlit-Data-Converser
streamlit run stream-app.py

# End of script
exit 0
```

## How to Use
To interact with the question-answering system, follow these simple steps:
1. **Navigate to the hosted Streamlit app**: Access the application through your web browser.
2. **Upload CSV Data**: Utilize the file uploader within the app to upload your dataset in CSV format.
3. **Ask Your Question**: Input your question regarding the data in the text field provided.
4. **Receive the Answer**: Submit your question and the system will utilize the underlying LLM to generate an answer based on your data.

## Built With
- **LangChain**: A powerful framework used to integrate Large Language Models (LLMs) for advanced data processing and answering capabilities.
- **Streamlit**: An open-source app framework that is the cornerstone of our interactive web interface, simplifying the deployment of data applications.
- **Pandas**: An essential data analysis and manipulation library for Python, utilized here for efficient handling of CSV files and dataset operations.


## Contact
For any queries or suggestions, feel free to contact me at [mayankbaluni@gmail.com]

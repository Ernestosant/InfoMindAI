# InfoMindAI

InfoMindAI is an application designed to interact with and extract knowledge from various types of documents such as PDFs, CSVs, and Excel files. Built on the power of GPT-3.5-turbo and the Streamlit framework, it provides a user-friendly interface for document summaries, direct chat interactions, and querying.

## Features

1. **Summarize PDFs**: Summarize content from multiple PDFs and download the summarized content as a Word document.
2. **Chat with PDFs**: Pose questions to the system and get answers based on the content of uploaded PDFs.
3. **Chat with CSV and Excel**: Interact with data from CSV and Excel files, ask questions, and get insights directly from the dataset.

## Prerequisites
1. streamlit
2. langchain
3. openai
4. pypdf
5. llama-index
6. pandas
7. nltk
8. python-dotenv
## Installation and Setup

1. Clone the repository.
2. Install required Python libraries using `pip install -r requirements.txt`.
3. Export your OpenAI API Key to the environment or directly input it in the sidebar of the Streamlit app.
4. Run the application using `streamlit run your_filename.py`.

## How to use

1. **Summarize PDFs**:
    - Upload the desired PDF files.
    - Use the default summary prompt or provide a custom one.
    - Click on "Generate summaries" to get a summarized version of the documents. 

2. **Chat with PDFs**:
    - Upload the desired PDF files.
    - Input your question into the text area.
    - View the system's response based on the content of the uploaded documents.

3. **Chat with CSV and Excel**:
    - Upload a CSV or Excel file.
    - Pose your query related to the dataset in the text area.
    - Receive an answer based on the data.

## Contributors

- [Your Name](Your GitHub Link) - Initial work

## License

This project is licensed under the MIT License.

## Acknowledgements

- OpenAI for their robust GPT-3.5-turbo model.
- Streamlit community for their continuous support and awesome framework.


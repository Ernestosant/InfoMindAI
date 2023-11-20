import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import  PromptTemplate
import glob
import os
from docx import Document
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
import pandas as pd
import openai
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import shutil

# Limpiando archivos temporales
try:
    shutil.rmtree('pdfs')
    shutil.rmtree('data_files')
except:
    with st.sidebar:
        st.write("There is no documents")


#***********************Functions***************************************************************************************
# Save pdf uploaded
def save_uploaded_pdfs(upload_files):
    # Crear el directorio 'pdfs' si no existe
    if not os.path.exists('pdfs'):
        os.makedirs('pdfs')

    # Guardar cada archivo en el directorio 'pdfs'
    for uploaded_file in upload_files:
        with open(os.path.join('pdfs', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getvalue())

def save_excel_uploaded(xlsx_file):
    # Make  path if not exist
    if not os.path.exists('data_files'):
        os.makedirs('data_files')

    with open(os.path.join('data_files', xlsx_file.name), 'wb') as f:
            f.write(xlsx_file.getvalue())

# Custom summary function
def custom_summary(pdf_folder, custom_prompt):
    summaries = []
    for pdf_file in glob.glob(pdf_folder + "/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        prompt_template = custom_prompt + """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce",
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        summary_output = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
        summaries.append(summary_output)

    return summaries

DEFAULT_SUMMARIZE_PROMPT = """Act as an expert researcher in writing scientific bibliographic review articles. You know perfectly how to summarize a scientific article. You never omit the numerical values of the performance metrics or experimental measurements that are present in the article to be summarized. Your summaries are short and to the point.
0. Title: Your summary should to begin with the article's title
1. Introduction and context: Begin by briefly summarizing the introduction to the article and the context in which the research was conducted. This will help contextualize the study and provide a framework for further discussion.
2. Methods: Describe the methods used in the study, including the participants, procedures, and instruments used. If there are any important limitations in the methods, such as a small sample size or lack of randomization, be sure to mention them.
3. Results: Summarize the main findings of the study. Be sure to include key statistics and any other significant results found.
4. Discussion: Analyze the results of the study in the context of the research question. How do the results relate to the existing literature? Are there any important implications for future practice or research? Be sure to discuss any limitations of the study and how they might be addressed in future research.
5. Conclusions: Summarize the main conclusions of the study and their importance. You can also mention any important limitations of the study and suggest areas for future research.
6. Citations: You should to include the citation's article in IEEE format"""


def summaries_to_word_download(content):
    # Crear un nuevo documento
    doc = Document()
    doc.add_paragraph(content)
    
    # Guardar el documento en un archivo temporal
    filename = "temp.docx"
    doc.save(filename)
    
    # Permitir la descarga del archivo a través de Streamlit
    with open(filename, "rb") as file:
        btn = col3.download_button(
            label="Download",
            data=file,
            file_name="documento.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    
    # Eliminar el archivo temporal
    os.remove(filename)


#***********************Interface**************************************************************************************
st.set_page_config(page_title="InfoMindAI", page_icon='project_logo.jpg',
                    layout='centered', initial_sidebar_state='auto')

st.title("InfoMindAI")


tab_titles = ['Sumarize PDFs','Chat with PDFs', 'Chat with CSV and Excel']

tabs = st.tabs(tab_titles)

with st.sidebar:
    apikey = st.text_input(label='OpenAI API_KEY', type= 'password')    
    os.environ["OPENAI_API_KEY"] =apikey
    openai.api_key = apikey
    if apikey:
        llm = ChatOpenAI(temperature=0.2, model = 'gpt-3.5-turbo')
    
    upload_files = st.file_uploader(label='Upload pdf files', type= ['pdf'], accept_multiple_files=True)
    save_uploaded_pdfs(upload_files)


    upload_excell_files = st.file_uploader(label='Upload excel file', type=['xlsx', 'csv'])
    try:
        save_excel_uploaded(upload_excell_files)
    except:
        st.write('There is not a data file')


# sesions states trackers

# state for uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = False

def chancge_uploaded_files_state():
    if not(st.session_state.uploaded_files):
        st.session_state.uploaded_files = True


#-------------------------------------------- Tab for summarize multiple pdfs------------------------------------------------------------------
with tabs[0]:

    col1, col2, col3 = st.columns(3)
    summaries = "" 
    custom_sumary = col2.checkbox(label='custom summary', value=0)
    # Make a custom summarize for each file and saving in a .txt file
    if custom_sumary:
        summarize_prompt = st.text_area(label='Custom summary prompt')
    else:
        summarize_prompt = DEFAULT_SUMMARIZE_PROMPT    

    get_summaries = col1.button(label='Generate summaries')
    

    if get_summaries:
        summaries_list = custom_summary("pdfs", custom_prompt=summarize_prompt)
        summaries = '\n\n'.join([summary for summary in summaries_list])     
                     
        
    # Save all summaries into one .word file  
    
    summaries_to_word_download(summaries)

    st.write(summaries)

#--------------------------------------------Tab for query  with multiples pdfs---------------------------------------------------------------
with tabs[1]:


    if 'embedings' not in st.session_state:
        st.session_state['embedings'] = False

    if upload_files:
        #Get text from pdfs
        pdf = SimpleDirectoryReader('pdfs').load_data()
        #Select the LLm model
        model = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'))

        # Create Vector database
        service_context = ServiceContext.from_defaults(llm_predictor=model)
        index = GPTVectorStoreIndex.from_documents(pdf, service_context = service_context)
        query_engine = index.as_query_engine()
        st.session_state['embedings'] = True    

    question = st.text_area("Question")
    if len(os.listdir('pdfs'))>0 and apikey:
        if question:
            response = query_engine.query(question)
            st.write(response.response)
      



#------------------------------------------- Chat with excel ---------------------------------------------------------------------------------------

with tabs[2]:
    if os.path.exists('data_files') and len(os.listdir('data_files'))>0:      
        
        # get the file's path
       
        data_files_path = os.listdir('data_files')[-1]
        # print('data files path:', data_files_path)
        if 'xlsx' in data_files_path:
            df = pd.read_excel('data_files/'+ data_files_path)
        st.dataframe(df)
        query = st.text_area(label='Query')
        st.markdown("## Answers")
        csv_name = data_files_path.split('/')[-1].rstrip('.xlsx')+'.csv'
        df.to_csv('data_files/'+csv_name)
        try:
            # create a csv agent
            agent = create_csv_agent(
                ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                path='data_files/'+csv_name,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
            )
            if query:
                st.write(agent.run(query))
        except:
            st.write('Set Api_key')




    
    
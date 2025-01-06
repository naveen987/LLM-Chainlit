# SRH-CHATBOT-V3

SRH-CHATBOT-V3 is an advanced chatbot system designed to streamline communication within university environments. It leverages cutting-edge technologies like Meta Llama2 and Pinecone to provide efficient and precise responses. This chatbot aims to enhance administrative efficiency, improve operational workflows, and foster interactive learning for students and faculty. By making information more accessible, SRH-CHATBOT-V3 transforms the academic support landscape within universities.

## Features
- **AI-Powered Conversations**: Utilizes Meta Llama2 for natural language understanding and response generation.
- **Vector Search**: Powered by Pinecone for fast and accurate document retrieval.
- **Interactive Learning**: Facilitates student-faculty interaction through a responsive chat interface.
- **Customizable UI**: Supports HTML and CSS-based UI customization.
- **Scalable Infrastructure**: Built for deployment across diverse university-related applications.

## Getting Started

### Prerequisites
To set up and run SRH-CHATBOT-V3, ensure you have the following:

- **Anaconda**: [Download and install](https://www.anaconda.com/products/individual).
- **Pinecone Account**: [Sign up here](https://www.pinecone.io/).
- **Google Cloud Storage Bucket**: Create a bucket in your Google Cloud account.
- **Bing Search API Key**: Follow [these steps](https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/) to obtain your API key.

### Setup Instructions

#### 1. Clone the Repository
Clone the SRH-CHATBOT-V3 repository to your local machine:
```bash
git clone https://github.com/naveen987/LLM-Chainlit.git
```

#### 2. Create a Conda Environment
Navigate to the project directory and create a new Conda environment:
```bash
conda create -n schatbot python=3.8 -y
conda activate schatbot
```

#### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
Create a `.env` file in the root directory with the following details:
```ini
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_API_ENV=your_pinecone_environment_here
BING_SEARCH_API_KEY=your_bing_search_api_key_here
```

#### 5. Download the Model
Download the Llama 2 Model (llama-2-13b-chat.ggmlv3.q4_0.bin) from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML) and place it in the `model` directory.

#### 6. Prepare the Data Index
Set up a Pinecone index named `srh-heidelberg-docs` with the following:
- **Metric**: Cosine
- **Dimensions**: 768

Prepare and upload document vectors:
- For local PDF files:
  ```bash
  python store_index.py
  ```
- For Google Cloud Storage PDFs:
  Update `BUCKET_NAME` in `gcs_store_index.py` and run:
  ```bash
  python gcs_store_index.py
  ```

## Running the Application

### Streamlit Interface
Run the chatbot using Streamlit:
```bash
streamlit run appstreamlit.py
```
Access it at: `http://localhost:8501/`

### Flask Interface
For a customizable UI, use Flask:
- Edit `templates/chat.html` for the layout.
- Modify `static/style.css` for design changes.

Run the Flask app:
```bash
python app.py
```
Access it at: `http://localhost:8080/`

## Tech Stack
- **Python**: Core programming language.
- **LangChain**: For building language model applications.
- **Flask**: Backend web framework.
- **Meta Llama2**: AI model for NLP.
- **Pinecone**: Vector database for similarity search.
- **Streamlit**: Framework for building interactive apps.
- **Google Cloud Storage**: For storing PDFs.
- **Bing Search API**: For web search capabilities.

## Acknowledgments
Special thanks to SRH Heidelberg for supporting the development of this project.


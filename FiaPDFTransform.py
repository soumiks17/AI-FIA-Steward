import os
import json
import pdfplumber
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def extract_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Extraction error on {pdf_path}: {e}")
    return text

def parse_document(text, chain):
    try:
        response = chain.invoke({"text": text})
        clean_json = response.content.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"LLM Parsing error: {e}")
        return None

def process_directory(input_dir, output_file):
    print(f"Checking directory: {os.path.abspath(input_dir)}")
    if not os.path.exists(input_dir):
        print("CRITICAL FAILURE: The input directory does not exist.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("CRITICAL FAILURE: OPENAI_API_KEY is not loaded into the environment.")
        return
    print("API Key found in environment.")

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    template = """
    Extract the following from the FIA Steward Decision document.
    Return ONLY a valid JSON object with these exact keys: "driver", "breach", "decision", "reasoning".
    If a field is missing, use "N/A".
    
    Document:
    {text}
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = prompt | llm
    
    file_count = 0
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith('.pdf'):
                    file_count += 1
                    filepath = os.path.join(root, filename)
                    print(f"Reading: {filepath}")
                    
                    raw_text = extract_text(filepath)
                    if not raw_text.strip():
                        print(f"Warning: Extracted text is empty for {filename}")
                        continue
                        
                    parsed_data = parse_document(raw_text, chain)
                    if parsed_data:
                        parsed_data['source_file'] = filepath
                        f.write(json.dumps(parsed_data) + '\n')
                        print(f"Successfully transformed: {filename}")
                    else:
                        print(f"Failed to transform: {filename}")
                        
    print(f"Execution complete. Total PDFs processed: {file_count}")

if __name__ == "__main__":
    process_directory("./fia_pdfs", "structured_precedents.jsonl")
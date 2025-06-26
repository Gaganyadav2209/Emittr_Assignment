# Emittr_Assignment
**Live link: (Deployed on HuggingFace Spaces)** https://huggingface.co/spaces/Gagan2209/Emitrr   (Wait for 5-10 second to app to load)

<ins>**Overview**</ins>
- I have developed an application that performs NER to extract [Symptoms, Treatment, Diagnosis, Prognosis] using Spacy.
- For Sentiment and Intent Detection I have used a zero-shot Classifier because I did not had any custome data available to fine-tuning BERT, I have used valhalla/distilbart-mnli-12-1 for sentiment and intent classification
- For SOAP Notes Generation, honestly I took help of AI tools for it since there were not much resource available on Internet, However I found out a research paper https://aclanthology.org/2021.acl-long.384.pdf  (Generating SOAP Notes from Doctor-Patient Conversations Using Modular Summarization Techniques), I think this clearly outlines what we are trying to solve.

<ins>**Task Specific Description**</ins>

**1.**  **Named Entity Recognition**
  - For performing NER I adopted a hybrid approach that combines both domain-specific models and custom rule-based logic to     maximize entity coverage.
  - en_core_sci_lg  and en_ner_bc5cdr_md (Both models are available in Scispacy)
  - By combining these two models, I ensure broader entity recognition across medical conditions, symptoms, drugs, and treatments — overcoming the limitations of any single model.
  - To robustly map extracted spans to dictionary entries, I used RapidFuzz for token-based fuzzy string matching, this allowed me soft-matching .


**2.** **Sentiment and Intent detetion**
- For detecting both sentiment and intent in patient transcript, I used a zero shot classifier valhalla/distilbart-mnli-12-1.
- I used this model so that we can classify text into custom labels without needing task-specific training data.
- I have assigned the target class on the basis of maximum probability.
- I designed two independent sets of labels [Sentiment, Intent]
  


**3.** **SOAP Notes Generation**
<ins>I have tried this approach however it is not that much robust and accurate, and honestly  I took chatgpt for it</ins>
- however  I found a research paper for the same https://aclanthology.org/2021.acl-long.384.pdf (Generating SOAP Notes from Doctor-Patient Conversations Using Modular Summarization Techniques) this paper clearly solve our problem.

  

<ins>**Instruction to setup the project**</ins>
step 1: create a virtual environment (Python version - 3.10)
step 2: Clone the Repository -> git clone https://github.com/Gaganyadav2209/Emittr_Assignment.git
step 3: pip install --upgrade pip
step 4: pip install -r requirements.txt


I have used python version **3.10** across my project.
1) I have provided 3 python notebooks -> Final_NER.ipynb (for named entity extraction), Intent_sentiment.ipynb (For classifying intent and sentiment of patient transcript), SOAP_implementation(to get SOAP notes of the transcription)
2) I have attached a requirements.txt file with it, which includes dependencies required to run the project.
3) Simply create a **virtual environment** and install all the dependencies inside of it. (pip install -r requirements.txt)
4) I have also provided an app.py file which is a gradio file for my project using this you can see demo for project, after installing all the dependencies just run this file.

<ins>**Output screenshots**</ins>
1) **NER**
   <img width="1440" alt="app2" src="https://github.com/user-attachments/assets/1b1499c1-f05d-486f-9cf1-402a4b1c7a52" />
<img width="1440" alt="app1" src="https://github.com/user-attachments/assets/a28eab1c-7230-4723-ac40-b9cec322f716" />

2) **Sentiment and Intent Detection**
   <img width="1440" alt="app5" src="https://github.com/user-attachments/assets/cdd63abb-32f3-43f7-97dd-a2fbef27004e" />
   <img width="1440" alt="app6" src="https://github.com/user-attachments/assets/140b8a2a-b901-4571-8bf1-a65989c62215" />

3) **SOAP Notes**
   <img width="1440" alt="app4" src="https://github.com/user-attachments/assets/eb412e9c-c8ed-47a4-8e16-08043b5f5364" />
   <img width="1440" alt="app3" src="https://github.com/user-attachments/assets/3ec56942-07d4-4333-9cd4-8f800c3ad573" />


<ins>**Questions asked in assignment**</ins>
1) How would you handle ambiguous or missing medical data in the transcript?
   - For this type of problem I came up with some approaches ->
   a) if any of the fields are not extracted then we can rather have a placeholder like "Not Specified" as it increases    transparency and reduces chances of any error as our use   case is  sensitive.
  b) Introducing Contextual aware entity recognition, this will help us build coreference between surrounding entitites.

2) What pre-trained NLP models would you use for medical summarization?
   - For medical summarization I tried out different models such as BioMed-T5 and facebook/bart-large-cnn, the results were great using these models.

3) How would you fine-tune BERT for medical sentiment detection?
   -  I would start with a base model such as Bio_ClinicalBERT, then I would collect and prepare my sentiment data with the target classes as "anxious", "neutral" and "reassured" , after this step I would fine tune my BERT model.

4) What datasets would you use for training a healthcare-specific sentiment model?
   -  Honestly I Googled to find out these and got some datasets such as SentiHealth, MIMIC-III/IV, DAIC-WOZ, I would also suggest to have custom data collected.

   


# Emittr_Assignment

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

  



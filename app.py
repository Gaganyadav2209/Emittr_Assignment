import re
import gradio as gr
from transformers import pipeline
from functools import lru_cache
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@lru_cache(maxsize=3)
def load_spacy_model(name):
    return spacy.load(name)

@lru_cache(maxsize=3)
def load_nli_pipeline():
    return pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1",
        device=0
    )

@lru_cache(maxsize=3)
def load_summarizer():
    
    model_name = "t5-small"  # lightweight alternative
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return tokenizer, model


SYMPTOM_DICT = {
    "headache", "migraine", "throbbing head pain", "sharp pain", "dull ache", "burning sensation", "stabbing pain",
    "nausea", "vomiting", "queasiness", "upset stomach",
    "dizziness", "lightheadedness", "vertigo", "spinning sensation",
    "chest pain", "pressure in chest", "tightness", "heartburn",
    "shortness of breath", "dyspnea", "breathlessness", "wheezing", "chest tightness", "difficulty breathing",
    "fatigue", "tiredness", "lack of energy", "exhaustion",
    "fever", "high temperature", "chills", "night sweats", "hot flashes", "cold sweats",
    "cough", "persistent cough", "dry cough", "productive cough", "hacking cough",
    "sore throat", "throat pain", "scratchy throat", "hoarseness",
    "runny nose", "nasal discharge", "stuffy nose", "congestion",
    "back pain", "lumbar pain", "lower back ache", "upper back pain",
    "neck pain", "cervical pain", "stiff neck",
    "abdominal pain", "stomach ache", "tummy ache", "cramping", "bloating", "gas",
    "muscle ache", "myalgia", "joint pain", "arthralgia",
    "rash", "skin eruption", "hives", "redness", "itching", "pruritus",
    "swelling", "edema", "inflammation", "puffiness",
    "tingling", "paresthesia", "pins and needles", "numbness",
    "bleeding", "hemorrhage", "nosebleed", "bruising",
    "insomnia", "sleep disturbance", "difficulty sleeping", "restlessness",
    "palpitations", "racing heart", "irregular heartbeat",
    "diarrhea", "loose stools", "constipation", "hard stools",
    "blurred vision", "double vision", "eye pain", "visual disturbance",
    "earache", "ringing in ears", "tinnitus", "hearing loss",
    "urinary frequency", "painful urination", "dysuria", "blood in urine",
    "weight loss", "weight gain", "loss of appetite", "increased appetite",
    "confusion", "memory loss", "difficulty concentrating", "brain fog"
}


TREATMENT_DICT = {
    "rest", "bed rest", "take it easy",
    "ice pack", "cold compress", "heat pack", "warm compress",
    "physical therapy", "physiotherapy sessions", "rehab", "exercise program",
    "surgery", "operative repair", "procedure", "surgical intervention", "minimally invasive surgery",
    "antibiotics", "amoxicillin", "doxycycline", "penicillin",
    "painkillers", "analgesics", "ibuprofen", "acetaminophen", "paracetamol", "naproxen",
    "steroids", "corticosteroids", "prednisone", "dexamethasone",
    "insulin therapy", "oral hypoglycemics", "metformin", "glipizide",
    "antihypertensives", "lisinopril", "amlodipine", "losartan",
    "chemotherapy", "radiation therapy", "immunotherapy",
    "inhaler", "bronchodilator", "albuterol", "salmeterol",
    "antidepressants", "SSRIs", "sertraline", "fluoxetine", "venlafaxine",
    "anticoagulants", "warfarin", "heparin", "rivaroxaban",
    "dietary changes", "low-salt diet", "gluten-free diet", "low-carb diet", "Mediterranean diet",
    "vaccination", "immunization", "flu shot", "tetanus shot",
    "wound dressing", "bandaging", "stitches", "sutures",
    "counseling", "psychotherapy", "cognitive behavioral therapy", "CBT",
    "hydration", "fluid intake", "electrolyte replacement", "IV fluids",
    "elevating the limb", "compression stockings", "massage",
    "applying ointment", "topical creams", "steroid creams", "antibiotic ointment",
    "breathing exercises", "pulmonary rehabilitation", "oxygen therapy",
    "diet modification", "exercise regimen", "weight management", "lifestyle changes",
    "mindfulness", "relaxation techniques", "meditation",
    "laparoscopic surgery", "endoscopy", "colonoscopy"
}


DIAGNOSIS_DICT = {
    "hypertension", "high blood pressure",
    "diabetes mellitus", "type 2 diabetes", "type II diabetes", "sugar diabetes",
    "myocardial infarction", "heart attack",
    "stroke", "cerebrovascular accident", "brain attack",
    "whiplash injury", "cervical strain", "neck sprain",
    "concussion", "mild traumatic brain injury", "head injury",
    "pneumonia", "lung infection", "chest infection",
    "urinary tract infection", "UTI", "bladder infection",
    "appendicitis", "inflammation of the appendix",
    "fracture", "broken bone", "hairline fracture",
    "sprain", "ligament tear", "twisted ankle",
    "gastroenteritis", "stomach flu", "food poisoning",
    "anemia", "low hemoglobin", "iron deficiency",
    "migraine disorder", "recurrent headaches",
    "osteoarthritis", "degenerative joint disease", "wear and tear arthritis",
    "depression", "major depressive disorder", "clinical depression",
    "anxiety disorder", "generalized anxiety", "panic disorder",
    "asthma", "reactive airway disease",
    "chronic obstructive pulmonary disease", "COPD", "emphysema",
    "bronchitis", "chest cold",
    "sinusitis", "sinus infection",
    "allergic rhinitis", "hay fever",
    "otitis media", "middle ear infection",
    "gastroesophageal reflux disease", "GERD", "acid reflux",
    "peptic ulcer disease", "stomach ulcer",
    "irritable bowel syndrome", "IBS",
    "kidney stones", "renal calculi",
    "prostatitis", "prostate infection",
    "arthritis", "rheumatoid arthritis", "gout",
    "osteoporosis", "brittle bones",
    "thyroid disorder", "hypothyroidism", "hyperthyroidism",
    "bipolar disorder", "manic depression",
    "schizophrenia", "psychotic disorder",
    "post-traumatic stress disorder", "PTSD",
    "skin cancer", "melanoma", "basal cell carcinoma",
    "hepatitis", "liver inflammation",
    "cirrhosis", "liver scarring",
    "fatty liver disease", "hepatic steatosis",
    "multiple sclerosis", "MS",
    "Parkinson's disease", "tremor disorder",
    "Alzheimer's disease", "dementia"
}


PROGNOSIS_DICT = {
    "full recovery expected", "complete recovery likely", "should make a full recovery",
    "partial recovery", "some residual symptoms possible", "may not fully resolve",
    "guarded prognosis", "uncertain outcome", "wait and see",
    "good prognosis", "favorable outcome", "positive outlook",
    "poor prognosis", "unfavorable outcome", "serious condition",
    "recovery expected within six months", "reconstructive healing time", "recovery in 2-3 weeks", "healing time of 4-6 weeks",
    "likely recurrence", "risk of relapse", "may come back",
    "chronic condition", "long-term management needed", "ongoing treatment required",
    "acute condition", "short-lived course", "temporary issue",
    "stable", "condition stable", "no change expected",
    "progressive", "worsening over time", "may deteriorate",
    "in remission", "disease in remission", "currently under control",
    "monitor regularly", "follow-up recommended", "keep an eye on it",
    "palliative care", "supportive management", "focus on comfort",
    "physiotherapy", "rehabilitation program", "exercise therapy",
    "expected to improve", "likely to get better", "should see improvement",
    "may require ongoing treatment", "long-term follow-up needed", "continued care necessary",
    "risk of complications", "potential for recurrence", "may need further intervention",
    "stable condition", "no immediate concerns", "holding steady",
    "full recovery anticipated", "partial recovery possible", "residual symptoms likely",
    "prognosis is excellent", "very good outlook", "should do well",
    "recovery timeline", "healing process", "expected duration",
    "short-term issue", "long-term condition", "permanent damage",
    "no long-term impact", "should not affect daily life", "minimal lasting effects",
    "10 weeks", "weeks", "months", "days", "years"
}




SENTIMENT_LABELS = [
    "The speaker shows signs of anxiety or worry",
    "The speaker shows a neutral or calm demeanor",
    "The speaker shows signs of reassurance, relief, or comfort"
]
INTENT_LABELS = [
    "The speaker is asking for reassurance or comfort",
    "The speaker is describing their symptoms or condition",
    "The speaker is voicing worry about a situation"
]
SENTIMENT_MAP = {
    SENTIMENT_LABELS[0]: "Anxious",
    SENTIMENT_LABELS[1]: "Neutral",
    SENTIMENT_LABELS[2]: "Reassured"
}
INTENT_MAP = {
    INTENT_LABELS[0]: "Seeking reassurance",
    INTENT_LABELS[1]: "Reporting symptoms",
    INTENT_LABELS[2]: "Expressing concern"
}


def extract_patient_name(text: str) -> str:
    nlp = load_spacy_model("en_core_web_sm")
    for line in text.splitlines()[:6]:
        for ent in nlp(line).ents:
            if ent.label_ == "PERSON":
                return ent.text
    return "Not Specified"

def clean_transcript(text: str) -> str:
    return re.sub(r"(?:Physician|Doctor|Patient):", "", text).strip()

def extract_medical_spans(text: str):
    nlp1 = load_spacy_model("en_core_sci_lg")
    nlp2 = load_spacy_model("en_ner_bc5cdr_md")
    spans = {ent.text for ent in nlp1(text).ents} | {ent.text for ent in nlp2(text).ents}
    return list(spans)

def map_to_bucket(span: str):
    from rapidfuzz import process, fuzz
    buckets = {
        "Symptoms": SYMPTOM_DICT,
        "Treatment": TREATMENT_DICT,
        "Diagnosis": DIAGNOSIS_DICT,
        "Prognosis": PROGNOSIS_DICT,
    }
    best, best_score, best_match = None, 0, None
    for name, terms in buckets.items():
        match, score, _ = process.extractOne(span, terms, scorer=fuzz.token_sort_ratio)
        if score > best_score:
            best, best_score, best_match = name, score, match
    if best_score >= 70:
        return best, best_match
    return None, None

def ner_pipeline(text: str):
    cleaned = clean_transcript(text)
    name = extract_patient_name(text)
    spans = extract_medical_spans(cleaned)
    results = {
        "Patient_Name": name,
        "Symptoms": set(),
        "Treatment": set(),
        "Diagnosis": set(),
        "Prognosis": set()
    }
    for span in spans:
        bucket, match = map_to_bucket(span)
        if bucket:
            results[bucket].add(match)

    for k in ["Symptoms","Treatment","Diagnosis","Prognosis"]:
        results[k] = list(results[k]) or ["Not Specified"]
    return results


def sentiment_intent_pipeline(text: str):
    nli = load_nli_pipeline()
    s = nli(text, candidate_labels=SENTIMENT_LABELS)
    i = nli(text, candidate_labels=INTENT_LABELS)
    return {
        "Sentiment": SENTIMENT_MAP[s["labels"][0]],
        "Sentiment_Confidence": round(s["scores"][0], 2),
        "Intent": INTENT_MAP[i["labels"][0]],
        "Intent_Confidence": round(i["scores"][0], 2)
    }


def summarize_text(text: str, min_length=15, max_length=100):
    if not text.strip():
        return "No information provided."
    tok, mdl = load_summarizer()
    inputs = tok([text], truncation=True, padding="longest", return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inputs, min_length=min_length, max_length=max_length, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)

def classify_utterance_by_rules(utt: str) -> str:
    loi = utt.lower()

    subj_kws = ["i feel","pain","discomfort","ache","nausea","dizzy","worried"]
    if any(kw in loi for kw in subj_kws) and loi.startswith("patient"):
        return "Subjective"

    obj_kws = ["exam","observed","range of motion","vitals","lab","imaging"]
    if any(kw in loi for kw in obj_kws) and loi.startswith("physician"):
        return "Objective"

    plan_kws = ["recommend","continue","follow up","prescribe","return"]
    if any(kw in loi for kw in plan_kws) and loi.startswith("physician"):
        return "Plan"
    return "Other"

def soap_pipeline(text: str):

    classified = {"Subjective": [], "Objective": [], "Plan": []}
    for line in text.splitlines():
        cat = classify_utterance_by_rules(line)
        if cat in classified:
            classified[cat].append(re.sub(r'^(Physician|Patient):\s*','', line))
    subj = " ".join(classified["Subjective"])
    obj  = " ".join(classified["Objective"])
    plan = " ".join(classified["Plan"])
    return {
        "Subjective": summarize_text(subj,   max_length=120),
        "Objective":  summarize_text(obj,    max_length= 80),
        "Plan":       summarize_text(plan,   max_length=100),
        "Assessment": summarize_text(
            f"Subjective: {summarize_text(subj)} "
            f"Objective: {summarize_text(obj)} "
            f"Plan: {summarize_text(plan)}",
            max_length= 60
        )
    }


INSTRUCTIONS = {
    "NER" : """
        **NOTE -> First Inference Could take upto 1 minute due to loading of heavy dependencies. (Further Inferences will be quick) I have used LRU Cache**
        
        **Paste the transcript in the below format:**
        Physician: Good morning, Ms. Jones. How are you feeling today?  
        Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.  
        Physician: I understand you were in a car accident last September. Can you walk me through what happened?  
        Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.  
        Physician: That sounds like a strong impact. Were you wearing your seatbelt?  
        .............
        
    """,
    "Sentiment & Intent": """
        **NOTE -> First Inference Could take upto 1 minute due to loading of heavy dependencies. (Further Inferences will be quick) I have used LRU Cache**
        **Paste the patient transcript in the below format:**
            
        Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future?  
    """,

    "SOAP Generation" : """
        **NOTE -> First Inference Could take upto 1 minute due to loading of heavy dependencies. (Further Inferences will be quick) I have used LRU Cache**
        
        SOAP generation is currently not fully accurate. 
        I’ve found a research paper for the same : https://aclanthology.org/2021.acl-long.384.pdf  :- 
        Generating SOAP Notes from Doctor-Patient Conversations Using Modular Summarization Techniques
    """
    
}

with gr.Blocks() as demo:
    gr.Markdown("Physician Notetaker")

    with gr.Row():
        task = gr.Radio(
            choices=["NER", "Sentiment & Intent", "SOAP Generation"],
            label="Select Task",
            value="NER"
        )

    instructions = gr.Markdown(INSTRUCTIONS["NER"])  # default

    inp = gr.Textbox(
        lines=8,
        placeholder="Paste your transcript here...",
        label="Transcript Input"
    )

    btn = gr.Button("Run")
    out = gr.JSON()

    def router(task, text):
        if not text.strip():
            return {"error": "Please provide input text."}
        if task == "NER":
            return ner_pipeline(text)
        if task == "Sentiment & Intent":
            return sentiment_intent_pipeline(text)
        if task == "SOAP Generation":
            return soap_pipeline(text)

    def update_task(selected_task):
        return gr.update(value="", visible=True), gr.update(value=INSTRUCTIONS[selected_task])

    task.change(update_task, inputs=task, outputs=[inp, instructions])
    btn.click(router, inputs=[task, inp], outputs=out)

if __name__ == "__main__":
    demo.launch(share=True)

import time
from langchain.chains import RetrievalQA
from nltk.translate.bleu_score import sentence_bleu
from src.data_loader import load_contracts
from src.pipeline import create_pipeline

# Load contracts and create pipeline
contracts = load_contracts("data/contracts")
qa_pipeline = create_pipeline(contracts)

# Define a list of test questions and their expected answers
test_questions = [
    {"query": "Who are the parties to the Agreement and what are their defined names?", "expected": "Cloud Investments Ltd. (“Company”) and Jack Robinson (“Advisor”)"},
    {"query": "What is the termination notice?", "expected": "According to section 4:14 days for convenience by both parties. The Company may terminate without notice if the Advisor refuses or cannot perform the Services or is in breach of any provision of this Agreement."},
    {"query": "What are the payments to the Advisor under the Agreement?", "expected": "According to section 6: 1. Fees of $9 per hour up to a monthly limit of $1,500, 2. Workspace expense of $100 per month, 3. Other reasonable and actual expenses if approved by the company in writing and in advance."},
    {"query": "Can the Agreement or any of its obligations be assigned?", "expected": "Under section 1.1 the Advisor can’t assign any of his obligations without the prior written consent of the Company, 2. Under section 9 the Advisor may not assign the Agreement and the Company may assign it, 3. Under section 9 of the Undertaking the Company may assign the Undertaking."},
    {"query": "Who owns the IP?", "expected": "According to section 4 of the Undertaking (Appendix A), Any Work Product, upon creation, shall be fully and exclusively owned by the Company."},
    {"query": "Is there a non-compete obligation to the Advisor?", "expected": "Yes. During the term of engagement with the Company and for a period of 12 months thereafter."},
    {"query": "Can the Advisor charge for meal time?", "expected": "No. See Section 6.1, Billable Hour doesn’t include meals or travel time."},
    {"query": "In which street does the Advisor live?", "expected": "1 Rabin st, Tel Aviv, Israel"},
    {"query": "Is the Advisor entitled to social benefits?", "expected": "No. According to section 8 of the Agreement, the Advisor is an independent consultant and shall not be entitled to any overtime pay, insurance, paid vacation, severance payments or similar fringe or employment benefits from the Company."},
    {"query": "What happens if the Advisor claims compensation based on employment relationship with the Company?", "expected": "If the Advisor is determined to be an employee of the Company by a governmental authority, payments to the Advisor will be retroactively reduced so that 60% constitutes salary payments and 40% constitutes payment for statutory rights and benefits. The Company may offset any amounts due to the Advisor from any amounts payable under the Agreement. The Advisor must indemnify the Company for any losses or expenses incurred if an employer/employee relationship is determined to exist."},
]

# Initialize evaluation metrics
total_accuracy = 0
total_response_time = 0
total_relevance = 0
total_consistency = 0

# Evaluation Loop
results = []

for i, test in enumerate(test_questions):
    question = test["query"]
    expected_answer = test["expected"]
    
    start_time = time.time()
    result = qa_pipeline({"query": question})
    end_time = time.time()
    
    response_time = end_time - start_time
    response = result["result"]
    
    # Calculate BLEU score for accuracy
    accuracy = sentence_bleu([expected_answer.split()], response.split())
    relevance = accuracy  
    consistency = accuracy
    
    total_accuracy += accuracy
    total_response_time += response_time
    total_relevance += relevance
    total_consistency += consistency
    
    results.append({
        "question": question,
        "response": response,
        "accuracy": accuracy,
        "relevance": relevance,
        "response_time": response_time,
        "consistency": consistency,
    })

# Calculate averages
avg_accuracy = total_accuracy / len(test_questions)
avg_response_time = total_response_time / len(test_questions)
avg_relevance = total_relevance / len(test_questions)
avg_consistency = total_consistency / len(test_questions)

# Print results
print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Response Time: {avg_response_time}")
print(f"Average Relevance: {avg_relevance}")
print(f"Average Consistency: {avg_consistency}")

for result in results:
    print(result)

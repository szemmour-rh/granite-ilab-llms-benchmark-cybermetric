import json
import re
import time
import requests
from tqdm import tqdm

class CyberMetricEvaluator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Your local ilab/vLLM endpoint
        self.url = "http://127.0.0.1:8000/v1/chat/completions"

    def read_json_file(self):
        with open(self.file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def extract_answer(text: str):
        m = re.search(r"ANSWER:?\s*([A-D])", text, re.IGNORECASE)
        return m.group(1).upper() if m else None

    def ask_llm(self, question, answers, max_retries=3):
        opts = ', '.join(f"{k}) {v}" for k,v in answers.items())
        prompt = (
            f"Question: {question}\n"
            f"Options: {opts}\n\n"
            "Choose the correct answer only in this format: ANSWER: X"
        )
        payload = {
            "model": "/var/home/instruct/.cache/instructlab/models/instructlab/merlinite-7b-lab",
            "messages": [
                {"role": "system", "content": "You are a security expert who answers questions."},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.0
        }
        headers = {"Content-Type": "application/json"}

        for attempt in range(max_retries):
            try:
                r = requests.post(self.url, headers=headers, json=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                ans = self.extract_answer(text)
                if ans:
                    return ans
                # if format was wrong, retry once more
            except Exception as e:
                wait = 2 ** attempt
                print(f"[Error] {e!r}, retrying in {wait}s…")
                time.sleep(wait)
        return None

    def run_evaluation(self):
        questions = self.read_json_file()["questions"]
        correct = 0
        wrongs = []

        with tqdm(total=len(questions), desc="Evaluating") as bar:
            for qobj in questions:
                q, opts, sol = qobj["question"], qobj["answers"], qobj["solution"]
                got = self.ask_llm(q, opts)
                if got == sol:
                    correct += 1
                else:
                    wrongs.append((q, sol, got))
                pct = correct / (bar.n + 1) * 100
                bar.set_postfix_str(f"Acc: {pct:.2f}%")
                bar.update(1)

        print(f"\n✅ Final Accuracy: {correct/len(questions)*100:.2f}%")
        if wrongs:
            print("\nIncorrect answers:")
            for q, sol, got in wrongs:
                print(f"Q: {q}\n → Expected {sol}, got {got}\n")

if __name__ == "__main__":
    FILES = [
        "CyberMetric-80-v1.json",
        "CyberMetric-500-v1.json",
        "CyberMetric-2000-v1.json",
        "CyberMetric-10000-v1.json"
    ]

    for file_path in FILES:
        print(f"\n🧪 Evaluating file: {file_path}")
        evaluator = CyberMetricEvaluator(file_path=file_path)
        evaluator.run_evaluation()

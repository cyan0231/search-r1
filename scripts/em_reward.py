import re
import string

def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # The agent ends at the first valid answer or at a generation/search limit.
    model_text = re.sub(r"<information>.*?</information>", "", solution_str, flags=re.DOTALL)
    match = re.search(r"<answer>(.*?)</answer>\s*\Z", model_text, re.DOTALL)
    pred = match.group(1).strip() if match else ""
    return float(bool(pred) and any(normalize(pred) == normalize(x) for x in ground_truth["target"]))

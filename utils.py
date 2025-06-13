### Import Libraries
import re

### Code Formatting
def remove_diffs(code_snippet):
    diff_removed = ""
    count = 1 # Give Line Numbers to Assist in Understanding
    for line in code_snippet.split("\n"):
        diff_removed += str(count) + " " + line[1:] + "\n"
        count+=1 
    return diff_removed[:-1]

### Prompts
acr_prompt = """
### The following {lang} code snippet has received a code review.
[{lang}]
{code_snippet}
[/{lang}]
[CODE REVIEW]
{code_review}
[/CODE REVIEW]
### Please generate a revised version of the code snippet according to the code review. Do not add explanations.
[{lang}]
"""

ct_formatter = {
    "add_only" : "only add new lines of code",
    "remove_only" : "only delete existing lines of code",
    "modify" : "modify the code"
}

ctr_prompt = """
### The following is a multiple choice question (with answers) that tests code review comprehension. 
Question: Given this {lang} code snippet, what type of change is the code review asking for?
[{lang}]
{code_snippet}
[/{lang}]
[CODE REVIEW]
{code_review}
[/CODE REVIEW]
### Possible answers:
A. {option_a}
B. {option_b}
C. {option_c}
### Answer with the letter symbol only. Answer:
"""

### ACR Evaluators
def remove_comments(code):
    pattern = r'/\*.*?\*/|//.*?$'
    tmp_code = re.sub(pattern, '', code, flags=re.DOTALL|re.MULTILINE)
    pattern = r'(?m)^\s*#.*?$'
    return re.sub(pattern, '', tmp_code)

def get_em_trim(gold, pred):
    gold_lines = gold.split("\n")
    pred_lines = pred.split("\n")
    jumps = [0]
    for line in pred_lines:
        jumps.append(len(line)+jumps[-1])
    gold_words = []
    pred_words = []
    for line in gold_lines:
        gold_words.extend(line.split())
    for line in pred_lines:
        pred_words.extend(line.split())
    em_trim = 0
    if len(pred_words) >= len(gold_words):
        for jump in jumps:
            if jump+len(gold_words) > len(pred_words):
                break
            if pred_words[jump:jump+len(gold_words)] == gold_words:
                em_trim = 1
                break

    return em_trim


def get_em_no_space(gold, pred):
    gold_lines = gold.split("\n")
    pred_lines = pred.split("\n")
    gold_line_no_space = [re.sub(r'\s', '', line) for line in gold_lines]
    pred_line_no_space = [re.sub(r'\s', '', line) for line in pred_lines]
    jumps = [0]
    for line in pred_line_no_space:
        jumps.append(len(line)+jumps[-1])
    gold_string_no_space = "".join(gold_line_no_space)
    pred_string_no_space = "".join(pred_line_no_space)
    em_no_space = 0
    if len(pred_string_no_space) >= len(gold_string_no_space):
        for jump in jumps:
            if jump+len(gold_string_no_space) > len(pred_string_no_space):
                break
            if pred_string_no_space[jump:jump+len(gold_string_no_space)] == gold_string_no_space:
                em_no_space = 1
                break
    return em_no_space


def get_em_no_comment(gold, pred):
    gold_no_comment = remove_comments(gold)
    pred_no_comment = remove_comments(pred)
    return get_em_no_space(gold_no_comment, pred_no_comment)


def get_em(gold, pred):
    gold_lines = gold.split("\n")
    pred_lines = pred.split("\n")
    gold_words = []
    pred_words = []
    for line in gold_lines:
        gold_words.extend(line.split())
    for line in pred_lines:
        pred_words.extend(line.split())
    em = 0
    if pred_words == gold_words:
        em = 1
    return em

def jaccard_similarity(linesA, linesB):
    A = set()
    for line in linesA.split("\n"):
        A.update(line.split())
        A.update(re.findall(r'[a-zA-Z]+', line))
    B = set()
    for line in linesB.split("\n"):
        B.update(line.split())
        B.update(re.findall(r'[a-zA-Z]+', line))
    if len(A.union(B)) == 0:
        return 0
    else:
        return len(A.intersection(B)) / len(A.union(B))

def myeval(gold, pred):
    em = get_em(gold, pred)
    em_trim = get_em_trim(gold, pred)
    em_no_space = get_em_no_space(gold, pred)
    em_no_comment = get_em_no_comment(gold, pred)
    return em, em_trim, em_no_space, em_no_comment

### MCQA Evaluators
def count_matching_elements(arr1, arr2):
    return sum(1 for a, b in zip(arr1, arr2) if a == b)

def calc_results(results):
    correct_responses = []
    for row in range(len(results)):
        x = results.iloc[row]
        correct_responses.append(count_matching_elements(x.model_answers, x.correct_answers))
        
    print("Invariant Accuracy (Out of 100): ", len(results.loc[results['model_answers'] == results['correct_answers']]))
    print("Avg Correct Answers (Out of 6):", sum(correct_responses)/len(correct_responses))
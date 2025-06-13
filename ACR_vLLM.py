### Import Libraries
import gc
import sys
import torch
import warnings
import pandas as pd
from tqdm import tqdm
from transformers import logging
from huggingface_hub import login
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import acr_prompt, remove_diffs, myeval
from vllm.distributed.parallel_state import destroy_model_parallel

### Silent Logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

### Prompt Constructor
def test_prompt(test_set, language_type):
    test_prompts = []
    
    for row in tqdm(range(len(test_set))):
        example = test_set.iloc[row]
        prompt = acr_prompt.format(lang = language_type, 
                               code_snippet = remove_diffs(example.old),
                               code_review = example.review) 
        test_prompts.append(prompt)
    return test_prompts

### Evaluation
def save_eval(gold, output):
    generated = "\n".join([line[2:] for line in output.text.split("\n")])
    result = myeval(gold, generated)
    record = [generated] + list(result)
    return pd.DataFrame([record], columns = ['generation', 'em', 'em_trim', 'em_no_space', 'em_no_comment'])

### Run Test
def main():
    login(token = sys.argv[1])
    language_type = sys.argv[2]
    model_name = sys.argv[3]
    mcqa_set = pd.read_json("hf://datasets/Tomo-Melb/CodeReviewQA/CodeReviewQA.jsonl", lines = True)

    # Ingest Data 
    if language_type == "C":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "c"]
        save_dir = 'results/acr/c/acr_c_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "CPP":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "cpp"]
        save_dir = 'results/acr/cpp/acr_cpp_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "CSharp":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "csharp"]
        save_dir = 'results/acr/csharp/acr_csharp_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "Go":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "go"]
        save_dir = 'results/acr/go/acr_go_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "Java":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "java"]
        save_dir = 'results/acr/java/acr_java_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "JavaScript":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "javascript"]
        save_dir = 'results/acr/javascript/acr_javascript_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "PHP":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "php"]
        save_dir = 'results/acr/php/acr_php_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "Python":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "python"]
        save_dir = 'results/acr/python/acr_python_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "Ruby":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "ruby"]
        save_dir = 'results/acr/ruby/acr_ruby_' + model_name.split("/")[1] + '.pkl'
    
    # Import Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature = 0,
                                    max_tokens = 512,
                                    stop = ["[/{lang}]".format(lang = language_type)])                                
    llm = LLM(model = model_name, tensor_parallel_size = torch.cuda.device_count(), max_model_len = 4000)
    
    # Run Inference
    test_prompts = test_prompt(mcqa_set, language_type)
    outputs = llm.generate(test_prompts, sampling_params)
    
    # Save Results
    c_save = pd.DataFrame(columns = ['generation', 'em', 'em_trim', 'em_no_space', 'em_no_comment'])
    for row in tqdm(range(len(outputs))):
        gold = "\n".join([line[1:] for line in mcqa_set.iloc[row].new.split("\n")])
        c_save = pd.concat([c_save, save_eval(gold, outputs[row].outputs[0])])

    c_save.to_pickle(save_dir)
    
    # Output Results
    print("EM_TRIM: ", c_save.em.sum())
    print("EM_NO_SPACE: ", c_save.em_no_space.sum())
    print("EM__NO_COMMENT: ", c_save.em_no_comment.sum())
    
    # Release Cache
    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    gc.collect()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()
    
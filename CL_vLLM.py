### Import Libraries
import gc
import sys
import torch
import warnings
import itertools
import pandas as pd
from tqdm import tqdm
from transformers import logging
from huggingface_hub import login
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.sampling_params import GuidedDecodingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from utils import cl_prompt, ct_formatter, remove_diffs, count_matching_elements, calc_results

### Silent Logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

### Prompt Constructor
def prompt_combinations(example, mode, language_type):
    symbol_index_map = {0:'A', 1:'B', 2:'C', 3:'D'}
    
    if mode == "easy":
        options = example.loc_wrong_easy.copy()
    elif mode == "hard":
        options = example.loc_wrong_hard.copy()
        
    options.append(example.loc_correct)
    all_permutations = list(itertools.permutations(options))
    
    prompts = []
    correct_symbols = []
    for permutation in all_permutations:
        correct_symbols.append(symbol_index_map[permutation.index(example.loc_correct)])
        prompts.append(cl_prompt.format(lang=language_type, 
                                 option_a = ",".join(str(x) for x in permutation[0]),
                                 option_b = ",".join(str(x) for x in permutation[1]),
                                 option_c = ",".join(str(x) for x in permutation[2]),
                                 option_d = ",".join(str(x) for x in permutation[3]),
                                 code_snippet = remove_diffs(example.old),
                                 code_review = example.review,
                                 ct = ct_formatter[example.type_correct]))
        
    return prompts, correct_symbols, all_permutations

### Evaluation
def test_example(example, tokenizer, llm, sampling_params, mode, language_type):
    symbols = ["A", "B", "C", "D"]
    symbol_ids = tokenizer.convert_tokens_to_ids(symbols)
    symbol_id_map = dict(zip(symbol_ids, symbols))

    prompt_permutations, correct_answers, combinations = prompt_combinations(example, mode, language_type)
    model_answers = []

    output = llm.generate(prompt_permutations, sampling_params)
    for permutation in output:
        logprobs = permutation.outputs[0].logprobs[0]
        
        symbol_probs = []
        for symbol_id in symbol_id_map.keys():
            if symbol_id in logprobs:
                symbol_probs.append((logprobs[symbol_id].decoded_token, 
                                     logprobs[symbol_id].logprob))
            else:
                symbol_probs.append((symbol_id_map[symbol_id], 
                                     -9999))     
                
        model_answers.append(dict(symbol_probs))
        
    example_record = [combinations, 
              model_answers, 
              [max(symbol_probs, key=symbol_probs.get) for symbol_probs in model_answers],
             correct_answers,
             example.type_correct]

    return pd.DataFrame([example_record], columns=['combinations', 'softmax_probs', 'model_answers', 'correct_answers','GT'])

### Run Test
def main():
    login(token = sys.argv[1])
    language_type = sys.argv[2]
    mode = sys.argv[3]
    model_name = sys.argv[4]
    mcqa_set = pd.read_json("hf://datasets/Tomo-Melb/CodeReviewQA/CodeReviewQA.jsonl", lines = True)

    # Ingest Data 
    if language_type == "C":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "c"]
        save_dir = 'results/cl/c/' + mode + '/cl_' + mode + '_c_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "CPP":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "cpp"]
        save_dir = 'results/cl/cpp/' + mode + '/cl_' + mode + '_cpp_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "CSharp":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "csharp"]
        save_dir = 'results/cl/csharp/' + mode + '/cl_' + mode + '_csharp_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "Go":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "go"]
        save_dir = 'results/cl/go/' + mode + '/cl_' + mode + '_go_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "Java":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "java"]
        save_dir = 'results/cl/java/' + mode + '/cl_' + mode + '_java_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "JavaScript":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "javascript"]
        save_dir = 'results/cl/javascript/' + mode + '/cl_' + mode + '_javascript_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "PHP":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "php"]
        save_dir = 'results/cl/php/' + mode + '/cl_' + mode + '_php_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "Python":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "python"]
        save_dir = 'results/cl/python/' + mode + '/cl_' + mode + '_python_' + model_name.split("/")[1] + '.pkl'
    elif language_type == "Ruby":
        mcqa_set = mcqa_set.loc[mcqa_set['lang'] == "ruby"]
        save_dir = 'results/cl/ruby/' + mode + '/cl_' + mode + '_ruby_' + model_name.split("/")[1] + '.pkl'

    # Import Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    guided_decoding_params = GuidedDecodingParams(choice=["A", "B", "C", "D"])
    sampling_params = SamplingParams(temperature=0,
                                    max_tokens=1,
                                    logprobs=20,
                                    guided_decoding=guided_decoding_params)
                                
    llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), max_model_len=4000)

    # Run Inference
    c_save = pd.DataFrame(columns=['combinations', 'softmax_probs', 'model_answers', 'correct_answers','GT'])
    for row in tqdm(range(len(mcqa_set))):
        example_save = test_example(mcqa_set.iloc[row], tokenizer, llm, sampling_params, mode, language_type)
        c_save = pd.concat([c_save, example_save])
    
    # Save and Output Results
    c_save.to_pickle(save_dir)
    results = pd.read_pickle(save_dir)
    calc_results(results)
    
    # Release Cache
    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

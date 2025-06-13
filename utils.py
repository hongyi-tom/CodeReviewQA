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


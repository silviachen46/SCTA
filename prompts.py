CODING_ROLE_PROMPT = """You are a coding agent. You are supposed to write python code strictly following the prompt."""

DEBUGGING_ROLE_PROMPT = """You are a reviewer agent. You are supposed to carry out instructions step by step to provide debugging or non-debugging general purpose review. If there are specified
format requirements in the prompt please follow it strictly."""

DEBUGGING_REVIEW_PROMPT = """You are provided with a code snippet, the task this code snippet is supposed to accomplish, and the error message from the console. Read line by line carefully 
to identify the source of error and provide helpful instructions on fixing it. Keep your review only about fixing the bug, succinct and straight to the point.
current code : {curr_code}
current error output: {curr_code_output}
You can also refer to this history of code and reviews to avoid mistakes made in the previous reviews:
{run_history}
"""
GENERAL_REVIEW_PROMPT = """You are given a task description, a code snippet from a jupyter notebook, and the running result from the code snippet.
Compare and analyze if the goal of the task has been completely satisfied by the code. If yes you should approve the result, if not deny it, give detailed reason and prompt for improvement. 
At this step you can assume the data integrity and focus on if the instruction is properly satisfied.
Task: {task}
Code: {code}
Output: {code_output}
You can also refer to this history of code and reviews to avoid mistakes made in the previous reviews:
{run_history}
Return a review in the Format {{"decision":"Approved/Denied", "reason":"Approved Reason/ Denied Reason"}}"""

UPDATE_CODE_PROMPT = """
You are given a flawed version of code which needs to be updated with the review. Carefully look into the code and align it with the initial task instruction.
You should also reference to the review given as a helpful source for improving the previous version.
Task Description: {task}
Previous Code : {code}
Review: {review}
You can also refer to this history of code and reviews to avoid mistakes made in the previous reviews:
{run_history}
You are also provided with following code from past action items that are successfully executed and approved from previous task,
you can reference to them for what your data looked like, but do not modify any of them:
{past_code}
"""

FINAL_PURPOSE_REVIEW_PROMPT = """
You are given an analysis from a Single Cell Analysis, containing the proportion of unhealthy versus healthy samples(denote by U versus C in adata_b.obs cell_type). 
Look at the proportion of healthy and unhealthy samples in each cluster. If there are specific clusters with large portion of U samples, the analysis is regarded as valid. If not,
Give reviews on the pipeline for improvements.
Analysis result: {analysis},
Current code pipeline: {result_list},

You are supposed to return your result in the form of a JSON:
{{"decision":"Approved/Denied", "reason":"Approved Reason/ Denied Reason", "metadata_revise": {"id1": "revise suggestion for code block with id1", "id2": "revise suggestion for code block with id2", etc}}}
"""

# prompt improvements - other people to use
# generalizability and wrong assumptions - debug review can't address properly if not provided in initial prompt
# JSON output not stable. - openai json output

CODING_INSTRUCTION = """
You are provided with the following instructions for current action items:
{instruction}
You are also provided with following code from past action items that are successfully executed and approved from previous task,
you can reference to them for what your data looked like, but do not modify any of them:
{past_code}
"""
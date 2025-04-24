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
# You should look into the analysis result carefully and find any possible improvements.
GENERAL_REVIEW_PROMPT = """You are given a task description, a code snippet from a jupyter notebook, and the running result from the code snippet.
You are also given a current review count: {review_count} 
Compare and analyze if the goal of the task has been completely satisfied by the code, and the result make sense biologically enough. 
By "make sense biolgically", you should reference to the specific figures in the code_output to analyze the result for improvements.
If yes and review count is >= 2(this one is mandatory), you should skip the rest requirements and approve the result.

If not deny it, give detailed reason and prompt for improving biologicially. 
You should not approve if there are any possible improvements to make the result more biologically make sense, like adjusting the parameters.
At this step you can assume the data integrity and focus on if the instruction is properly satisfied.
When giving reviews, you should mention:
1. the specific part for improvements, as well as referecning to the provided run result to support your point on improvement. You should explicitly mention which part in the result indicates the changes you proposed.
For instance: using a larger value for x, because in result we see cluster_n / sample_y is showing an abnormal pattern: [list the source you saw in code output].
2. If ARI is ever used, propose to not use it again and decide whether to use the best n_pcs and n_neighbors param found using it.

Task: {task}
Code: {code}
Output: {code_output}
You can also refer to these records of past codes to make your new proposed improvement different from previoius ones: (if empty then we are reviewing for first time)
{past_codes}
Return a review in the Format {{"decision":"Approved/Denied", "reason":"Detailed Review Based on current code and code output."}}"""

UPDATE_CODE_PROMPT = """
You are given a flawed version of code which needs to be updated with the review. Carefully look into the code and align it with the initial task instruction.
You should also reference to the review given as a helpful source for improving the previous version.
Task Description: {task}
Previous Code : {code}
Review: {review}

After writing all your code, you should annotate the python code with python annotation to explain which parameter you have changed according to the review.
"""
# You are also provided with following code from past action items that are successfully executed and approved from previous task,
# you can reference to them for what your data looked like, but do not modify any of them:
# {past_code}

FINALIZE_CMP_CODE_PROMPT = """
You are given a couple of past codes for this single cell analysis task. Referencing to their outputs, decide which one is the MOST biologically effective solution by referencing to their output and return their metadata.

Task_instruction:
{task_instruction}

Past_Code_With_result:
{past_code_wth_result}

return your result in format: {{"is_cmp":"True", "most_effective_metadata": "metadata", "reason": "reason"}}
"""




# prompt improvements - other people to use
# generalizability and wrong assumptions - debug review can't address properly if not provided in initial prompt
# JSON output not stable. - openai json output

CODING_INSTRUCTION = """
You are supposed to write python code following the instruction.
You are provided with the following instructions for current action items:
{instruction}
"""

REVIEW_CMP_CODE = """
You are given a task description, a current code, a run result for curr code, a list of past codes written for this task.
Task:
{task}
Curr Code:
{code}
Curr Result:
{result}
Past Codes:
{past_codes}

You are supposed review the code according to the run result, compare it with the past code provided, and give reviews and guides on writing more diverse codes compared to the past code given. You can try adjusting parameters or choose
between alternative function options that's marked with [OPTION n]. The eventual goal is to generate different plans for the task and compare to get the plan that makes most sense biologically.
Note that if ARI search is already present in previous plans, do not use it again. But if it's not used, try to propose to use it.
Keep your review short and only containing the most important core information.
"""

UPDATE_CMP_CODE = """
You are given a task description, a list of past codes written for this task, and a review to guide you write diverse code.
Task:
{task}
Past Codes:
{past_codes}
Review:
{review}
You are supposed to leverage the given functions to write more diverse codes compared to the code given. But the 
eventual goal is to carry out the task and obtain results that make most sense biologically. You can try adjusting parameters or choose
between alternative functions that's marked with [OPTION n].
"""

SELECT_PROMPT = """
You are given a list of Single cell analysis results and a task description. Each result is obtained using its corresponding code.
Carefully examine the results, analyze which one makes the most biological sense using bioinformatics knowledge according to the task, and return the index of the result you think align the most. 
Result:
{result}
Task Description:
{task}

Return in the format:
{{"selected": "index"}}
"""
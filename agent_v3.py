from call_gpt import OpenAiClient
from executenote import execute_code_in_notebook, extract_python_code
import json
from prompts import CODING_ROLE_PROMPT, DEBUGGING_REVIEW_PROMPT, DEBUGGING_ROLE_PROMPT, GENERAL_REVIEW_PROMPT, UPDATE_CODE_PROMPT, CODING_INSTRUCTION, REVIEW_CMP_CODE, UPDATE_CMP_CODE, SELECT_PROMPT
GLOBAL_PATH = "v3agent/v3agent_test{num}.ipynb"
GENERAL_REVIEW_PATH = "v3agent/v3agent_general{num}.ipynb"
import re
from call_gpt import GeneralReview, Selection
from action_units import general_action_units, ActionUnit, ActionUnitWithContext

class ReviewerAgent:
    def __init__(self, role_prompt = DEBUGGING_ROLE_PROMPT, max_round = 3):
        self.role_prompt = role_prompt
        self.client = OpenAiClient()
    
    def debug_review(self, code, result, task_description, run_history):
        try:
            debug_prompt = DEBUGGING_REVIEW_PROMPT.format(curr_code = code, curr_code_output = result, run_history = run_history)
            result = self.client.call_openai_gpt(sys_prompt=self.role_prompt, prompt = debug_prompt)
            return result
        except Exception as e:
            print(str(e))

    def compare_review(self, code, result, task_description, past_codes):
        try:
            debug_prompt = REVIEW_CMP_CODE.format(code = code, result = result, task = task_description, past_codes = past_codes)
            result = self.client.call_openai_gpt(sys_prompt=self.role_prompt, prompt = debug_prompt)
            return result
        except Exception as e:
            print(str(e))

    
    def general_review(self, code, result, task_description, past_history, review_count): # approve or not
        try:
            # chunk the step_size here
            general_prompt = GENERAL_REVIEW_PROMPT.format(code = code, code_output = result, task = task_description, past_codes = past_history, review_count = review_count)
            result = self.client.gpt_parsed_call(prompt = general_prompt, format = GeneralReview)
            return result
        except Exception as e:
            print(str(e))

    def select(self, results, task_description):
        try:
            result_string = "\n".join(
                f"####### result index {i} #########\n{results[i]}"
                for i in range(len(results))
            )
            select_prompt = SELECT_PROMPT.format(result = result_string, task = task_description)
            selection = self.client.gpt_parsed_call(prompt= select_prompt, format = Selection)
            return selection
        except Exception as e:
            print(str(e))


# class SummarizeAgent:
#     def __init__(self, role_prompt = RESULT_SUMMARIZE_PROMPT):
#         self.role_prompt = role_prompt
#         self.client = OpenAiClient()
    
#     def get_result_summary(self, result):
#         sum_prompt = SUM_PROMPT.format(result = result)
#         result = self.client.call_openai_gpt(sys_prompt=self.role_prompt, prompt = sum_prompt)

# number of compare reviews is another point to be determined

class DataScienceAgent:
    def __init__(self, role_prompt, reviewer = ReviewerAgent(), action_units = general_action_units):
        self.role_prompt = role_prompt
        self.client = OpenAiClient()
        self.reviewer = reviewer
        self.action_units = general_action_units

    def process_action_units(self):
        try:
            print("start processing")
            for action_name, action_unit in self.action_units.items():
                self.take_action(action_unit)
        except Exception as e:
            print(str(e))
    
    

    def generate_initial_cmp_code(self, action_unit, cmp_round = 2):
        results = []
        past_codes = []
        coding_prompt = CODING_INSTRUCTION.format(instruction = action_unit)
        print("executing first generation")
        initial_code = extract_python_code(self.client.call_openai_gpt(prompt = coding_prompt, sys_prompt=CODING_ROLE_PROMPT))
        initial_result = execute_code_in_notebook(GLOBAL_PATH.format(num = "cmp" + str(0)), initial_code)
        curr_code = initial_code
        curr_result = initial_result
        for i in range(cmp_round):
            past_codes_str = "####### next code #########\n".join(past_codes) if len(past_codes) > 0 else ""
            curr_review = self.reviewer.compare_review(code = curr_code, result = curr_result, task_description=action_unit, past_codes=past_codes_str)

            differed_prompt = UPDATE_CMP_CODE.format(code = curr_code, review = curr_review, past_codes = past_codes_str, task = action_unit)
            print("generating next comparison code")
            next_code = extract_python_code(self.client.call_openai_gpt(prompt = differed_prompt, sys_prompt= CODING_ROLE_PROMPT))
            next_result = execute_code_in_notebook(GLOBAL_PATH.format(num = "cmp" + str(i)), initial_code)

            past_codes.append(curr_code)
            results.append(curr_result)
            curr_code = next_code
            curr_result = next_result
        # added at last
        past_codes.append(curr_code)
        results.append(curr_result)
        
        selection = self.reviewer.select(results, action_unit)
        return past_codes[int(selection["selected"])], results[int(selection["selected"])] # return the one selected
        

    def take_action(self, action_unit, review_round = 3):
        # generate action code
        # compare generation of different plans, generate 1 -> review 1 -> generate 2 -> stop and do compare review
        # inner reviews for making the plans more different, compare review is for picking the one that best align with bio context
        # after having optimal plan, we continue review on this plan for k rounds and stop
        # our goal is to adjust the resolution, markers to best align with bio meanings
        print("starting comparison generation")
        selected_code, selected_result = self.generate_initial_cmp_code(action_unit = action_unit)
        curr_code = selected_code
        curr_result = selected_result
        past_codes = []
        for i in range(review_round):
            past_codes_str = "####### next code #########\n".join(past_codes) if len(past_codes) > 0 else ""
            review_result = self.reviewer.general_review(code = curr_code, result = curr_result, task_description=action_unit, past_history=past_codes_str, review_count=i)
            if review_result["decision"] == "Approved":
                break
            review_result = json.dumps(review_result)
            update_prompt = UPDATE_CODE_PROMPT.format(task = action_unit, code = curr_code, review = review_result)
            new_code = extract_python_code(self.client.call_openai_gpt(sys_prompt=CODING_ROLE_PROMPT, prompt = update_prompt))
            new_result = execute_code_in_notebook(GENERAL_REVIEW_PATH.format(num = i), new_code)
            past_codes.append(curr_code)
            curr_code = new_code
            curr_result = new_result
        
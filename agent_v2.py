from call_gpt import OpenAiClient
from executenote import execute_code_in_notebook, delete_notebook_blocks, extract_python_code
import json
from prompts import CODING_ROLE_PROMPT, DEBUGGING_REVIEW_PROMPT, DEBUGGING_ROLE_PROMPT, GENERAL_REVIEW_PROMPT, UPDATE_CODE_PROMPT, CODING_INSTRUCTION
GLOBAL_PATH = "v2agent1.ipynb"
import re
from action_units import general_action_units, ActionUnit, ActionUnitWithContext


class ReviewerAgent: # split into code review and general review(based on biological understanding)

    def __init__(self, role_prompt = DEBUGGING_ROLE_PROMPT, max_round = 3):
        self.role_prompt = role_prompt
        self.max_round = max_round
        self.tools = []
        self.history = [] # a list of HistoricalContext
        self.client = OpenAiClient()

    def review(self, code, run_result, task_desc, is_debug, run_history): # only give review
        # for debug review we need action unit prompt, code, and execution error.
        try:
            if is_debug:
                print("is_debug")
              
                debug_prompt = DEBUGGING_REVIEW_PROMPT.format(curr_code = code, curr_code_output = run_result, run_history = run_history)
                # idea: maybe in reviewer we can also make it return a json containing the code that it thinks problematic?

                result = self.client.call_openai_gpt(sys_prompt=self.role_prompt, prompt = debug_prompt)
            # for general bio purpose review we need action unit prompt, code, result, and biological backgrounds.
                print("debugreview")
                return result
            else:
                general_review_prompt = GENERAL_REVIEW_PROMPT.format(task = task_desc, code = code, code_output = run_result,run_history = run_history)
                result = self.client.gpt_parsed_call(prompt = general_review_prompt) # return a dict
                return result
        except Exception as e:
            print(str(e))
            
            


class DataScienceAgent:
    def __init__(self, role_prompt, reviewer = ReviewerAgent(), action_units = [], max_round = 2, tools = []):
        self.role_prompt = role_prompt
        self.action_units = action_units
        self.max_round = max_round
        self.tools = []
        self.history = [] # a list of HistoricalContext
        self.client = OpenAiClient()
        self.reviewer = reviewer
        self.result_code_list = []
    def initialize_action_units(self):
        for action_unit_name, action_unit_instr in general_action_units.items():
            self.action_units.append(ActionUnit(action_unit_name, action_unit_instr))
            
    
    def process_action_unit(self):
        success_code_history = [] # past code along with code run result
        try:
            for action_unit in self.action_units:
                final_code = self.take_action(action_unit, success_code_history) # process each action units in sequence
                success_code_history.append(final_code)
        except Exception as e:
            print(str(e))

        # we should have a functioning ipy notebook here


    def take_action(self, curr_action_unit: ActionUnit, success_code_history, max_attempt = 3): # for a single action we want to keep total number of running and debugging within a range
        
        # generate code for the new turn, either with debug or review
        curr_history, curr_code = self.generate_action_code(curr_action_unit)#success_code_history
        print("done")
        
        result_text, result_code = self.run_action_code(curr_code, curr_history, success_code_history) # until we have approved result
        
        result = "Code: " + result_code + "\n" + result_text
        # had an updated notebook here
        # the current action unit is finished.
        return result

    def generate_action_code(self, curr_action_unit: ActionUnit):
        past_code = "\n".join(self.result_code_list)
        coding_prompt = CODING_INSTRUCTION.format(instruction = curr_action_unit.instruction, past_code = past_code)
        result_code = self.client.call_openai_gpt(prompt = coding_prompt, sys_prompt=self.role_prompt)
        
        extracted_code = extract_python_code(result_code)
        
        # parse code and store to history
        curr_history = ActionUnitWithContext(curr_action_unit.name, curr_action_unit.instruction, curr_action_unit)

        id = f"coding_attempt{curr_history.debug_num}" # curr coding id
        curr_history.debug_num += 1

        curr_history.codes[id] = result_code # save to history

        return curr_history, extracted_code
    
    def update_code_with_review(self, prev_code, prev_review, curr_history, run_history): # review is either debug review or general purpose review.
        past_code = "\n".join(self.result_code_list)
        updated_prompt = UPDATE_CODE_PROMPT.format(task = curr_history.action_unit.instruction, code = prev_code, review = prev_review, run_history = run_history, past_code = past_code)
        updated_code = self.client.call_openai_gpt(updated_prompt, sys_prompt = CODING_ROLE_PROMPT)
        extracted_code = extract_python_code(updated_code)
        return extracted_code ## this is enough since notebook helper deals with strings containing code directly

    # execute and get result
    def run_action_code(self, curr_code, curr_history, success_code_history, max_round = 3):
        result_list = []
        result = execute_code_in_notebook(GLOBAL_PATH, curr_code)
        print(result)
        result_list.append(result)
        # result is a dictionary with bug state
        # {'code_result': ['yes\nNew block added and executed!', '5'], 'code_state': 'Success'}
        curr_history.run_num += 1
        result['run_number'] = curr_history.run_num
        curr_round = 0
        # we only need result_list when passing review history to reviewer agent

        while curr_round < max_round:
   
            if curr_round != 0: # not first round and not terminated, replace curr_code
                print("replacing current code")
                result_list_str = " ".join(json.dumps(result) for result in result_list)
                #print(result_list_str)
                curr_code = self.update_code_with_review(curr_code, result['code_review'], curr_history, run_history = result_list_str)#
                result = execute_code_in_notebook(GLOBAL_PATH, curr_code)
                print(result)
            # rewrite until no error and aligns with result
            if result['code_state'] == 'Success': # compile without error
                # send to reviewer agent for review
                curr_history.curr_result = result['code_result']

                general_review_result = self.reviewer.review(curr_code, result, task_desc = curr_history.action_unit.instruction, is_debug = False, run_history = result_list, succcess_code_history = curr_code)
                print(general_review_result)
                if general_review_result['decision'] == 'Approved':
                    self.result_code_list.append(curr_code) # collection of all code blocks successfully run and approved by reviewer
                    return result['code_result'], curr_code# we terminate only if error less and approved
                else:
                    result['code_review'] = general_review_result['reason'] # contains a dictionary like {"decision":"Approved/Denied", "reason":"Approved Reason/ Denied Reason"}
                    delete_notebook_blocks(GLOBAL_PATH, cell_id=result['metadata']) # code is correct but not approved, removed manually

                
            else: # there's error
                print("review start")
                debug_review_result = self.reviewer.review(curr_code, result, task_desc= "", is_debug = True, run_history = result_list, success_code_history = success_code_history) # format convert result_list automatically to string
                print("review finished")
                print(debug_review_result)
                result['code_review'] = debug_review_result
                # if notebook is errored it self we won't be adding it to the notebook
            
            curr_round += 1
            result_list.append(result)
        
    
# only attached the code history for debugging purpose, not general purpose review
# a code generator can be provided with two stuff: the whole previous successful code history + the past reviewed but failed code
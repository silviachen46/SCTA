import uuid
from call_gpt import OpenAiClient, GeneSelection
from executenote import extract_python_code, execute_code_in_notebook, read_last_n_results
from utils_agent import fetch_string_neighbors_clean, fetch_gene_summary
import json
import os
# define your load context here

### everything to replace starts here
TIME_AWARE = False
BIOLOGICAL_CONTEXT = """This study is on sample with patients with COVID-19, Flu, and healthy controls."""
#"""This dataset comprises single-cell data of human prostate adenocarcinoma samples, including both tumor and benign tissues."""

TEST_FILE_NAME = "covid_test_agent.ipynb"##"template_code/tumor_v4agent_test3.ipynb"

GLOBAL_RESULT = "graph_result1.txt" #"template_code/tumor_graph_results1.txt"

ADATA_SOURCE_PATH = "/Users/silviachen/Documents/Software/new_sca_agent/SCAagent/covid_adata.h5ad" #"template_code/tumor_adata.h5ad"

### everything to replace ends here

PREPROCESS_ROLE_PROMPT = """You are responsible for cleaning, normalizing, and preparing the data for downstream analysis."""


VISUALIZE_ROLE_PROMPT = """You generate insightful plots and visual representations to highlight structure and patterns in the data."""

INSIGHT_ROLE_PROMPT = """You analyze processed and visualized data to extract meaningful biological or statistical insights."""

ANNOTATE_ROLE_PROMPT = """You assign biological interpretations or cell type labels based on gene expression and reference knowledge."""

TIME_AWARE_ROLE_PROMPT = """You are a time-aware agent grouping and provide timewise insights."""

RESULT_ROLE_PROMPT = """You compile outputs from all previous agents and generate a final, structured report for the user."""

FILTER_ROLE_PROMPT = """You should pick genes from the given dictionary to further investigate them."""

REVIEW_ROLE_PROMPT = """You should give review for the current results with respect to its biological context"""

DEBUG_ROLE_PROMPT = """You should fix the given code with regard to reported errors."""

PREPROCESS_TOOL = """
functions that would return an updated version of adata would have  -> return adata at the end. You should use adata = function() for that
You should import these functions from utils_agent. You should also import scanpy, pandas, numpy, and matplotlib.pyplot.
for function returning None you should use them directly without trying to return adata.

def filter_cells(adata, min_cells=3, min_genes=200, max_genes=50000) -> return None
def normalize_log_transform(adata, target_sum = 1e4) -> return None
def filter_lowqc_cells(adata, pct_counts_mt_upbound = 10, n_genes_by_counts = None, pct_counts_mt_lowbound = None) -> return adata
def save_high_var_gene(adata, n_top_genes = 4000) -> return adata
def pca_and_plot_umap(
    adata, 
    n_pcs=40, 
    n_neighbors=10, 
    resolution=0.7,
) -> return None
def get_top_marker_genes(adata, groupby='leiden', method='wilcoxon', top_n=5) -> return None
"""


PREPROCESS_TASK_PROMPT = """
Task Description: 
You are to write python code to:
conduct Single Cell analysis on the given data. 
Before start, use import warnings
warnings.filterwarnings("ignore", category=UserWarning) to filter the warning

use adata = sc.read_h5ad(""" + ADATA_SOURCE_PATH + """) to load your source data.

You should first quality control & filter the data,
PCA and compute UMAP.
get top marker gene for each cluster using the function provided.
save your adata locally to current folder named as adata_preprocessed.h5ad
You should also inspect adata structure by calling the given quick inspect function. 
I will use this regex to match the code. Generate python code following this patter: pattern = r"```python\n(.*?)\n```"
Here is the set of functions available for you to use. You should use functions whenever possible:
{functions}
"""

ANNOTATE_TOOL = """
import the functions you need from utils_agent.
functions that would return an updated version of adata would have  -> return adata at the end.
def assign_cell_categories(adata, cluster_to_category = None) -> return adata
def assign_cell_subtype(adata, cluster_to_subtype = None) -> return None
def quick_inspect_adata(adata, max_unique=8)
def annotate_with_celltypist(adata, model_path = 'Immune_All_Low.pkl') -> return None
"""
ANNOTATE_TASK_PROMPT = """
Task Description:
You are to write python code to:
LOAD your adata from adata_preprocessed.h5ad in current folder
based on the top-expressed genes in each cluster and the biological context of the data, analyze which cell types does each cluster represent.
Please annotate the cells with major lineages, aiming for a coarse-level classification (around 6 ~ 10 broad cell types).
based on your analysis, define a python dictionary in the following format:
cluster_to_category = {{
        "0": "cell_type",
        "1": "cell_type",
        "2": "cell_type",
        "3": "cell_type" }}
and inject it as param to the given function to assign categories.
use pd.crosstab(adata.obs["cell_category"], adata.obs["group"]) to show some insights for further analysis
according to your major lineage identification, also identify subtypes based on the deg analysis and put it in similar format into a dictionary named cluster_to_subtype.
assign subtype using the corresponding function given.
also call given celltypist annotate function to assign label
SAVE your adata locally to current folder named as adata_annotated.h5ad
I will use this regex to match the code. Generate python code following this patter: pattern = r"```python\n(.*?)\n```"

Here's the biological context of the adata:
{context}
Here's the DEG results from previous step:
{results}
Here is the set of functions available for you to use. You should use functions whenever possible:
{functions}
"""

INSIGHT_TOOL = """
import the functions you need from utils_agent.
def get_deg_full(
    adata,
    groupby: str = "condition",
    reference: str = "Control",
    method: str = "wilcoxon",
    top_n: int = 20,
    use_raw: bool = False,
    layer: str = None,
    pts: bool = True,
    key_added: str = "rank_genes_groups",
    n_genes : int = 300
) -> dict[str, pd.DataFrame]
def get_gene_by_disease(adata, curr_adata, curr_group, cell_types_to_analyze, n_genes = 300, control_type="Ctl") -> Dictionary
The default value for n_genes is 300, you should decide what specific value to use based on the adata info.
def get_filtered_gene_list(adata, gene_list) -> List
"""
## insight is changed
INSIGHT_TASK_PROMPT = """
Task Description:
You are to write python code to:
load your data from adata_annotated.h5ad in current folder
remove potential logging by 
"import logging
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)"
use your result from last step of pd.crosstab(adata.obs["cell_category"], adata.obs["group"]) and biological context below to figure out which cell types are worth further analysis, give a python list of their names.
biological_context:
{biological_context}
result:
{result}
Based on former result, first figure out different types of disease subtype to study here (everything other than Control or Normal group) for e.g. to_study = ["a", "b", "c"]
Save a copy for each of their adata with the control type: adata_a = adata[adata.obs['group'].isin(['control_type', 'a'])].copy() for each of the disease subtype. You should adjust control_type name as given.
Now for each of these sub adata using something like for curr_adata, curr_group in zip([adata_a, adata_b], ["a", "b"]): , 
Get potential gene set for each disease type using the function provided with adata(the original one), curr_adata, curr_group, cell_types_to_analyze, and the control_type name in current dataset, store the output in variable potential_gene_set.
Filter the gene set with given function, which accepts a list of gene symbols(keys of the gene dictionary) and return a list of filtered gene symbols. You should filter the dictionary and only keep those present in list.
Print the current group name along with the filtered potentail gene dictionary, switch line everytime.
I will use this regex to match the code. Generate python code following this patter: pattern = r"```python\n(.*?)\n```"

Here is the set of functions available for you to use. You should use functions whenever possible:
{functions}
"""

TIME_AWARE_TOOL = """
def deg_for_time_series(adata, time_groups, sample_group_col="sample_group", k=10) -> return results
"""

TIME_AWARE_TASK_PROMPT = """
Given previous results, first group each sample group according to time range into a list of lists according to time order named input_list.
Example: input_list = [["time1-disease1", "time2-disease1"], ["time1-disease2", "time2-disease2"]]
use the given function get a insights into timewise changes.
"""

# if time aware merge this prompt into insight agent prompt
if TIME_AWARE:
    INSIGHT_TASK_PROMPT = INSIGHT_TASK_PROMPT + "\n" + TIME_AWARE_TASK_PROMPT
    INSIGHT_TOOL = INSIGHT_TOOL + "\n" + TIME_AWARE_TOOL


# batch filtering
FILTER_TASK_PROMPT = """
You are given biological background knowledge, protein interaction context, and DEG&TF enrichment info. 
You should decide which genes in given gene set to keep for biologist to investigate as potentail novel target for given disease context.
You should return a list of selected gene and give reasoning for your selection.
Disease biological context: {bio_context}
Detailed information on each genes:
{gene_info}
"""

# replace result agent with filter agent

RESULT_TASK_PROMPT = """
Given all the above results, compile a comprehensive report summarizing the key findings and the possible target genes from the result. 
Make sure to reference to the actual numbers when naming target genes and show your reasoning. You should filter through the results to identify genes that are most likely to be the target.
For each of these potential target gene choice you are also provided with top frequent subtypes and groups marked with CellTypist as reference for your judgements.
Also reference to the biological context: {bio_context}
Results from DEG for individual cell groups:
{results}
"""

# need to pass task description as part of the output
DEBUG_TASK_PROMPT = """
You are supposed to fix the reported bug based on the given task description, wrong code snippet, and reported error.
Task Description : {task_desc}
Wrong Code: {wrong_code}
Error Trace: {error_trace}
Write correct version of the python code to conduct the task.
I will use this regex to match the code. Generate python code following this patter: pattern = r"```python\n(.*?)\n```"
"""
class AgentBase:
    def __init__(self, role_prompt="", tool_prompt=None, task_prompt = None):
        self.role_prompt = role_prompt
        self.tool_prompt = tool_prompt
        self.task_prompt = task_prompt
        self.client = OpenAiClient()
        self.codes = []
        self.results = []

    def act(self):
        raise NotImplementedError("Each agent must implement its own act() method.")
    
    def fix(self, *, max_retry=3, task_desc="", wrong_code="", error_trace=""):
        raise NotImplementedError("This agent does not support fix()")


class AgentNode:
    def __init__(self, agent: AgentBase, name=None, local_context = None, metadata_id = None):
        self.agent = agent
        self.name = name or agent.__class__.__name__
        self.next_nodes = []
        self.local_context = local_context if local_context is not None else {}
        self.metadata_id = metadata_id
        self.output = None

    def add_next(self, next_node):
        self.next_nodes.append(next_node)

    def run(self, global_context=None):
        print(f"Running {self.name}")
        output = self.agent.run(global_context=global_context)
        return output
    

    

class PreprocessingAgent(AgentBase):
    def run(self, global_context=None):
        self.task_prompt = self.task_prompt.format(functions=self.tool_prompt)
        current_code = extract_python_code(self.client.call_openai_gpt(self.task_prompt, sys_prompt=self.role_prompt))
        self.codes.append(current_code)
        result = execute_code_in_notebook(TEST_FILE_NAME, current_code)
        result["task_desc"] = self.task_prompt
        self.results.append(result)
        return result

# class VisualizationAgent(AgentBase):
#     def act(self, input_data):
#         return f"[VisualizationAgent] Visualizing: {input_data}"
    
#     def run(self, input_data, global_context=None):
#         print("Planning with goal:", global_context.get("goal"))
#         return input_data  # or modified data

class AnnotationAgent(AgentBase):
    def run(self, global_context=None):
        with open(GLOBAL_RESULT, "r", encoding="utf-8") as f:
            results = f.read()
        self.task_prompt = self.task_prompt.format(context=BIOLOGICAL_CONTEXT, results = results, functions=self.tool_prompt)
        current_code = extract_python_code(self.client.call_openai_gpt(self.task_prompt, sys_prompt=self.role_prompt))
        self.codes.append(current_code)
        result = execute_code_in_notebook(TEST_FILE_NAME, current_code)
        result["task_desc"] = self.task_prompt
        self.results.append(result)
        return result


class InsightAgent(AgentBase):
    def run(self, global_context=None):
        with open(GLOBAL_RESULT, "r", encoding="utf-8") as f:
            result = f.read()
        self.task_prompt = self.task_prompt.format(functions=self.tool_prompt, biological_context = BIOLOGICAL_CONTEXT, result = result)
        current_code = extract_python_code(self.client.call_openai_gpt(self.task_prompt, sys_prompt=self.role_prompt))
        self.codes.append(current_code)
        result = execute_code_in_notebook(TEST_FILE_NAME, current_code)
        result["task_desc"] = self.task_prompt
        self.results.append(result)
        return result

class TimeAwareAgent(AgentBase):
    def run(self, global_context=None):
        with open(GLOBAL_RESULT, "r", encoding="utf-8") as f:
            result = f.read()
        self.task_prompt = self.task_prompt.format(functions=self.tool_prompt, result = result)
        current_code = extract_python_code(self.client.call_openai_gpt(self.task_prompt, sys_prompt=self.role_prompt))
        self.codes.append(current_code)
        result = execute_code_in_notebook(TEST_FILE_NAME, current_code)
        result["task_desc"] = self.task_prompt
        self.results.append(result)
        return result

class ResultAgent(AgentBase):
    def run(self, global_context=None):
        # if TIME_AWARE:
        #     last_result = read_last_n_results(filepath=GLOBAL_RESULT, n = 2) # we wanna also include the timewise comparison if set to timeaware
        # else:
        last_result = read_last_n_results(filepath=GLOBAL_RESULT, n = 1)
        self.task_prompt = self.task_prompt.format(results=last_result, bio_context = BIOLOGICAL_CONTEXT)
        result = self.client.call_openai_gpt(self.task_prompt, sys_prompt=self.role_prompt)
        return {"code_state": "Success", "code_result": result, "code_error": None}

# filter agent runs batch filtering until we get ideal amount of target genes
class FilterAgent(AgentBase):
    def run(self, global_context=None, batch_size=5, target_gene_number=5):

        # folder where all potential gene sets are stored
        folder_path = os.path.join(os.getcwd(), "potential_gene_set")

        if not os.path.exists(folder_path):
            return {
                "code_state": "Failed",
                "code_error": "potential_gene_set folder not found",
                "code_result": None
            }

        # list all JSON files inside potential_gene_set/
        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

        # extract group names from filenames
        group_list = [os.path.splitext(f)[0] for f in json_files]

        # process each group
        for group in group_list:
            print(f"🔍 Processing group = {group}")
            self.single_filter_target(group, batch_size, target_gene_number)

        return {"code_state": "Success", "code_error": None, "code_result": f"Processed groups: {group_list}"}

    def single_filter_target(self, curr_group, batch_size=5, target_gene_number=5):

        # --------------------------
        # Load correct JSON file
        # --------------------------
        potential_folder = os.path.join(os.getcwd(), "potential_gene_set")
        potential_file = os.path.join(potential_folder, f"{curr_group}.json")

        if not os.path.exists(potential_file):
            print(f"⚠️ WARNING: {potential_file} not found. Skip group.")
            return

        with open(potential_file, "r", encoding="utf-8") as f:
            potential_gene_set = json.load(f)

        gene_search_list = sorted(list(potential_gene_set.keys()))

        # --------------------------
        # Prepare result folder
        # --------------------------
        result_folder = os.path.join(os.getcwd(), "result_gene")
        os.makedirs(result_folder, exist_ok=True)

        # result file
        result_file = os.path.join(result_folder, f"result_{curr_group}.json")

        # initialize empty result JSON
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4)

        ptr = 0

        # --------------------------
        # Main filtering loop
        # --------------------------
        while True:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if len(data) >= target_gene_number:
                break

            gene_info = ""
            tmp_gene_info = {}

            for i in range(ptr, min(ptr + batch_size, len(gene_search_list))):
                curr_gene = gene_search_list[i]
                curr_gene_deg = potential_gene_set[curr_gene]
                curr_gene_neighbor = fetch_string_neighbors_clean(curr_gene)
                curr_gene_context = fetch_gene_summary(curr_gene)

                tmp_gene_info[curr_gene] = {
                    "deg": curr_gene_deg,
                    "neighbor": curr_gene_neighbor,
                    "context": curr_gene_context
                }

                curr_gene_info = (
                    curr_gene + "\n" +
                    str(curr_gene_deg) + "\n" +
                    str(curr_gene_context) + "\n" +
                    str(curr_gene_neighbor)
                )

                if gene_info:
                    gene_info += "\n" + curr_gene_info
                else:
                    gene_info = curr_gene_info

            # ------------ LLM call ------------
            base_prompt = self.task_prompt
            formatted_prompt = base_prompt.format(
                bio_context=BIOLOGICAL_CONTEXT,
                gene_info=gene_info
            )
            result = self.client.gpt_parsed_call(prompt=formatted_prompt, format=GeneSelection)
            print("LLM result:", result)

            # update result JSON safely
            for selected_gene in result["selected"]:
                if selected_gene in tmp_gene_info:
                    data[selected_gene] = tmp_gene_info[selected_gene]
                    data[selected_gene]["reasoning"] = result["reasoning"]
                else:
                    print(f"⚠ LLM selected gene not in batch: {selected_gene}")

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            # move pointer
            ptr += min(batch_size, len(gene_search_list) - ptr)
            if ptr >= len(gene_search_list):
                break

        print(f"✅ Finished group {curr_group}")
        return

    # def single_filter_target(self, group, batch_size, target_gene_number):
    #     # read from a json file to check if we have enough target gene
    #     # for each key(gene symbol) prepare the bio info
    #     with open("results.json", "w", encoding="utf-8") as f:
    #         json.dump({}, f, indent=4)
    #     with open("potential_gene_set.json", "r", encoding="utf-8") as f:
    #         potential_gene_set = json.load(f)
    #     gene_search_list = sorted(list(potential_gene_set.keys()))
    #     ptr = 0
        
        

    #     while True:
    #         with open("results.json", "r", encoding="utf-8") as f:
    #             data = json.load(f)

    #         if len(data) >= target_gene_number:
    #             break

    #         gene_info = ""
    #         tmp_gene_info = {}
    #         # prepare gene_info, process it in sequence
    #         for i in range(ptr, min(ptr + batch_size, len(gene_search_list))):
    #             curr_gene = gene_search_list[i]
    #             curr_gene_deg = potential_gene_set[curr_gene]
    #             curr_gene_neighbor = fetch_string_neighbors_clean(curr_gene)
    #             curr_gene_context = fetch_gene_summary(curr_gene)
    #             tmp_gene_info[curr_gene] = {
    #                 "deg": curr_gene_deg,
    #                 "neighbor": curr_gene_neighbor,
    #                 "context": curr_gene_context
    #             }
    #             curr_gene_info = curr_gene + "\n" + str(curr_gene_deg) + "\n" + str(curr_gene_context) + "\n" + str(curr_gene_neighbor)
    #             if gene_info:
    #                 gene_info += "\n" + curr_gene_info
    #             else:
    #                 gene_info += curr_gene_info

    #         base_prompt = self.task_prompt
    #         formatted_prompt = base_prompt.format(bio_context = BIOLOGICAL_CONTEXT, gene_info = gene_info)
    #         result = self.client.gpt_parsed_call(prompt = formatted_prompt, format = GeneSelection)
    #         print(result)
    #         # print(result["reasoning"])
    #         ptr += min(batch_size, len(gene_search_list) - ptr)
    #         if ptr >= len(gene_search_list):
    #             break
    #         for selected_gene in result["selected"]:
    #             data[selected_gene] = tmp_gene_info[selected_gene]

    #         with open("results.json", "w", encoding="utf-8") as f:
    #             json.dump(data, f, indent=4) # additional data
        


class DebugAgent(AgentBase):
    def fix(self, max_retry, task_desc, wrong_code, error_trace):
        print("Trying to fix ...")
        # given former code, cold error, task description, try to write a new code snippet and see if it works
        for i in range(max_retry):
            self.task_prompt = self.task_prompt.format(task_desc = task_desc, wrong_code = wrong_code, error_trace = error_trace)
            current_code = extract_python_code(self.client.call_openai_gpt(self.task_prompt, sys_prompt=self.role_prompt))
            result = execute_code_in_notebook(TEST_FILE_NAME, current_code)
            if result["code_state"] == "Error": # task_desc unchanged
                wrong_code = current_code
                error_trace = result["code_error"]
                continue
            else: 
                self.codes.append(current_code)
                self.results.append(result)
                print("Fixed successfully! returning...")
                return {"code_state": "Success", "code_result": result, "code_error": None}
        if result["code_state"] == "Error":
            return result
        self.codes.append(current_code) # not early returned but still suceeded
        self.results.append(result)
        print("Fixed successfully! returning...")
        return {"code_state": "Success", "code_result": result, "code_error": None}

def persist_results(results, path=GLOBAL_RESULT):
    """Append all current results to the given file, overwriting each time."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            for idx, r in enumerate(results):
                f.write(f"Step {idx + 1} Result:\n{r}\n\n")
        print(f"Appended {len(results)} results to {path}")
    except Exception as e:
        print(f"⚠ Failed to append graph_results to file: {e}")

class AgentGraph:
    def __init__(self):
        self.nodes = {}
        self.execution_order = []
        
        self.preprocess = AgentNode(PreprocessingAgent(role_prompt=PREPROCESS_ROLE_PROMPT, tool_prompt=PREPROCESS_TOOL, task_prompt = PREPROCESS_TASK_PROMPT), name="Preprocessing")
        # self.visualize = AgentNode(VisualizationAgent(role_prompt=VISUALIZE_ROLE_PROMPT),name="Visualization")
        self.insight = AgentNode(InsightAgent(role_prompt=INSIGHT_ROLE_PROMPT, tool_prompt = INSIGHT_TOOL, task_prompt=INSIGHT_TASK_PROMPT), name="Insight")
        self.annotate = AgentNode(AnnotationAgent(role_prompt=ANNOTATE_ROLE_PROMPT, tool_prompt=ANNOTATE_TOOL, task_prompt=ANNOTATE_TASK_PROMPT), name="Annotation")
        # self.result = AgentNode(ResultAgent(role_prompt=RESULT_ROLE_PROMPT, task_prompt= RESULT_TASK_PROMPT), name="Result")
        self.debugger = AgentNode(DebugAgent(role_prompt=DEBUG_ROLE_PROMPT, task_prompt = DEBUG_TASK_PROMPT), name="Debug")
        # self.timewise = AgentNode(TimeAwareAgent(role_prompt=TIME_AWARE_ROLE_PROMPT, task_prompt=TIME_AWARE_TASK_PROMPT), name = "TimeAware")
        self.filterer = AgentNode(FilterAgent(role_prompt=FILTER_ROLE_PROMPT, tool_prompt="", task_prompt=FILTER_TASK_PROMPT), name = "Filterer")
        self.global_context = None
        self.graph_results = []
        self.graph_codes = []

    def add_node(self, name, agent):
        metadata_id = str(uuid.uuid4())[:8]
        self.nodes[name] = AgentNode(agent, metadata_id)

    def add_edge(self, from_node, to_node):
        self.edges.setdefault(from_node, []).append(to_node)

    def execute_graph(self, initial_input=None, max_retry = 3):
        # return the task desc together with the output dictionary
        for node_name in self.execution_order:
            node = self.nodes[node_name]
            print(f"\n▶ Executing: {node_name} ({node.metadata_id})")
            output = node.agent.run(global_context=self.graph_results)
            print(f"✔ Output: {output}")
            if output["code_state"] == "Error":
                print(f"Error in {node_name}: {output['code_error']}")
                # pass the task description, code error, and wrong code snippet to the debugger
                # if error resolved and the code executed without error just continue
                result = self.debugger.agent.fix(max_retry = max_retry, task_desc = output["task_desc"], wrong_code = output["errored_code"], error_trace = output["code_error"]) # repetitively debug
                if result["code_state"] == "Error":
                    # still error, end it
                    print("still errored after max retries. Terminating")
                    print(f"Error: {result['code_error']}")
                    return result
                else:
                    # fixed, continue
                    self.graph_results.append(result["code_result"])
                    persist_results(self.graph_results)
                    continue
            else:
                self.graph_results.append(output["code_result"])
                persist_results(self.graph_results) # persist result trace stage-by-stage
                
        
            

class PlanningAgent(AgentBase):
    def __init__(self):
        self.graph = AgentGraph()

    def build_graph(self):
        self.graph.add_node("Preprocessing", self.graph.preprocess)
        self.graph.add_node("Annotation", self.graph.annotate)
        self.graph.add_node("Insight", self.graph.insight)
        self.graph.add_node("Filterer", self.graph.filterer)
        # self.graph.add_node("TimeAware", self.graph.result)
        # self.graph.add_node("Result", self.graph.result)
        
        self.graph.execution_order = [
            "Preprocessing",
            "Annotation",
            "Insight",
            "Filterer"        
            # "Result"
        ]

# result 格式 is specified in execute in notebook
# attach task descirption to what is being returned

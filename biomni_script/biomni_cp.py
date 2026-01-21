from biomni.agent import A1
import time
# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', source = "OpenAI", llm="gpt-4o")


prompt = """
This is the biological context on my gene file, BIOLOGICAL_CONTEXT = The dataset contains single-cell RNA sequencing profiles of peripheral blood mononuclear cells collected from healthy individuals, patients with mild COVID-19, patients with severe COVID-19, and patients with severe influenza.
you will be given this adata, and you are supposed to follow the typical Single Cell Analysis pipeline procedure to find the top10 potential target gene for EACH disease group in adata. Give detailed reasoning for your selection. You should not refer to external knowledge of prior targets or clear known disease-gene association.
The file path is ./data/biomni_data/data_lake/covid_adata.h5ad
"""

# prompt = """
# This is the biological context on my gene file, BIOLOGICAL_CONTEXT = This dataset comprises single-cell data of human prostate adenocarcinoma samples, including both tumor and benign tissues.
# you will be given this adata, and you are supposed to follow the typical Single Cell Analysis pipeline procedure to find the top10 potential target gene for EACH disease group in adata. Give detailed reasoning for your selection. You should not refer to external knowledge of prior targets or clear known disease-gene association.
# The file path is ./data/biomni_data/data_lake/tumor_adata.h5ad
# """
# prompt = """
# This is the biological context on my gene file, BIOLOGICAL_CONTEXT = The dataset contains single-cell RNA sequencing profiles of peripheral blood mononuclear cells collected from healthy individuals, patients with mild COVID-19, patients with severe COVID-19, and patients with severe influenza.
#  you will be given this adata, and you are supposed to follow the typical Single Cell Analysis pipeline procedure to find the top10 potential target gene for each disease group. Give detailed reasoning for your selection. You should not refer to external knowledge of prior targets or clear known disease-gene association.
# The file path is ./data/biomni_data/data_lake/covid_adata.h5ad"""


start_time = time.perf_counter()
# Execute biomedical tasks using natural language
agent.go(prompt)
end_time = time.perf_counter()

print(f"Total execution time: {end_time - start_time:.2f} seconds")
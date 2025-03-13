

import os
from openai import AzureOpenAI
import pandas as pd
import numpy as np
import json
import ast
import re
import subprocess
import tqdm
from agent_v2 import DataScienceAgent

role_prompt = """You are supposed to write python code with scanpy, numpy, pandas, matplotlib.pyplot 
to carry out the following bioinformatics subtasks in Single Cell Analysis. If a parameter required by the function is not specified,
You are supposed to assign a proper value according to biological commonsense."""


ds_agent = DataScienceAgent(role_prompt=role_prompt)
ds_agent.initialize_action_units()
ds_agent.process_action_unit()

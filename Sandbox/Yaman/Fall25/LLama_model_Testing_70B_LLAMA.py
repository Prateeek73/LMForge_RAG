import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return the number of GPUs
print(torch.cuda.get_device_name(0))  # Should show the GPU model


import transformers
import torch
import os
import json
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
from datetime import timedelta, datetime
import pandas as pd
from dotenv import load_dotenv
import shutil 

import evaluate
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load environment variables
load_dotenv(dotenv_path="../../.env") # path is relative to this script, adjust as needed

run_id = "LMForge_RUN08_DGX_SPARK_Llama-3-3-70B-Instruct"  # <- Change this manually for each experiment
batch_size = 10  # <- Change this manually for each experiment

#from transformers.utils import LossKwargs


import logging
logging.basicConfig(filename='generation.log', level=logging.INFO)
logging.info(f"Run ID: {run_id}")


# setting huggingface token
login(token=os.getenv("#"))

# os.environ["HF_HOME"] = "D:/huggingface_cache" 
# os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface_cache"

# print("HF_HOME:", os.getenv("HF_HOME"))
# print("TRANSFORMERS_CACHE:", os.getenv("TRANSFORMERS_CACHE"))
# print("HUGGINGFACE_HUB_CACHE:", os.getenv("HUGGINGFACE_HUB_CACHE"))

# logging.info(f"HF_HOME: {os.getenv('HF_HOME')}")
# logging.info(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE')}")
# logging.info(f"HUGGINGFACE_HUB_CACHE: {os.getenv('HUGGINGFACE_HUB_CACHE')}")

# transformers.utils.hub.TRANSFORMERS_CACHE = "D:/huggingface_cache"


model_name = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)

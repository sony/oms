
import configparser
import os
import pdb
import time

from langchain import hub
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_react_agent
from langchain_community.document_loaders.s3_file import S3FileLoader

import pandas as pd

import random
import importlib.util

#from google_util.google_ads_api import get_keyword_planner_metrics

import boto3
import io


# from metrics import evaluate_keywords_against_paragraph, jaccard_similarity, cosine_similarity_calc, find_most_relevant_keywords,find_best_match_for_keyword,update_clicks,r_kw_plan_bert,r_kw_plan


# from concurrent.futures import ProcessPoolExecutor

# import torch
# from transformers import AutoTokenizer, AutoModel
# from bert_score import score as bert_score
# from bert_score import BERTScorer
# from sklearn.feature_extraction.text import CountVectorizer

# from transformers import logging
# logging.set_verbosity_error()

# # Load the BERT model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
# ## Initialize the BERTScorer for multilingual BERT or a Japanese-specific BERT
# scorer = BERTScorer(model_type="bert-base-multilingual-cased", lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')

# from transformers import BertTokenizer, BertModel


#tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
#bert_model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
#scorer = BERTScorer(lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')
#scorer = BERTScorer(model_type="cl-tohoku/bert-base-japanese", lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')

#from okg.load_and_embed import customized_trend_retriever
#from okg.utils import run_with_retries
#from tools import getSerpTool, OutputTool, ClickAggregator

from langchain_community.agent_toolkits.load_tools import load_tools


#import sys
#sys.path.append('../')



class okg_agent:
    def __init__(self ,s3_bucket, s3_path, config_path, setting_day, code_in = 0):

        # 0. Read the configuration file
        self.config = configparser.ConfigParser()
        try:
            self.config.read(config_path)
            #self.config.read('./config_base.ini')
        except Exception as e:
            raise ValueError("Failed to read the configuration file: " + str(e))
        
        self.observation_period = int(self.config['SETTING']['OBSERVATION_PERIOD_DAY'])
        
        # self.csv_file_path = self.config['FILE']['CSV_FILE']

        #self.setting_day = pd.to_datetime(self.config['SYSTEM']['SETTING_DAY'])
        self.setting_day = pd.to_datetime(setting_day)

        # self.dataframe = pd.read_csv(str(self.config['FILE']['CSV_FILE']))

        if str(self.config["EXE"]["S3_DEPLOY"]) == "True":
            self.save_dir =  s3_path
            base_path = 'local_policy'
            self.initial_keyword_data = self.adjust_path_for_local_policy(self.config["SETTING"]["initial_keyword_data"])
            self.generated_keyword_data = self.adjust_path_for_local_policy(self.config["SETTING"]["generated_keyword_data"])
            self.rule_data = self.adjust_path_for_local_policy(self.config["SETTING"]["rule_data"])
            self.rejected_keyword_list = self.adjust_path_for_local_policy(self.config["SETTING"]["rejected_keyword_list"])
            
        elif str(self.config["EXE"]["S3_DEPLOY"]) == "False":
            self.save_dir = './'
            base_path = '.'
            self.initial_keyword_data = self.config["SETTING"]["initial_keyword_data"]
            self.generated_keyword_data = self.config["SETTING"]["generated_keyword_data"]
            self.rule_data = self.config["SETTING"]["rule_data"]
            self.rejected_keyword_list = self.config["SETTING"]["rejected_keyword_list"]
            self.get_keyword_planner_metrics = self.dynamic_import('google_util/google_ads_api', 'get_keyword_planner_metrics')

        
        # Dynamic import based on calculated base path
        self.customized_trend_retriever = self.dynamic_import(f"{base_path}/okg/load_and_embed", "customized_trend_retriever")
        self.run_with_retries = self.dynamic_import(f"{base_path}/okg/utils", "run_with_retries")
        self.normalize_keyword  = self.dynamic_import(f"{base_path}/okg/utils", "normalize_keyword")
        self.getSerpTool = self.dynamic_import(f"{base_path}/tools", "getSerpTool")
        self.OutputTool = self.dynamic_import(f"{base_path}/tools", "OutputTool")
        self.ClickAggregator = self.dynamic_import(f"{base_path}/tools", "ClickAggregator")
        
        self.s3_path = s3_path
        self.s3_bucket = s3_bucket
        self.good_kw_list = []
        self.attempt = 0
        self.successfull_pass = False
        self.running_week_day = tuple(int(day.strip()) for day in self.config['EXE']['RUNNING_WEEK_DAY'].split(','))

        self.code = code_in

        os.environ['SERPAPI_API_KEY'] = self.config['KEYS']['SERPAPI_API_KEY']
        
    def dynamic_import(self, module_name, function_name):
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, function_name)
    
    def adjust_path_for_local_policy(self, original_path):
        # Change the path to include './local_policy/' prefix by replacing '../' with './local_policy/'
        return original_path.replace('../', 'local_policy/')
    
    def _generate_keywords(self, max_attempts=15):
        attempts = 0
        while attempts < max_attempts:
            # Simulate keyword generation here
            new_words_check = [random.randint(10, 100) for _ in range(10)]  # Dummy search volumes
            all_pass = all(x >= int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']) for x in new_words_check)

            if all_pass:
                print("All keywords meet the threshold.")
                break
            else:
                print(f"Attempt {attempts + 1}: Not all keywords meet the threshold.")
                attempts += 1

        # Process the last set of keywords
        # Assuming `new_words_check` and `action_int_dic` are updated with the last attempt's keywords
        new_keywords_df = pd.DataFrame(
            [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in action_int_dic[branded].items() for kw in kws],
            columns=['Category', 'Keyword', 'Branded']
        )

        # Filter out keywords that meet the threshold
        new_keywords_df = new_keywords_df[new_keywords_df['Keyword'].apply(lambda kw: int(kw) >= int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']))]

        if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
            new_keywords_df.to_csv(self.generated_keyword_data, index=False)
        else:
            #full_path = self.save_dir + self.generated_keyword_data[2:]
            buffer = io.BytesIO()
            new_keywords_df.to_csv(buffer, index=False)
            buffer.seek(0)
            client = boto3.client('s3')
            client.put_object(Bucket=self.s3_bucket, Key=self.save_dir + self.generated_keyword_data, Body=buffer.getvalue())

        return new_keywords_df
    
    #def run(self, step = 0, start_month = '2023-06', end_month = '2024-09', google_account_id='1252903913'):
    def run(self,cb_get_kw_metrics,cb_exec_kw_plan,current_keyword_list):    
        
        if self.code == 0 and self.setting_day.weekday() not in self.running_week_day:
            return [], {}, int(2)
        
        
        #self.good_kw_list = []

        prompt_template = """
                You are a {6} ad keyword setting expert for Google search ads for {5} (you can search it on the internet). You will review specific keyword settings in {6} for {5}, 
                including the search keywords, their corresponding conversions, cost per conversion ('Cost/conv.'), and clicks.

                I would like you to determine the final keyword list by (make sure to process every step listed as follow!!):
                
                1. Finding all categories of the keywords and identifying the current keywords for each category.
                2. Use keyword_rule_example_search (the tool we prepare for you) to find the general good examples and rules for the keyword setting for another product and general rule.
                3. Use google_search (the tool we prepare for you) as much as you want to gain extensive insite about the positive attributes/features of {5} (use whole Japanese to search if it is a Japanese keyword setting) for which we are delivering ads.
                4. Use tools to find clicks per keyword and clicks per category. Taking the clicks into account, generate new keywords only for categories that will likely generate more traffic.
                5. By referring to the good example and rules, generate up to {0} more keywords in {6} for each category that you think are suitable, considering the attributes of {5}. Refer the following keywords as your new keyword: {4}. Make sure to not generate keywords that are already included in the category. Each keyword may contain multiple words, use spaces to separate them.
                6. Also generate {1} more categories with each category having {2} new keywords in {6}, that you think are suitable keywords for {5}. you can choose to use the following keywords as your new keyword: {4} . Each keyword may contain multiple words, use spaces to separate them.
                7. Generate two sets of keywords, one with the branding (individual product name or company name, depending on the context of the keyword) to target users who are looking for brand products, and others without the branding to target users who are looking for a overall good product. 
                8. Double check the following negative list : {3} as they have been proven not good for ad keywords as no one search the keywords in this list(search volumn check in Google ad API). These might be good for keywords but not good for ad keywords, as it is too long, or too specific for user search habit
                9. Doube check the whole new generated dictionary should not include overlapped keywords (same keyword in different category).
                10. Final Output should be one dictionary format ({{"Branded": {{Category 1: [Keywords..], Category 2: [Keywords..],...}}, "Non-Branded": {{...}}}}).  Each sub-dic () consists of two newly generated sets of {6} keywords for both existing and new categories (only newly generated keywords without the exsiting ones). The key is the category (you need to give an approperate category name for newly generated categories) and the value is a string list.  Only output the dictionary as the final answer without any extra words for parsing string to dic later.
                """
         
        prompt_template = """
                You are a {6} ad keyword setting expert for Google search ads for {5} (you can search it on the internet). You will review specific keyword settings in {6} for {5}, focusing on search keywords, conversions, cost per conversion ('Cost/conv.'), and clicks.

                Your task is to output a final keyword including "Branded" keywords should include specific product or company names, and  "Non-Branded" Keywords should focus on general qualities without brand names, by following each step below (make sure to go through all steps and then output the final answer):

                1. **Identify Categories**: List all keyword categories and their current keywords.
                2. **Gather Examples & Rules**: Use `keyword_rule_example_search` (tool provided) to find general examples and best practices for keyword settings.
                3. **Research Product Attributes**: Use `google_search` (tool provided) extensively to understand the key features of {5} (search in Japanese if needed).
                4. **Analyze Click Data**: Find clicks per keyword and category. Generate new keywords only for categories likely to drive more traffic.
                5. **Generate Additional Keywords**: Refer to the examples and rules to generate up to {0} new keywords in {6} for each existing category for both Branded type and Non-Branded type. Use {4} as possible new keywords, ensuring none are duplicates of existing ones. 
                6. **Add New Categories**: Generate {1} new categories with {2} keywords in {6} for each for both Branded type and Non-Branded type, using {4} for guidance. Ensure each keyword has spaces between words. 
                7. **Final Answer**: Output a **string** that resembles a dictionary format containing two main keys: "Branded" and "Non-Branded". The "Branded" set should include specific product or company names, while the "Non-Branded" set should focus on general qualities without brand names. 
                **Final Answer Format**: The final output should be **a single string** formatted like a dictionary (not actual JSON). Make sure the LLM outputs this as plain string text, not as a parsed dictionary:
                
                ```json
                {{
                "Branded": {{
                    "Category 1": ["Keyword 1", "Keyword 2", ...],
                    "Category 2": ["Keyword 1", "Keyword 2", ...]
                }},
                "Non-Branded": {{
                    "Category 1": ["Keyword 1", "Keyword 2", ...],
                    ...
                }}
                }}
                
                **Important**: final answer should be a string text using the format without additional text.
                """
                
        prompt_template = """
                You are a {6} ad keyword setting expert for Google search ads for {5} (you can search it on the internet). You will review specific keyword settings in {6} for {5}, focusing on search keywords, conversions, cost per conversion ('Cost/conv.'), and clicks.

                Your task is to output a final keyword including "Branded" keywords should include specific product or company names, and  "Non-Branded" Keywords should focus on general qualities without brand names, by following each step below (make sure to go through all steps and then output the final answer):

                1. **Identify Categories**: List all keyword categories and their current keywords.
                2. **Gather Examples & Rules**: Use `keyword_rule_example_search` (tool provided) to find general examples and best practices for keyword settings.
                3. **Research Product Attributes**: Use `google_search` (tool provided) extensively to understand the key features of {5} (search in Japanese if needed).
                4. **Analyze Click Data**: Find clicks per keyword and category. Generate new keywords only for categories likely to drive more traffic.
                5. **Generate New Keywords & Categories, then Format Output**:
                    - Use the gathered examples, rules, and research insights to generate new keywords.
                    - For each existing category, generate up to {0} additional keywords in {6} (both Branded and Non-Branded) using {4} as guidance. Avoid duplicates.
                    - Create {1} new categories, each with {2} keywords in {6} for both Branded and Non-Branded types using {4} for inspiration.
                    - Format the final result as a **single string** that resembles a dictionary with two main keys: "Branded" and "Non-Branded".
                    - The "Branded" section should include product or company-specific names, while the "Non-Branded" section should include general descriptive qualities without brand names.
                    - Ensure all keywords have spaces between words. you need to use output_tool tool to output your thought.
                
                **Final Answer Format**: The final output should be **a single string** formatted like a dictionary (not actual JSON). Make sure the LLM outputs this as plain string text, not as a parsed dictionary:
                
                ```json
                {{
                "Branded": {{
                    "Category 1": ["Keyword 1", "Keyword 2", ...],
                    "Category 2": ["Keyword 1", "Keyword 2", ...]
                }},
                "Non-Branded": {{
                    "Category 1": ["Keyword 1", "Keyword 2", ...],
                    ...
                }}
                }}
                
                **Important**: final answer should be a string text using the format without additional text. 
                """
                
        prompt_template = """
                You are a {6} ad keyword setting expert for Google search ads for {5} (you can search it on the internet). You will review specific keyword settings in {6} for {5}, focusing on search keywords, conversions, cost per conversion ('Cost/conv.'), and clicks.

                Your task is to output a final keyword including "Branded" keywords should include specific product names, and  "Non-Branded" Keywords should focus on general qualities without brand names, by following each step below (make sure to go through all steps and then output the final answer):

                1. **Identify Categories**: List all keyword categories and their current keywords.
                2. **Gather Examples & Rules**: Use `keyword_rule_example_search` (tool provided) to find general examples and best practices for keyword settings.
                3. **Research Product Attributes**: Use `google_search` (tool provided) extensively to understand the key features of {5} (search in Japanese if needed).
                4. **Analyze Click Data**: Find clicks per keyword and category. Generate new keywords only for categories likely to drive more traffic.
                5. **Generate New Keywords & Categories, then Format Output**:
                    - Use the gathered examples, rules, and research insights to generate new keywords.
                    - For each existing category, generate up to {0} additional keywords in {6} (both Branded and Non-Branded) using {4} as guidance. Avoid duplicates.
                    - Create {1} new categories, each with {2} keywords in {6} for both Branded and Non-Branded types using {4} for inspiration.
                    - Format the final result as a **single string** that resembles a dictionary with two main keys: "Branded" and "Non-Branded".
                    - The “Branded” section should include product names (sometimes with multiple names connected by “aka.” In such cases, you may need to generate combinations of product names and keywords). Meanwhile, the “Non-Branded” section should focus on general descriptive qualities without mentioning brand names.
                    - Ensure all keywords have spaces between words. you need to use output_tool tool to output your thought.
                
                **Final Answer Format**: The final output should be **a single string** formatted like a dictionary (not actual JSON). Make sure the LLM outputs this as plain string text, not as a parsed dictionary:
                
                ```json
                {{
                "Branded": {{
                    "Category 1": ["Keyword 1", "Keyword 2", ...],
                    "Category 2": ["Keyword 1", "Keyword 2", ...]
                }},
                "Non-Branded": {{
                    "Category 1": ["Keyword 1", "Keyword 2", ...],
                    ...
                }}
                }}
                
                **Important**: final answer should be a string text using the format without additional text. 
                """
        # to check whether there is file in s3 
        if self.setting_day.day <= 7:
            is_init = True
        else:
            is_init = False
        
        
        if ((self.setting_day.weekday() in self.running_week_day) and is_init and self.code == 0) or (self.code == 1 and is_init):        
        #if (self.setting_day.weekday() == 6 and self.setting_day.day <= 7 and self.code == 0) or (self.code == 1 and self.setting_day.day <= 7) :

            with open(self.rejected_keyword_list, 'r') as file:
                rejected_kw_list = [line.strip() for line in file]
            
            Aggrerator_init_flag = True
            
            KW_loader = CSVLoader(self.initial_keyword_data)
            df = pd.read_csv(self.initial_keyword_data)
            original_keywords = [item["keyword"] for item in current_keyword_list]
            #original_keywords = df["Keyword"].tolist()

            KW_retriever = self.customized_trend_retriever(KW_loader, str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

        
            # 2. define a retriever_tool
            KW_retriever_tool = create_retriever_tool(
                KW_retriever,
                str(self.config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
                str(self.config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
            )


            # 4. exampler tool
            exampler_loader = TextLoader(str(self.rule_data))
            exampler_retriever = self.customized_trend_retriever(exampler_loader, str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT'])) 

            # define a retriever_tool
            exampler_retriever_tool = create_retriever_tool(
                exampler_retriever,
                str(self.config['TOOL']['RULE_RETRIEVAL_NAME']),
                #'Search',
                str(self.config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
            )
            
            # search_tool = load_tools(["serpapi"])
            # search_tool[0].name = "google_search"
            
            # 3. Initilize LLM and the agent chain
            llm = AzureChatOpenAI(deployment_name=str(self.config['LLM']['deployment_name']), openai_api_version= str(self.config['LLM']['openai_api_version']), openai_api_key = str(self.config['KEYS']['OPENAI_GPT4O_API_KEY']), azure_endpoint = str(self.config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']), \
                                    temperature = float(self.config['LLM']['temperature']))
            prompt = hub.pull("hwchase17/react")
            
            if str(self.config["SETTING"]["language"]) == 'English':
                tools = [KW_retriever_tool,self.getSerpTool(region = "us", language = "en"),exampler_retriever_tool, self.OutputTool(), self.ClickAggregator(self.initial_keyword_data, self.config, self.s3_bucket, self.s3_path, Aggrerator_init_flag)]
            elif str(self.config["SETTING"]["language"]) == 'Japanese':
                tools = [KW_retriever_tool,self.getSerpTool(region = "jp", language = "ja"),exampler_retriever_tool, self.OutputTool(), self.ClickAggregator(self.initial_keyword_data, self.config, self.s3_bucket, self.s3_path, Aggrerator_init_flag)]
            agent_chain = create_react_agent(
                llm,
                tools,
                prompt
            )
            agent_executor = AgentExecutor(agent=agent_chain, tools=tools, return_intermediate_steps=True, verbose=True, handle_parsing_errors='Check your output and make sure it either uses: Correct Action/Action Input syntax, or Outputs the Final Answer directly after the thought in the form of  "Thought: I now know the final answer\nFinal Answer: the final answer to the original input question"')
            
            #print("Reflection is unabled")
            print("the first step")
  
            


            # Define the hyperparameters
            num_keywords_per_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY'])
            num_new_categories = int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])
            num_keywords_per_new_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_NEW_CATEGORY'])
            
            self.good_kw_list = [s for s in self.good_kw_list if s not in df["Keyword"].tolist()]
            
            rejected_keywords_string = ", ".join(rejected_kw_list[-30:])  # Converts list to string
            good_kw_string = ", ".join(self.good_kw_list)  # Converts list to string

            # 4. Process the first prompt
            # Define the prompt with placeholders for the hyperparameters
            
            # Format the prompt with the hyperparameters
            first_prompt = prompt_template.format(num_keywords_per_category, num_new_categories, num_keywords_per_new_category, rejected_keywords_string, good_kw_string, str(self.config["SETTING"]['product']), self.config["SETTING"]["language"])
            # 5. Output the first qustion and Run the agent chain

            
            print("Question: " + first_prompt)
            
            action_int_dic, scratch_pad, code_out = self.run_with_retries(agent_executor, first_prompt, int (self.config['LLM']['max_attempts']))
            
            if code_out == 4:
                return [], {}, int (4)
            
            if str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"])  == 'True':
                # Create a list of all strings in 'Branded'
                all_branded = []
                for key in action_int_dic["Branded"]:
                    all_branded.extend(action_int_dic["Branded"][key])

                # Create a list of all strings in 'Non-Branded'
                all_non_branded = []
                for key in action_int_dic["Non-Branded"]:
                    all_non_branded.extend(action_int_dic["Non-Branded"][key])

                if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                    keyword_list_needed = all_non_branded
                else: 
                    keyword_list_needed = all_branded
              
              
                # this should be replaced by the func. of Ascade san 
                if str(self.config["EXE"]["S3_DEPLOY"])  == 'False':
                    metrics = self.get_keyword_planner_metrics(
                        '1252903913',
                        keyword_list_needed,
                        '2023-06',
                        '2024-06',
                    )
                    # Wait for 10 seconds
                    time.sleep(10)
                    # Normalize keyword casing for accurate matching
                    #keyword_data_normalized = {k['Keyword'].lower(): k['Avg. monthly searches'] for k in metrics}
                    keyword_to_search_volume = {self.normalize_keyword(metric['Keyword']): metric['Avg. monthly searches'] for metric in metrics}
                    pdb.set_trace( )
                    # Extract 'Avg. monthly searches' for keywords in the list
                    #new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in keyword_list_needed]
                    new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword in keyword_list_needed]

                elif str(self.config["EXE"]["S3_DEPLOY"])  == 'True': 
                    search_result = cb_exec_kw_plan(keyword_list_needed, 12)
                    # Wait for 10 seconds
                    time.sleep(10)
                    # Normalize keyword casing for accurate matching
                    #keyword_data_normalized = {k['keyword'].lower(): k['avg_monthly_searches'] for k in search_result}
                    keyword_to_search_volume = {self.normalize_keyword(k['keyword']): k['avg_monthly_searches'] for k in search_result}
                    # Extract 'Avg. monthly searches' for keywords in the list
                    #new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in keyword_list_needed]
                    new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword in keyword_list_needed]
                else:
                    raise ValueError("Invalid value for S3_DEPLOY: " + str(self.config["EXE"]["S3_DEPLOY"]))
                
                
                
                #print('=== Keyword Planner metrics:')
                #for i, m in enumerate(metrics):
                    #print(f'[{i}]:')
                    #print(f'  Keyword: "{m["Keyword"]}"')
                    #print(f'  Avg. monthly searches: {m["Avg. monthly searches"]:,}')
                    #print(f'  Three month change: {m["Three month change"]}%')
                    #print(f'  YoY change: {m["YoY change"]}%')
                    #print(f'  Competition: {m["Competition"]}')
                    #print(f'  Top of page bid (low range): ¥{m["Top of page bid (low range)"]}')
                    #print(f'  Top of page bid (high range): ¥{m["Top of page bid (high range)"]}')
            
            elif str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"])  == 'False':
                new_words_check = [60, 70, 80, 90, 100, 100, 100, 100, 100]

            else:
                raise ValueError("Invalid value for SEARCH_VOLUMN_CHECK: " + str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"]))

            # Regular expression to find "Observation 1"
            # observation_pattern = r"Observation 1: (.+?)]\n"

            # Find the Observation 1 content
            # match = re.search(observation_pattern, scratch_pad, re.DOTALL)

            # if match:
            #     observation_1_str = match.group(1) + "]"
            #     # Convert string representation of list to an actual list
            #     observation_1_list = eval(observation_1_str)
            #     # Print or use the extracted list
            #     print(observation_1_list)
            # else:
            #     print("Observation 1 not found.")


            
            while self.attempt < int (self.config['LLM']['max_attempts']):
                # if all the element in new_words_check is over 50, break the loop
                if all(x >= int (self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']) for x in new_words_check):
                    self.successfull_pass = True
                    # add the new generated keywords to the /data/initial_KW.csv
                    # 1. covert the dic to dataphrame
                    new_keywords_df = pd.DataFrame(
                        [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in action_int_dic[branded].items() for kw in kws],
                        columns=['Category', 'Keyword', 'Branded']
                    )

                    # List of existing categories in the original DataFrame
                    existing_categories = df['Category'].unique()
                    # Determine if the category is old or new
                    new_keywords_df['Category Status'] = new_keywords_df['Category'].apply(
                        lambda x: 'deeper' if x in existing_categories else 'wider'
                    )
                    

                    # 2. merge the new_keywords_df with the original df
                    df = pd.concat([df, new_keywords_df], ignore_index=True)
                    # 3. replace Nah in click with 0
                    df['Clicks'] = df['Clicks'].fillna(0)
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        df = df[df['Branded'] == False]
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        df = df[df['Branded'] == True]
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))

                    # df['Clicks'] = [random.randint(1,100) for k in df.index]
                    
                    if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                        with open(self.rejected_keyword_list, 'w') as file:
                            for item in rejected_kw_list:
                                file.write("%s\n" % item)
                    elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                        s3_client = boto3.client('s3')
                        buffer=io.BytesIO()
                        # Write each keyword to the buffer as a new line
                        for item in rejected_kw_list:
                            buffer.write(f"{item}\n".encode("utf-8"))
                        # Move to the start of the StringIO buffer
                        buffer.seek(0)
                        # Upload the buffer content to S3
                        s3_client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.rejected_keyword_list, Body=buffer.getvalue())

                    if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                        df.to_csv(self.generated_keyword_data, index=False)
                    else:
                        #full_path = self.save_dir + self.generated_keyword_data[2:]
                        buffer=io.BytesIO()
                        df.to_csv(buffer, index=False)
                        body=buffer.getvalue()
                        client=boto3.client('s3')
                        client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.generated_keyword_data, Body=body)
                    
                    break


                else:
                    for i in range (len(new_words_check)):
                        if new_words_check[i] < int (self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']):
                            # add the low search check new words to the tried_kw_list
                            rejected_kw_list.append(str (keyword_list_needed[i]))
                            print ("The new words whose search check is less than 5 is: " + str (keyword_list_needed[i]))
                            # save the rejected_kw_list to a file
                            if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                                with open(self.rejected_keyword_list, 'w') as file:
                                    for item in rejected_kw_list:
                                        file.write("%s\n" % item)
                            elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                                s3_client = boto3.client('s3')
                                buffer=io.BytesIO()
                                # Write each keyword to the buffer as a new line
                                for item in rejected_kw_list:
                                    buffer.write(f"{item}\n".encode("utf-8"))
                                # Move to the start of the StringIO buffer
                                buffer.seek(0)
                                # Upload the buffer content to S3
                                s3_client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.rejected_keyword_list, Body=buffer.getvalue())

                                # Clear the buffer after uploading to reuse it
                                buffer.seek(0)
                                buffer.truncate()
                                
                        else: 
                            # add keywords to the self.good_kw_list
                            self.good_kw_list.append(str (keyword_list_needed[i]))
                            self.good_kw_list = list(set(self.good_kw_list))
                    
                    self.attempt += 1
                    
                    
                    return self.run(cb_get_kw_metrics,cb_exec_kw_plan,current_keyword_list)
                        
                    #print("Next Round, the rejected low search keywords are: " + str(rejected_kw_list))
            # response = agent_chain ({"input":  first_prompt})
            if not self.successfull_pass:
                # Flatten the dictionary into a list of keywords without considering the category
                #keywords_flat_list = [kw for sublist in action_int_dic["Non-Branded"].values() for kw in sublist]
                # Flatten the dictionary into a list of tuples (category, keyword)
                if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                    keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Non-Branded"].items() for kw in kws]
                else: 
                    keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Branded"].items() for kw in kws]
                # keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Non-Branded"].items() for kw in kws]
                
                # Filter keywords by matching them with their corresponding search amount
                #good_keywords = [kw for kw, check in zip(keywords_flat_list, new_words_check) if check > int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD'])]
                good_keywords = [(cat, kw) for (cat, kw), check in zip(keywords_with_categories, new_words_check) if check > int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD'])]

                if good_keywords:
                    # Create DataFrame from good keywords
                    new_keywords_df =  pd.DataFrame(good_keywords, columns=['Category', 'Keyword'])
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        new_keywords_df['Branded'] = False  # Adding a constant column for branding
                    else: 
                        new_keywords_df['Branded'] = True
                    
                    existing_categories = df['Category'].unique()
                    new_keywords_df['Category Status'] = new_keywords_df['Category'].apply(
                        lambda x: 'deeper' if x in existing_categories else 'wider'
                    )
                    
                    df = pd.concat([df, new_keywords_df], ignore_index=True)
                    # only keep the colomn Branded with the value of 'Non-Branded'
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        df = df[df['Branded'] == False]
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        df = df[df['Branded'] == True]
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    
                    # Remove rows with duplicate values in the 'Keyword' column, keeping the first occurrence
                    df = df.drop_duplicates(subset='Keyword', keep='first')
                    
                    df['Clicks'] = df['Clicks'].fillna(0)
                    #df['Clicks'] = [random.randint(1, 100) for _ in df.index]  # Optional, simulate clicks

                if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                    df.to_csv(self.generated_keyword_data, index=False)
                else:
                    #full_path = self.save_dir + self.generated_keyword_data[2:]
                    buffer = io.BytesIO()
                    df.to_csv(buffer, index=False)
                    body = buffer.getvalue()
                    client = boto3.client('s3')
                    client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.generated_keyword_data, Body=body)
                
                self.good_kw_list = []
            
                                        
                
            # return action_int_list  
        

        elif ((self.setting_day.weekday() in self.running_week_day) and (not is_init) and self.code == 0 ) or ( (not is_init) and self.code == 1):
        #elif (self.setting_day.weekday() == 6 and self.setting_day.day > 7 and self.code == 0 ) or (self.setting_day.day > 7 and self.code == 1):
            
            # get data from call back function
            # df = cb_get_kw_metrics(self.observation_period)
            
            if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':   
                with open(self.rejected_keyword_list, 'r') as file:
                    rejected_kw_list = [line.strip() for line in file]
            elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                # Initialize the S3 client
                s3_client = boto3.client('s3')
                
                # Get the object from the bucket
                obj = s3_client.get_object(Bucket=self.s3_bucket, Key=self.save_dir+self.rejected_keyword_list)
                
                # Read data from the S3 object
                data = obj['Body'].read().decode('utf-8')
                
                # Split data into lines and strip whitespace
                rejected_kw_list = [line.strip() for line in data.splitlines()]
            
            Aggrerator_init_flag = False
            
            if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                df = pd.read_csv(self.generated_keyword_data)
                KW_loader = CSVLoader(self.generated_keyword_data)
                KW_retriever = self.customized_trend_retriever(KW_loader, str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))
                original_keywords = df["Keyword"].tolist()
            elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                # get the dataphrame 
                df = cb_get_kw_metrics(self.observation_period)
                # only keep the colomn with name  'keyword' and 'clicks'
                kw_df = df[['keyword', 'clicks']]
                original_keywords = [item["keyword"] for item in current_keyword_list]
                #original_keywords = df["keyword"].tolist()
                # Initialize S3 client
                s3_read_client = boto3.client('s3')
                
                # Get the object
                response = s3_read_client.get_object(Bucket=self.s3_bucket, Key=self.save_dir+self.generated_keyword_data)

                # Read the data into a pandas DataFrame
                df_ref = pd.read_csv(io.BytesIO(response['Body'].read()))
                df_ref = df_ref[df_ref['Keyword'].isin(df['keyword'].unique())]

                # merge the two dataframes on 'Keyword' and 'Category'
                merged_df = kw_df.merge(df_ref, left_on="keyword", right_on="Keyword", how="left")

                # Select and rename the columns as needed
                df = merged_df[['keyword', 'clicks', 'Category', 'Branded', 'Category Status']]
                
                # change the colomn name from 'keyword' to 'Keyword'
                
                df = df.rename(columns={'keyword': 'Keyword'})
                # change the colomn name from 'clicks' to 'Clicks'
                df = df.rename(columns={'clicks': 'Clicks'})


                # save it to a new csv file in the s3
                full_path = self.save_dir + self.generated_keyword_data[2:]
                buffer=io.BytesIO()
                df.to_csv(buffer, index=False)
                body=buffer.getvalue()
                client=boto3.client('s3')
                client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.generated_keyword_data, Body=body)
                #KW_loader = CSVLoader(self.generated_keyword_data)
                KW_loader = S3FileLoader(bucket=self.s3_bucket, key=self.save_dir+self.generated_keyword_data)
                KW_retriever = self.customized_trend_retriever(KW_loader, str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))
        
            # 2. define a retriever_tool
            KW_retriever_tool = create_retriever_tool(
                KW_retriever,
                str(self.config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
                #'Search',
                str(self.config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
            )

            # 3. rule tool 
            # rule_loader = TextLoader(str(self.config['FILE']['DOMAIN_KNOWLEDGE_FILE']))

            # 4. exampler tool
            exampler_loader = TextLoader(str(self.rule_data))
            exampler_retriever = self.customized_trend_retriever(exampler_loader, str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),  str(self.config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT'])) 

            # define a retriever_tool
            exampler_retriever_tool = create_retriever_tool(
                exampler_retriever,
                str(self.config['TOOL']['RULE_RETRIEVAL_NAME']),
                #'Search',
                str(self.config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
            )
            
            # search_tool = load_tools(["serpapi"])
            #search = SerpAPIWrapper()
            # ロードしたツールの中から一番目のものの名前を変更
            # https://book.st-hakky.com/data-science/agents-of-langchain/
            # search_tool[0].name = "google_search"
            
            # 3. Initilize LLM and the agent chain
            llm = AzureChatOpenAI(deployment_name=str(self.config['LLM']['deployment_name']), openai_api_version=str(self.config['LLM']['openai_api_version']), openai_api_key = str(self.config['KEYS']['OPENAI_GPT4O_API_KEY']), azure_endpoint = str(self.config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']), \
                                    temperature = float(self.config['LLM']['temperature']))
            
            prompt = hub.pull("hwchase17/react")
            
            #from tools import CustomCounterTool2

            if str(self.config["SETTING"]["language"]) == 'English':
                tools = [KW_retriever_tool,self.getSerpTool(region = "us", language = "en"),exampler_retriever_tool, self.OutputTool(), self.ClickAggregator(self.generated_keyword_data, self.config, self.s3_bucket, self.s3_path, Aggrerator_init_flag)]
            elif str(self.config["SETTING"]["language"]) == 'Japanese':
                tools = [KW_retriever_tool,self.getSerpTool(region = "jp", language = "ja"),exampler_retriever_tool, self.OutputTool(), self.ClickAggregator(self.generated_keyword_data, self.config, self.s3_bucket, self.s3_path, Aggrerator_init_flag)]
            #tools = [KW_retriever_tool,self.getSerpTool(),exampler_retriever_tool, self.ClickAggregator(self.generated_keyword_data, self.config, self.s3_bucket, self.s3_path, Aggrerator_init_flag)]
            agent_chain = create_react_agent(
                llm,
                tools,
                prompt
            ) 
            agent_executor = AgentExecutor(agent=agent_chain, tools=tools, return_intermediate_steps=True, verbose=True, handle_parsing_errors='Check your output and make sure it either uses: Correct Action/Action Input syntax, or Outputs the Final Answer directly after the thought in the form of  "Thought: I now know the final answer\nFinal Answer: the final answer to the original input question"')
                
            # need to find the click growth of each keyword
            # read the whole_kw.csv
            #df_whole = pd.read_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv')
            
            # Merge the two dataframes on 'Keyword' and 'Category'
            #merged_df = pd.merge(df, df_whole, on=['Keyword', 'Category'], suffixes=('_df1', '_df2'))

            # Calculate the difference in Clicks
            #merged_df['Clicks Difference'] = merged_df['Clicks_df2'] - merged_df['Clicks_df1']

            # Group by 'Category Status' and calculate the mean Clicks Difference
            #category_status_mean_difference = merged_df.groupby('Category Status')['Clicks Difference'].mean().reset_index()
            
            # Cast 'Category Status' colomn to type string
            df['Category Status'] = df['Category Status'].astype(str)
            
            # Filtering and summing clicks for 'wider' and 'deeper' categories
            wider_click_difference = max (1, df[df['Category Status'] == 'wider']['Clicks'].sum())
            deeper_click_difference = max (1, df[df['Category Status'] == 'deeper']['Clicks'].sum()) 
     
            # Filter the dataframe to get the click difference for 'deeper'
            #deeper_click_difference = category_status_mean_difference[category_status_mean_difference['Category Status'] == 'deeper']['Clicks Difference'].values[0]

            # Filter the dataframe to get the click difference for 'wider'
            #wider_click_difference = category_status_mean_difference[category_status_mean_difference['Category Status'] == 'wider']['Clicks Difference'].values[0]

            print("Clicks Difference for 'deeper':", deeper_click_difference)
            print("Clicks Difference for 'wider':", wider_click_difference)

            # Define the hyperparameters
            # Calculate the total sum from the configuration
            total_original_sum = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY']) + int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])

            # Calculate the total difference
            total_difference = wider_click_difference + deeper_click_difference

            # Calculate the proportion of each click difference
            wider_proportion = wider_click_difference / total_difference
            deeper_proportion = deeper_click_difference / total_difference

            if (str(self.config['KEYWORD']['GENERATION_DYNAMICS']) == 'True'):
                # Allocate the total sum with a minimum threshold of 1
                num_keywords_per_category = max(1, int(total_original_sum * wider_proportion))
                num_new_categories = max(1, int(total_original_sum * deeper_proportion))
            
            elif (str(self.config['KEYWORD']['GENERATION_DYNAMICS']) == 'False'):
            
                num_keywords_per_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY'])
                num_new_categories = int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])
            

            # Adjust for rounding errors and maintain the sum
            current_sum = num_keywords_per_category + num_new_categories
            if current_sum != total_original_sum:
                difference = total_original_sum - current_sum
                # Adjust the larger proportion to keep both values above 0
                if num_keywords_per_category > num_new_categories:
                    num_keywords_per_category += difference
                else:
                    num_new_categories += difference
            num_keywords_per_new_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_NEW_CATEGORY'])

            self.good_kw_list = [s for s in self.good_kw_list if s not in df["Keyword"].tolist()]
            
            rejected_keywords_string = ", ".join(rejected_kw_list[-30:])  # Converts list to string
            good_kw_string = ", ".join(self.good_kw_list)  # Converts list to string

            # 4. Process the first prompt
            # Define the prompt with placeholders for the hyperparameters
            
            # Format the prompt with the hyperparameters
            first_prompt = prompt_template.format(num_keywords_per_category, num_new_categories, num_keywords_per_new_category, rejected_keywords_string, good_kw_string, str(self.config["SETTING"]['product']), self.config["SETTING"]["language"])
            # 5. Output the first qustion and Run the agent chain

            
            print("Question: " + first_prompt)
            
            action_int_dic, _, code_out = self.run_with_retries (agent_executor, first_prompt, int (self.config['LLM']['max_attempts']))
            
            if code_out == 4:
                return [], {}, int (4)
            # transfer the dic to list by dumping the key
            #new_words_list = list(action_int_dic.values())

            # Initialize an empty list to hold all values
            new_words_list = []
            # Iterate over the dictionary and extend the list with each value list
            for key in action_int_dic:
                new_words_list.extend(action_int_dic[key])


            # this should be replaced by the func. of Ascade san 
            if str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"])  == 'True':
                # Create a list of all strings in 'Branded'
                all_branded = []
                for key in action_int_dic["Branded"]:
                    all_branded.extend(action_int_dic["Branded"][key])

                # Create a list of all strings in 'Non-Branded'
                all_non_branded = []
                for key in action_int_dic["Non-Branded"]:
                    all_non_branded.extend(action_int_dic["Non-Branded"][key])


                # this should be replaced by the func. of Ascade san 
                if str(self.config["EXE"]["S3_DEPLOY"])  == 'False':
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        metrics = self.get_keyword_planner_metrics(
                            '1252903913',
                            all_non_branded,
                            '2023-06',
                            '2024-06',
                        )
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        metrics = self.get_keyword_planner_metrics(
                            '1252903913',
                            all_branded,
                            '2023-06',
                            '2024-06',
                        )
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # Wait for 10 seconds
                    time.sleep(10)
                    
                    # Normalize keyword casing for accurate matching
                    #keyword_data_normalized = {k['Keyword'].lower(): k['Avg. monthly searches'] for k in metrics}
                    # Create a dictionary for lookup with normalized keys
                    keyword_to_search_volume = {self.normalize_keyword(metric['Keyword']): metric['Avg. monthly searches'] for metric in metrics}
                    
                    # Extract 'Avg. monthly searches' for keywords in the list
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        #new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]
                        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword in all_non_branded]

                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        #new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_branded]
                        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword in all_branded]
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    #new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]

                elif str(self.config["EXE"]["S3_DEPLOY"])  == 'True': 
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        search_result = cb_exec_kw_plan(all_non_branded, 12)
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        search_result = cb_exec_kw_plan(all_branded, 12)
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    #search_result = cb_exec_kw_plan(all_non_branded, 12)
                    # Wait for 10 seconds
                    time.sleep(10)
                    # Normalize keyword casing for accurate matching
                    #keyword_data_normalized = {k['keyword'].lower(): k['avg_monthly_searches'] for k in search_result}
                    keyword_to_search_volume = {self.normalize_keyword(k['keyword']): k['avg_monthly_searches'] for k in search_result}
                    
                    # Extract 'Avg. monthly searches' for keywords in the list
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        #new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]
                        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword in all_non_branded]
                        
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        #new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_branded]
                        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword in all_branded]
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    #new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]
                
                
              
                
            
            elif str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"])  == 'False':
                new_words_check = [60, 70, 80, 90, 100, 100, 100, 100, 100]

            else:
                raise ValueError("Invalid value for SEARCH_VOLUMN_CHECK: " + str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"]))
            
            
            
            
            while self.attempt < int (self.config['LLM']['max_attempts']):
                # if all the element in new_words_check is over 50, break the loop
                if all(x >= int (self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']) for x in new_words_check):
                    self.successfull_pass = True
                    # add the new generated keywords to the /data/initial_KW.csv
                    # 1. covert the dic to dataphrame
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        new_keywords_df = pd.DataFrame(
                            [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in action_int_dic[branded].items() for kw in kws],
                            columns=['Category', 'Keyword', 'Branded']
                        )
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        new_keywords_df = pd.DataFrame(
                            [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in action_int_dic[branded].items() for kw in kws],
                            columns=['Category', 'Keyword', 'Branded']
                        )
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    #new_keywords_df = pd.DataFrame(
                        #[(k, kw, branded == "Non-Branded") for branded in ["Branded", "Non-Branded"] for k, kws in action_int_dic[branded].items() for kw in kws],
                        #columns=['Category', 'Keyword', 'Branded']
                    #)

                    # List of existing categories in the original DataFrame
                    existing_categories = df['Category'].unique()
                    # Determine if the category is old or new
                    new_keywords_df['Category Status'] = new_keywords_df['Category'].apply(
                        lambda x: 'deeper' if x in existing_categories else 'wider'
                    )
                    

                    # 2. merge the new_keywords_df with the original df
                    df = pd.concat([df, new_keywords_df], ignore_index=True)
                    # 3. replace Nah in click with 0
                    df['Clicks'] = df['Clicks'].fillna(0)
                    # only keep the colomn Branded with the value of 'Non-Branded'
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        df = df[df['Branded'] == False]
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        df = df[df['Branded'] == True]
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    #df = df[df['Branded'] == False]
                    df = df.drop_duplicates(subset='Keyword', keep='first')

                    if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                        with open(self.rejected_keyword_list, 'w') as file:
                            for item in rejected_kw_list:
                                file.write("%s\n" % item)
                    elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                        s3_client = boto3.client('s3')
                        buffer=io.BytesIO()
                        # Write each keyword to the buffer as a new line
                        for item in rejected_kw_list:
                            buffer.write(f"{item}\n".encode("utf-8"))
                        # Move to the start of the StringIO buffer
                        buffer.seek(0)
                        # Upload the buffer content to S3
                        s3_client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.rejected_keyword_list, Body=buffer.getvalue())

                    if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                        df.to_csv(self.generated_keyword_data, index=False)
                    else:
                        #full_path = self.save_dir + self.generated_keyword_data[2:]
                        buffer=io.BytesIO()
                        df.to_csv(buffer, index=False)
                        body=buffer.getvalue()
                        client=boto3.client('s3')
                        client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.generated_keyword_data, Body=body)
                    
                    break


                else:
                    for i in range (len(new_words_check)):
                        if new_words_check[i] < int (self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']):
                            # add the low search check new words to the tried_kw_list
                            if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                                rejected_kw_list.append(str (all_non_branded[i]))
                                print ("The new words whose search check is less than 5 is: " + str (all_non_branded[i]))
                            elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                                rejected_kw_list.append(str (all_branded[i]))
                                print ("The new words whose search check is less than 5 is: " + str (all_branded[i]))
                            else:
                                raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                            #rejected_kw_list.append(str (all_non_branded[i]))
                            
                            # save the rejected_kw_list to a file
                            if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                                with open(self.rejected_keyword_list, 'w') as file:
                                    for item in rejected_kw_list:
                                        file.write("%s\n" % item)
                            elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                                s3_client = boto3.client('s3')
                                buffer=io.BytesIO()
                                # Write each keyword to the buffer as a new line
                                for item in rejected_kw_list:
                                    buffer.write(f"{item}\n".encode("utf-8"))
                                # Move to the start of the StringIO buffer
                                buffer.seek(0)
                                # Upload the buffer content to S3
                                s3_client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.rejected_keyword_list, Body=buffer.getvalue())

                                # Clear the buffer after uploading to reuse it
                                buffer.seek(0)
                                buffer.truncate()
                                
                        else: 
                            # add keywords to the self.good_kw_list
                            if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                                self.good_kw_list.append(str (all_non_branded[i]))
                                self.good_kw_list = list(set(self.good_kw_list))
                            elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                                self.good_kw_list.append(str (all_branded[i]))
                                self.good_kw_list = list(set(self.good_kw_list))
                            else:
                                raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                            #self.good_kw_list.append(str (all_non_branded[i]))
                    
                    self.attempt += 1
                    
                    
                    return self.run(cb_get_kw_metrics,cb_exec_kw_plan,current_keyword_list)
                        
                    #print("Next Round, the rejected low search keywords are: " + str(rejected_kw_list))
            # response = agent_chain ({"input":  first_prompt})
            
            if not self.successfull_pass:
                # Flatten the dictionary into a list of keywords without considering the category
                #keywords_flat_list = [kw for sublist in action_int_dic["Non-Branded"].values() for kw in sublist]
                # Flatten the dictionary into a list of tuples (category, keyword)
                if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                    keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Non-Branded"].items() for kw in kws]
                elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                    keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Branded"].items() for kw in kws]
                else:
                    raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                #keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Non-Branded"].items() for kw in kws]
                
                # Filter keywords by matching them with their corresponding search amount
                #good_keywords = [kw for kw, check in zip(keywords_flat_list, new_words_check) if check > int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD'])]
                good_keywords = [(cat, kw) for (cat, kw), check in zip(keywords_with_categories, new_words_check) if check > int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD'])]

                if good_keywords:
                    # Create DataFrame from good keywords
                    new_keywords_df =  pd.DataFrame(good_keywords, columns=['Category', 'Keyword'])
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        new_keywords_df['Branded'] = False
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        new_keywords_df['Branded'] = True
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # new_keywords_df['Branded'] = "Non-Branded"  # Adding a constant column for branding
                    
                    existing_categories = df['Category'].unique()
                    new_keywords_df['Category Status'] = new_keywords_df['Category'].apply(
                        lambda x: 'deeper' if x in existing_categories else 'wider'
                    )
                    
                    df = pd.concat([df, new_keywords_df], ignore_index=True)
                    # only keep the colomn Branded with the value of 'Non-Branded'
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'False':
                        df = df[df['Branded'] == False]
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"])  == 'True':
                        df = df[df['Branded'] == True]
                    else:
                        raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # Remove rows with duplicate values in the 'Keyword' column, keeping the first occurrence
                    df = df.drop_duplicates(subset='Keyword', keep='first')

                    df['Clicks'] = df['Clicks'].fillna(0)
                    #df['Clicks'] = [random.randint(1, 100) for _ in df.index]  # Optional, simulate clicks

                if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                    df.to_csv(self.generated_keyword_data, index=False)
                else:
                    #full_path = self.save_dir + self.generated_keyword_data[2:]
                    buffer = io.BytesIO()
                    df.to_csv(buffer, index=False)
                    body = buffer.getvalue()
                    client = boto3.client('s3')
                    client.put_object(Bucket=self.s3_bucket, Key=self.save_dir+self.generated_keyword_data, Body=body)

                
                self.good_kw_list = []
            
            
            
            
            
            
            
            
            
            
            # # if all the element in new_words_check is over 50, break the loop
            # if all(x >= 50 for x in new_words_check):
                
            #     # add the new generated keywords to the /data/initial_KW.csv
            #     # 1. covert the dic to dataphrame
            #     new_keywords_df = pd.DataFrame(
            #         [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in action_int_dic[branded].items() for kw in kws],
            #         columns=['Category', 'Keyword', 'Branded']
            #     )
                
            #     # List of existing categories in the original DataFrame
            #     existing_categories = df['Category'].unique()
            #     # Determine if the category is old or new
            #     new_keywords_df['Category Status'] = new_keywords_df['Category'].apply(
            #         lambda x: 'deeper' if x in existing_categories else 'wider'
            #     )

            #     # 2. merge the new_keywords_df with the original df
            #     df = pd.concat([df, new_keywords_df], ignore_index=True)
            #     # 3. replace Nah in click with 0
            #     df['Clicks'] = df['Clicks'].fillna(0)

            #     #df['Clicks'] = [random.randint(1,100) for k in df.index]

            #     # 4. save the new df to the csv file
            #     #df.to_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv', index=False)


            #     # change from dic to list
            #     # action_int_list = list(action_int_dic.values())
            #     if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
            #         df.to_csv(self.config["SETTING"]['generated_keyword_data'], index=False)
            #     elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
            #         full_path = self.save_dir + self.config["SETTING"]['generated_keyword_data'][2:]
            #         buffer=io.BytesIO()
            #         df.to_csv(buffer, index=False)
            #         body=buffer.getvalue()
            #         client=boto3.client('s3')
            #         client.put_object(Bucket=self.s3_bucket, Key=full_path, Body=body)
            #     else:
            #         raise ValueError("Invalid value for S3_DEPLOY: " + str(self.config["EXE"]["S3_DEPLOY"]))


                
            # else:
            #     for i in range (len(new_words_check)):
            #         if new_words_check[i] < int (self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']):
            #             # add the low search check new words to the tried_kw_list
            #             rejected_kw_list.append(str (all_non_branded[i]))
            #             print ("The new words whose search check is less than 5 is: " + str (all_non_branded[i]))
            #             # save the rejected_kw_list to a file
                    
            #             if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
            #                 with open(self.rejected_keyword_list, 'w') as file:
            #                     for item in rejected_kw_list:
            #                         file.write("%s\n" % item)
            #             elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
            #                 s3_client = boto3.client('s3')
            #                 buffer=io.BytesIO()
            #                 # Write each keyword to the buffer as a new line
            #                 for item in rejected_kw_list:
            #                     buffer.write("%s\n" % item)
            #                 # Move to the start of the StringIO buffer
            #                 buffer.seek(0)
            #                 # Upload the buffer content to S3
            #                 s3_client.put_object(Bucket=self.s3_bucket, Key=self.s3_path + self.rejected_keyword_list[2:], Body=buffer.getvalue())

            #                 # Clear the buffer after uploading to reuse it
            #                 buffer.seek(0)
            #                 buffer.truncate()
            #         else: 
            #             # add keywords to the self.good_kw_list
            #             self.good_kw_list.append(str (all_non_branded[i]))
            #     return self.run(cb_get_kw_metrics,cb_exec_kw_plan)
            self.attempt = 0        
            print("Next Round, the rejected low search keywords are: " + str(rejected_kw_list))
            # response = agent_chain ({"input":  first_prompt})
            
            # return action_int_list
            
            # return action_int_list
        else:
            return [], {}, int (2)
        # df = df[df['Branded'] == False]
        keywords_list = df["Keyword"].tolist()
        # Find the intersection of both lists
        common_items = set(keywords_list) & set(original_keywords)

        # Remove common items from both lists
        new_keyword_list = [item for item in keywords_list if item not in common_items]
        
        unique_keywords = list(set(new_keyword_list))
        category_list = df["Category"].tolist()
        return unique_keywords, category_list, code_out

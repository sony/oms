import ast
import unicodedata
import numpy as np
from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from collections import Counter
import json

from typing import List, Any, Dict, Optional, Type, Tuple
import re
from langchain_community.utilities.serpapi import SerpAPIWrapper
import difflib
from transformers import AutoTokenizer

import pickle
import os
import pandas as pd
import io
import random
import pdb 
from langchain_openai import AzureChatOpenAI
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

import pandas as pd
import io
import time
import importlib.util




nbz_product_info = "PC LCMは新OS Windows 11へのスムーズな切り替えをアシストするNURO Bizのキッティングサービスです。" \
                   "Windows 10は 2025年10月14日でサポートが終了します。このままWindows 10を利用していると、①情報漏洩などのセキュリティリスクが高まる、" \
                   "②OSの不具合が修正されないままに、③Windows10に対応しない周辺機器やソフトウェアが使用不可に、といったリスクがあります。" \
                   "しかし、現実としては、リソース不足で移行準備や移行が進まない、Windows 11への移行計画が定まらない、人員やスペースのリソース確保ができない、" \
                   "在宅や支社の従業員が多くPC管理が行き届かない、といった課題があります。" \
                   "そこで、20,000社以上の企業のネットワークをサポートしてきたNURO BizがWindows 10からのOSアップグレードまたはWindows 11搭載PC購入をサポートします。" \
                   "SBNのキッティングサービスは、業務用PCのソフト・アプリのインストール、ハード構成変更などの設定を代行。" \
                   "お客様のご要望に応じて、キッティングの手順や内容はカスタマイズ可能です。マスター端末作成サービス、複製、個別設定を行います。" \
                   "お客様の事業環境に合わせ、最適な移行計画をPC購入からオーダーメイドのキッティングサービスでご提案します。" \
                   "キッティングセンターにてドライバやアプリのインストール、PCの設定を実施。Windows 11の一斉導入の際も、すぐに業務を開始いただける状態で納品します。" \
                   "サービス対応エリアは日本全国対応可能です。PCの販売はキッティングをご希望のお客様にのみ可能です。" \
                   "新規でご発注の場合は最低10台から、追加の場合は1台から対応可能です。"


easy_predictive_info =  "Easy Predictive Analytics" \
                   "- Unleash the Potential of Your Data" \
                   "- Harness the power of predictive analytics with ease" \
                   "- Trusted by hundreds of Japanese companies" \
                   "- Try Easy Predictive Analytics free for 7 days" \
                   "- Trusted track record. Used in over 50 papers" \
                   "Features of Easy Predictive Analytics" \
                   "Easy Predictive Analytics is a user-friendly predictive analytics tool that works on a web browser, designed to help businesses make data-driven decisions. It automatically performs advanced predictive analytics with just a few clicks." \
                   "You can perform predictive analytics without requiring specialized skills. For experts, the process of predictive modeling becomes significantly more efficient." \
                   "- Feature 1: Simple and easy to use" \
                   "- Feature 2: High-precision predictions through automatic modeling" \
                   "- Feature 3: Users can understand the rationale behind predictions" \
                   "- Feature 4: Enables collaborative work among multiple personnel" \
                   "Case Study 1:" \
                   "A simple data analytics tool that helps busy researchers" \
                   "With your data and ideas, Easy Predictive Analytics can easily perform advanced data analytics to obtain new results." \
                   "Challenge: Data is available, but there isn’t enough time to study it, and data analytics is not your specialty." \
                   "Introduction: Easy Predictive Analytics automatically performs advanced predictive analytics with just a few clicks if the data is prepared." \
                   "Effect: With Easy Predictive Analytics, the customer could try new research approaches that were previously impossible, leading to new results that have been compiled into papers." \
                   "Case Study 2:" \
                   "Improving manufacturing efficiency through predictive analytics of product characteristics" \
                   "By predicting the characteristics of the end product in the early stages of manufacturing, Easy Predictive Analytics can help adjust subsequent processes to improve product yield." \
                   "Challenge: Predict the characteristics after the final process based on the intermediate manufacturing status to determine the need for adjustments in subsequent processes." \
                   "Introduction: Based on hundreds of data points obtained from an early-stage process, Easy Predictive Analytics was used to predict the characteristics of the final product." \
                   "Effect: Easy Predictive Analytics was able to predict with sufficiently high accuracy for practical use. This is expected to lead to improvements in manufacturing efficiency." \
                   "Case Study 3:" \
                   "Efficiently approach customers with needs" \
                   "Use Easy Predictive Analytics to predict the strength of each customer’s needs and prioritize approaching customers with high needs." \
                   "Challenge: Experience and intuition are used to determine the specific needs of potential customers, without utilizing data." \
                   "Introduction: Easy Predictive Analytics was used to predict the probability of customers purchasing a service based on customer data. The higher this probability, the more likely the customer has a need for the service." \
                   "Effect: This has enabled efficient approaches to potential customers with specific needs. The sales department employs the data analytics tool because it is easy to use." \
                   "Plan" \
                   "$399/month Billed monthly" \
                   "One simple plan. Enjoy full access to all features necessary for predictive analytics." \
                   "Main functions:" \
                   "- Predictive analytics functions: Probability prediction, numerical prediction, class categorization" \
                   "- Prediction contribution analysis" \
                   "- Up to three users can collaborate by sharing the data and results" \
                   "- Prediction API deployment function" \
                   "How to use Easy Predictive Analytics: " \
                   "There is no need to prepare a large amount of data from the beginning for Easy Predictive Analytics." \
                   "Easy Predictive Analytics can easily make high-precision predictions using existing data with just a few clicks." \
                   "- Step1: Prepare tabular data" \
                   "- Step2: Load the data into Easy Predictive Analytics by dragging and dropping" \
                   "- Step3: Set the prediction targets and click the Create Model button!"

class CounterInput(BaseModel):
    in_str: str = Field(
        description="A List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]. Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy.")


class Phrase_checker(BaseModel):
    in_str: str = Field(description="A List of lists composed in the form: [sentence1, sentence2, ...].")

def is_similar(str1, str2, threshold):
    """Return True if str1 and str2 are similar enough."""
    similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
    return similarity >= threshold

class CustomCounterTool(BaseTool):
    name: str = "character_counter"
    description: str = "A character counter. Useful for counting the number of characters in a sentence. Takes as input a List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]. \
        Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy."
    args_schema: Type[BaseModel] = CounterInput
    return_direct: bool = False

    def _run(
            self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        """Returns the number of characters in each input sentence."""
        return_str = ""
        for sent in in_sent:
            c_count = count_chars(sent[0])
            limit = int(sent[1])
            return_str += f"{sent[0]}: {c_count}/{sent[1]} characters"
            if c_count > limit:
                return_str += " (Too long)\n"
            elif c_count < limit // 2:
                return_str += " (Too short)\n"
            else:
                return_str += "\n"
        return return_str


class CustomCounterTool2(BaseTool):
    name: str = "character_counter"
    description: str = "A character counter. Useful for counting the number of characters in a sentence. Takes as input a List of lists composed in the form: [[sentence, character limit], [sentence, character limit],...]. \
        Sentence is the sentence to count the characters of, and character limit is the character limit the count should satisfy."
    args_schema: Type[BaseModel] = CounterInput
    return_direct: bool = False

    def _run(
            self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        """Returns the number of characters in each input sentence."""
        return_str = ""
        for sent in in_sent:
            c_count = count_chars(sent[0])
            return_str += f"{sent[0]}: {c_count}/{sent[1]} characters"
            return_str += "\n"
        return return_str


import csv


class RejectWordTool(BaseTool):
    name: str = "reject_words_filter"
    description: str = "A reject word checker. Checks whether each sentence contains words that should not be included. Takes as input a list composed in the form: [sentence1, sentence2, ...]."
    args_schema: Type[BaseModel] = Phrase_checker
    return_direct: bool = False
    reject_list: list = []

    def __init__(self, file_path):
        super().__init__()
        self.reject_list = []
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.reject_list += row

    def _run(
            self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[int]:
        in_sent = ast.literal_eval(in_str)
        return_str = ""
        for sent in in_sent:
            rejected = []
            for reject in self.reject_list:
                if reject in sent:
                    rejected.append(reject)

            if len(rejected) > 0:
                return_str += f"{sent}: Rejected {rejected}\n"
            else:
                return_str += f"{sent}: No Rejected Words Included\n"

        return return_str


def count_chars(s):
    count = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ['F', 'W']:  # Full-width or Wide characters
            count += 2
        else:
            count += 1
    return count


class SerpAPIInput(BaseModel):
    in_str: str = Field(description="Input")


# from typing import Optional, Type
# from pydantic import BaseModel
# from langchain.tools.base import BaseTool
# from langchain_community.utilities import SerpAPIWrapper

# from typing import Optional, Type
# from pydantic import BaseModel
# from langchain.tools.base import BaseTool
# from langchain_community.utilities import SerpAPIWrapper

class SerpAPITool(BaseTool):
    name: str = "google_search"
    description: str = "A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False
    region: str = "jp"  # Default region
    language: str = "ja"  # Default language

    def __init__(self, region: str = "jp", language: str = "ja", **kwargs):
        super().__init__(**kwargs)  # Initialize BaseTool attributes
        self.region = region
        self.language = language

    def _run(
            self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        searched_dict = retrieveSearches()

        if "biz PC" in in_str:
            return nbz_product_info

        if "Easy Predictive" in in_str:
            return easy_predictive_info

        if in_str in searched_dict:
            print("\nNote: Loaded from backup")
            return searched_dict[in_str]

        # Add region and language customization
        search_res = SerpAPIWrapper(params={"gl": self.region, "hl": self.language}).run(in_str)
        # searched_dict[in_str] = search_res
        # saveSearches(searched_dict)
        return search_res


class OutputTool(BaseTool):
    name: str = "output_tool"
    description: str = "A tool to simply output your thoughts. Nothing will be done upon input."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False

    def _run(
            self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return "My thought is : " + in_str


class ClickAggregator(BaseTool):
    name: str = "click_aggregator"
    description: str = "Returns the total number of clicks per category for the current ad setting."
    args_schema: Type[BaseModel] = SerpAPIInput
    return_direct: bool = False
    click_df: pd.DataFrame = None

    def __init__(self, file, config, s3_bucket, s3_path, flag):
        super().__init__()
        # Standalone version only supports local file reads
        self.click_df = pd.read_csv(file)

    def _run(
            self, in_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        aggr_clicks = self.click_df.groupby("Category", as_index=False).sum()
        average = aggr_clicks["Clicks"].mean()
        return_text = ""
        for row in range(len(aggr_clicks)):
            return_text += f'Category: {aggr_clicks["Category"][row]}\nClicks: {aggr_clicks["Clicks"][row]}\nDifference to Average: {aggr_clicks["Clicks"][row] - average: .3f}\n\n'
        return return_text


def retrieveSearches():
    if not os.path.exists('../searches/searches.pkl'):
        return {}
    else:
        with open('../searches/searches.pkl', 'rb') as f:
            return pickle.load(f)


def saveSearches(key_dict):
    print("\nNote: Saved to backup")
    with open('../searches/searches.pkl', 'wb') as f:
        pickle.dump(key_dict, f)


def getSerpTool(region: str = "jp", language: str = "ja"):
    # Comment out if live SerpAPI is needed
    return SerpAPITool(region=region, language=language)

    search_tool = load_tools(["serpapi"])
    search_tool[0].name = "google_search"
    return search_tool[0]


class FilterGeneratedKeywordsTool(BaseTool):
    name: str = "filter_generated_keywords"
    description:str = '''This tool receives the final generated keyword result (formatted as a single string that resembles a dictionary with two main keys: 'Branded' and 'Non-Branded'). "
        "It checks whether the result contains any predefined rejected keywords. "
        "If any such keywords are found, they are filtered out, and the tool returns the cleaned keyword result.'''
    rejected_keywords: list = []
    def _run(self, tool_input: str, **kwargs) -> str:
        try:
            # using the ast library to parse the input string into a dictionary
            tool_input = tool_input.replace("```", "")
            if "json" in tool_input:
                tool_input = tool_input.replace("json", "")
            tool_input = tool_input.replace("```", "")
            tool_input = tool_input.lstrip().rstrip()
            result_dict = ast.literal_eval(tool_input)
            if not isinstance(result_dict, dict):
                return "Error: Format Error. Input Cannot Be Parsed"
        except Exception as e:
            if '\\' in tool_input:
                return "Error: Cannot Parse, Please remove all slashes '\\' from the input"
            else:
                return f"Error: Cannot Parer，Please Refer：{e}"
        # parse both branded and non-branded sections
        rejeceted_result_dict = {}
        for section in ["Branded", "Non-Branded"]:
            rejeceted_result_dict[section] = {}
            if section in result_dict and isinstance(result_dict[section], dict):
                for category, keywords in result_dict[section].items():
                    if isinstance(keywords, list):
                        filtered_keywords = [kw for kw in keywords if kw not in self.rejected_keywords]   
                        rejeced_keywords = [kw for kw in keywords if kw in self.rejected_keywords]                    
                        result_dict[section][category] = filtered_keywords
                        rejeceted_result_dict[section][category] = rejeced_keywords
        return f'''
        {result_dict}
        \n Above keywords passed the filter which means they are acceptable, you should keep them.

        Following keywords are rejected keywords:
        {rejeceted_result_dict}
        For keywords that are rejected above, generate new keywords to replace them, and then proceed to Step 4 to validate the search volume'''
        #return str(result_dict)+"\n Those filtered keywords passed, and then proceed to Step 4 to validate their search volume"

    async def _arun(self, tool_input: str, **kwargs) -> str:
        return self._run(tool_input, **kwargs)

class FilterRepeatedKeywords(BaseTool):
    name: str = "filter_repeated_keywords"
    description:str = '''This tool receives the intermediate generated keyword result (formatted as a single string that resembles a dictionary with two main keys: 'Branded' and 'Non-Branded') and provide the final generated keywords results. "
        "It filers whether generated keywords are totally new or not as the generated keywords cannot be the same with previous generated history "
        "If any such keywords are found, they are flagged and keyword should be regerated.'''
    history_keywords: list = []
    record_repeated_keywords: list = []
    def _run(self, tool_input: str, **kwargs) -> str:
        try:
            # using the ast library to parse the input string into a dictionary
            tool_input = tool_input.replace("```", "")
            if "json" in tool_input:
                tool_input = tool_input.replace("json", "")
            tool_input = tool_input.replace("```", "")
            tool_input = tool_input.lstrip().rstrip()
            result_dict = ast.literal_eval(tool_input)
            if not isinstance(result_dict, dict):
                return "Error: Format Error. Input Cannot Be Parsed"
        except Exception as e:
            return f"Error: Cannot Parer，Please Refer：{e}"
        # parse both branded and non-branded sections
        flag = ""
        for section in ["Branded", "Non-Branded"]:
            if section in result_dict and isinstance(result_dict[section], dict):
                for category, keywords in result_dict[section].items():
                    if isinstance(keywords, list):
                        filtered_keywords = [kw for kw in keywords if kw in self.history_keywords]
                        for kw in keywords:
                            for hist_kw in self.history_keywords:
                                if kw.replace('"', "") == hist_kw.replace('"', ""):
                                    filtered_keywords.append(kw)
                        #result_dict[section][category] = filtered_keywords
                        for keyword in filtered_keywords:
                            flag += f"This keyword {keyword} is contained in the history keywords, please regenerate this keyword\n"
                            self.record_repeated_keywords.append(keyword)
        if flag == "":
            flag = "All keywords are new, no need to regenerate. Next step is step 2 to evaluate the coherence of the generated keywords"
        else:
            flag += "Following keywords are repeated keyword that you have generated, you should not generate them again:"
            for filtered_keywords in list(set(self.record_repeated_keywords)):
                flag += f"{filtered_keywords}\n"
        return str(flag)

    async def _arun(self, tool_input: str, **kwargs) -> str:
        return self._run(tool_input, **kwargs)

# class FilterRepeatedKeywords(BaseTool):
#     name: str = "filter_repeated_keywords"
#     description: str = (
#         "This tool receives the intermediate generated keyword result (formatted as a single string "
#         "that resembles a dictionary with two main keys: 'Branded' and 'Non-Branded') and provides the final "
#         "generated keywords results. It filters whether generated keywords are totally new or not, as they "
#         "cannot be the same as previously generated history. If any such keywords are found, they are flagged "
#         "and should be regenerated."
#     )
#     history_keywords: List[str] = Field(default_factory=list)
#     record_repeated_keywords: List[str] = Field(default_factory=list)
#     last_repeated_cache: List[str] = Field(default_factory=list)
#     repeat_count: int = Field(default = 0)
    
#     def _run(self, tool_input: str, **kwargs) -> str:
#         try:
#             # using the ast library to parse the input string into a dictionary
#             tool_input = tool_input.replace("```", "")
#             if "json" in tool_input:
#                 tool_input = tool_input.replace("json", "")
#             tool_input = tool_input.replace("```", "")
#             tool_input = tool_input.lstrip().rstrip()
#             result_dict = ast.literal_eval(tool_input)
#             if not isinstance(result_dict, dict):
#                 return "Error: Format Error. Input cannot be parsed."
#         except Exception as e:
#             return f"Error: Cannot parse input. Please refer: {e}"
        
#         flag = ""
#         current_repeated = []
#         # Parse both Branded and Non-Branded sections
#         for section in ["Branded", "Non-Branded"]:
#             if section in result_dict and isinstance(result_dict[section], dict):
#                 for category, keywords in result_dict[section].items():
#                     if isinstance(keywords, list):
#                         # Find repeated keywords in this list based on history_keywords
#                         for kw in keywords:
#                             # Standardize keyword by removing quotes for comparison
#                             standardized_kw = kw.replace('"', "")
#                             if standardized_kw in self.history_keywords:
#                                 current_repeated.append(kw)
        
#         # Remove duplicates from current_repeated list
#         current_repeated = list(set(current_repeated))
        
#         # Check if the current rejected keywords are identical to the last round
#         if current_repeated == self.last_repeated_cache and current_repeated:
#             # If so, simply instruct regeneration without any analysis.
#             return (
#                 "Please regenerate these keywords as they are identical to the previously rejected ones, so the tool will not even check them, you are forced to regenerate them: " +
#                 str(current_repeated) +
#                 "\nNext step: Regenerate new keywords and re-enter Step 1."
#             )
#             self.repeat_count += 1
#             if self.repeat_count > 5:
#                 raise Exception("Too many repeats, aborting.")
#         else:
#             # Update the cache with current rejected keywords.
#             self.last_repeated_cache = current_repeated
#             self.repeat_count = 0
        
#         # If any repeated keywords found, prepare the flag message.
#         if current_repeated:
#             flag += "The following keywords are repeated from history and must be regenerated:\n"
#             for kw in current_repeated:
#                 flag += f"- {kw}\n"
#             flag += "Next step remains Step 1."
#         else:
#             flag = "All keywords are new, no need to regenerate. Next step is Step 2 to evaluate the coherence of the generated keywords."
        
#         # Update history_keywords with all keywords from the input (if needed)
#         for section in ["Branded", "Non-Branded"]:
#             if section in result_dict and isinstance(result_dict[section], dict):
#                 for category, keywords in result_dict[section].items():
#                     if isinstance(keywords, list):
#                         for kw in keywords:
#                             if kw not in self.history_keywords:
#                                 self.history_keywords.append(kw)
        
#         return flag

#     async def _arun(self, tool_input: str, **kwargs) -> str:
#         return self._run(tool_input, **kwargs)

class CoherenceRefelctionTool(BaseTool):
    name: str = "coherence_reflection"
    description: str = '''This tool receives the intermediate generated keyword result (formatted as a single string that resembles a dictionary with two main keys: 'Branded' and 'Non-Branded') and a evaluator in this tool will should think about whether those keywords well represent the provide product information or not.
       The evalutor in this tool will evaluate the coherence between those keywords and the product information. Each keyword will be given a score from 1 to 5 based on how well it represents the product information, reaons for the score and suggestion indicator whether this keyword should be replaced or kept or not.
       For keywords that are suggested to be replaced by the evaluator. You should replace them with new keywords and re-run the keyword generation process.
       '''
    
    product: str = Field(default="")
    product_information: str = Field(default="")
    config: dict = Field(default_factory=dict)
    evaluation_prompt: str = Field(default="")
    coherence_evaluator: Any = Field(default=None)
    history_evaluation: str = Field(default="")

    def __init__(self, config=None, product="", **kwargs):
        super().__init__(**kwargs)
        # Now set these values
        self.product = product
        self.config = config or {}
        self.product_information = ""
        self.evaluation_prompt = ''''You are given the intermediate generated keyword result (formatted as a single string that resembles a dictionary with two main keys: 'Branded' and 'Non-Branded') and the production information, you should evaluate the coherence between those keywords and the product information. 
        For each keyword, give a score from 1 to 5 based on how well it represents the product information, give a reason for this score, give a suggestion indicator whether this keyword should be replaced or kept or not.
        1: The keyword does not represent the product information at all.
        2: The keyword does not represent the product information well.
        3: The keyword somewhat represents the product information.
        4: The keyword represents the product information well.
        5: The keyword perfectly represents the product information.
        Additionanlly, how well the keyword matches the product information is not the only metric, as we are going to use those keywords in advertisement campaigns, the keywords should also be attractive to the target audience, especially the target audience of the product and how it matches the search behavior of user.
        For each keyword, provide a score strictly in the following dictionary format:
        {{"keyword": "keyword1", "score": 4 , "reason": "You analysis reason for the score", "suggestion": "keep"}}
        ....
        {{"keyword": "keywordn", "score": 2 , "reason": "You analysis reason for the score", "suggestion": "replace"}}

        The generated keywords that you are required to evaluate are:
        {generated_keywords}
        The product information is:
        {product_information}
        You evaluation history is following, for the keywords that we ask you to evaluate but you have evaluated in the follow history information, you should not evaluate them again so you only produce results for new keywords:
        {history_evaluation}
        '''
        self.coherence_evaluator = self._initialize_evaluator()
    
    def _initialize_evaluator(self):
        if not self.config:
            raise ValueError("Config is required for CoherenceRefelctionTool")
        
        # Check if config is a ConfigParser object or a dict
        if hasattr(self.config, 'get') and callable(self.config.get) and not isinstance(self.config, dict):
            # It's a ConfigParser object
            try:
                model = self.config.get('LLM', 'CoherenceEvaluatorModel')
                if model == 'GPT-4o':
                    return AzureChatOpenAI(
                        deployment_name=self.config.get('LLM', 'gpt4o_deployment_name'),
                        openai_api_version=self.config.get('LLM', 'gpt4o_openai_api_version'),
                        openai_api_key=self.config.get('KEYS', 'OPENAI_GPT4O_API_KEY'),
                        azure_endpoint=self.config.get('KEYS', 'OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT'),
                        temperature=float(self.config.get('LLM', 'temperature'))
                    )
                elif model == 'o3-mini':
                    return AzureChatOpenAI(
                        deployment_name=self.config.get('LLM', 'o3_deployment_name'),
                        openai_api_version=self.config.get('LLM', 'o3_openai_api_version'),
                        openai_api_key=self.config.get('KEYS', 'OPENAI_o3mini_API_KEY'),
                        azure_endpoint=self.config.get('KEYS', 'OPENAI_o3mini_AZURE_OPENAI_ENDPOINT')
                    )
            except Exception as e:
                raise ValueError(f"Error accessing config values: {e}")
        else:
            # It's a dictionary
            try:
                model = self.config.get('CoherenceEvaluatorModel', {}).get('MODEL')
                if model == 'GPT-4o':
                    return AzureChatOpenAI(
                        deployment_name=str(self.config['CoherenceEvaluator']['gpt4o_deployment_name']),
                        openai_api_version=str(self.config['CoherenceEvaluator']['gpt4o_openai_api_version']),
                        openai_api_key=str(self.config['KEYS']['OPENAI_GPT4O_API_KEY']),
                        azure_endpoint=str(self.config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']),
                        temperature=float(self.config['CoherenceEvaluator']['temperature'])
                    )
                elif model == 'o3-mini':
                    return AzureChatOpenAI(
                        deployment_name=str(self.config['CoherenceEvaluator']['o3_deployment_name']),
                        openai_api_version=str(self.config['CoherenceEvaluator']['o3_openai_api_version']),
                        openai_api_key=str(self.config['KEYS']['OPENAI_o3mini_API_KEY']),
                        azure_endpoint=str(self.config['KEYS']['OPENAI_o3mini_AZURE_OPENAI_ENDPOINT'])
                    )
            except Exception as e:
                raise ValueError(f"Error accessing config dictionary values: {e}")
        
        raise ValueError(f"Unknown model or config format")

    def _run(self, tool_input: str, **kwargs) -> str:
                    
        # Set the product information based on the product
        if "biz PC" in self.product:
            self.product_information = nbz_product_info
        elif "Easy Predictive" in self.product:
            self.product_information = easy_predictive_info
        else:
            return "Error: Unknown product, cannot set product information"
        # parse both branded and non-branded sections
        combined_prompt = self.evaluation_prompt.format(generated_keywords=tool_input,  product_information=self.product_information, history_evaluation=self.history_evaluation)
        
        coherence_evaluator_response = self.coherence_evaluator.invoke(combined_prompt)
        self.history_evaluation += coherence_evaluator_response.content
        return str(coherence_evaluator_response.content+ "\n After regnerating keywords that are flagged with Replace, next Step is Step 3 to filter **rejeceted** keyword not repeated keyword so do not go back to Step 1.")

    async def _arun(self, tool_input: str, **kwargs) -> str:
        return self._run(tool_input, **kwargs)


class RejectReflextionTool(BaseTool):
    name: str = "reject_reflection"
    description:str = " This tool receives the rejected keyword list and think about why those keywords are bad. The results of this reflextion process should benifit the follow keyword generation process. "
    rejected_keywords: list = []
    product: str = Field(default="")
    product_information: str = Field(default="")
    config: dict = Field(default_factory=dict)
    evaluation_prompt: str = Field(default="")
    reject_evaluator: Any = Field(default=None)
    history_evaluation: str = Field(default="")
    def __init__(self, config=None, product="", rejected_keywords=[], **kwargs):
        super().__init__(**kwargs)
        # Now set these values
        self.product = product
        self.config = config or {}
        self.product_information = ""
        self.reject_evaluator = self._initialize_evaluator()
        self.rejected_keywords = rejected_keywords
    
    def _initialize_evaluator(self):
        if not self.config:
            raise ValueError("Config is required for CoherenceRefelctionTool")
        
        # Check if config is a ConfigParser object or a dict
        if hasattr(self.config, 'get') and callable(self.config.get) and not isinstance(self.config, dict):
            # It's a ConfigParser object
            try:
                model = self.config.get('LLM', 'CoherenceEvaluatorModel')
                if model == 'GPT-4o':
                    return AzureChatOpenAI(
                        deployment_name=self.config.get('LLM', 'gpt4o_deployment_name'),
                        openai_api_version=self.config.get('LLM', 'gpt4o_openai_api_version'),
                        openai_api_key=self.config.get('KEYS', 'OPENAI_GPT4O_API_KEY'),
                        azure_endpoint=self.config.get('KEYS', 'OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT'),
                        temperature=float(self.config.get('LLM', 'temperature'))
                    )
                elif model == 'o3-mini':
                    return AzureChatOpenAI(
                        deployment_name=self.config.get('LLM', 'o3_deployment_name'),
                        openai_api_version=self.config.get('LLM', 'o3_openai_api_version'),
                        openai_api_key=self.config.get('KEYS', 'OPENAI_o3mini_API_KEY'),
                        azure_endpoint=self.config.get('KEYS', 'OPENAI_o3mini_AZURE_OPENAI_ENDPOINT')
                    )
            except Exception as e:
                raise ValueError(f"Error accessing config values: {e}")
        else:
            # It's a dictionary
            try:
                model = self.config.get('CoherenceEvaluatorModel', {}).get('MODEL')
                if model == 'GPT-4o':
                    return AzureChatOpenAI(
                        deployment_name=str(self.config['CoherenceEvaluator']['gpt4o_deployment_name']),
                        openai_api_version=str(self.config['CoherenceEvaluator']['gpt4o_openai_api_version']),
                        openai_api_key=str(self.config['KEYS']['OPENAI_GPT4O_API_KEY']),
                        azure_endpoint=str(self.config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']),
                        temperature=float(self.config['CoherenceEvaluator']['temperature'])
                    )
                elif model == 'o3-mini':
                    return AzureChatOpenAI(
                        deployment_name=str(self.config['CoherenceEvaluator']['o3_deployment_name']),
                        openai_api_version=str(self.config['CoherenceEvaluator']['o3_openai_api_version']),
                        openai_api_key=str(self.config['KEYS']['OPENAI_o3mini_API_KEY']),
                        azure_endpoint=str(self.config['KEYS']['OPENAI_o3mini_AZURE_OPENAI_ENDPOINT'])
                    )
            except Exception as e:
                raise ValueError(f"Error accessing config dictionary values: {e}")
        
        raise ValueError(f"Unknown model or config format")

    def _run(self, tool_input: str, **kwargs) -> str:
        if "biz PC" in self.product:
            self.product_information = nbz_product_info
        elif "Easy Predictive" in self.product:
            self.product_information = easy_predictive_info
        else:
            return "Error: Unknown product, cannot set product information"
        
        rejected_keywords_prompt =  """The definition of a rejected keyword is that the search volume returned from the Google API is less than 5 which means basically no one searched those keywords. 
        You have generated following rejeceted keywords either in this turn or previously:
        {reject_keywords_str}
        The product information is:
        {product_information}
        Think about why those keywords are rejected. Whlie the reason may not be specific since it may be related to the quality of the product itself, but it is still important to reflect on the quality of the keywords generated.
        Analyze the reason why they are bad and this information will be used to guide our model to generate better keywords. However, do not leak those rejected keywords in you answer.
        """
        combined_prompt = rejected_keywords_prompt.format(reject_keywords_str=str(self.rejected_keywords),  product_information=self.product_information)
        reject_evaluator_response = self.reject_evaluator.invoke(combined_prompt)
        return str(reject_evaluator_response.content)


class KeywordSearchVolumeTool(BaseTool):
    """
    A tool to validate keywords based on search volume metrics and manage keyword lists.
    This tool checks if keywords meet specified search volume thresholds and manages
    accepted and rejected keyword lists.
    """
    
    name: str = "keyword_search_volume_validator"
    description: str = """
    This tool validates a set of keywords by checking their search volume metrics.
    It categorizes keywords as acceptable or rejected based on a search volume threshold.
    It works with both branded and non-branded keywords, and manages the storage of 
    keyword data in files or S3 buckets.
    """
    config: Optional[Dict] = None
    get_keyword_planner_metrics: Any = Field(default=None)
    base_path: str = Field(default=".")
    normalize_keyword: Any = Field(default=None)
    s3_bucket: Any = Field(default=None)
    rejected_keyword_list: str = Field(default="")
    save_dir: str = Field(default="")
    history_rejected_keyword: List[str] = Field(default_factory=list)
    tokenizer: Any = Field(default=None)
    retry_counter: int = Field(default=0)
    last_rejeceted_cache: str = Field(default="")
    repeat_counter: int = Field(default=int)
    reject_evaluator: Any = Field(default=None)

    def __init__(self, config=None, s3_bucket=None, rejected_keyword_list="", save_dir="", get_keyword_planner_metrics=None):
        """
        Initialize the keyword search volume validation tool.
        
        Args:
            config: Configuration dictionary containing settings for keyword validation.
            s3_bucket: S3 bucket name if using S3 for storage.
            save_dir: Directory path for saving files.
            get_keyword_planner_metrics: Function to retrieve keyword metrics.
        """
        super().__init__() 
        self.config = config
        self.get_keyword_planner_metrics = get_keyword_planner_metrics
        if str(self.config["EXE"]["S3_DEPLOY"]) == "True":
            self.base_path = 'local_policy/ad_group_keyword_v2'
        else:
            self.base_path = "."
        self.normalize_keyword = self.dynamic_import(f"{self.base_path}/okg/utils", "normalize_keyword")
        self.rejected_keyword_list = rejected_keyword_list
        self.save_dir = save_dir
        self.s3_bucket = s3_bucket
        self.retry_counter = 0
        self.last_rejeceted_cache = []
        # Initialize a multilingual tokenizer that can handle Japanese, English, etc.
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.repeat_counter = 0
        
    def dynamic_import(self, module_name, function_name):
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, function_name)
        return cls
        
    def normalize_keyword(self, keyword: str) -> str:
        """
        Normalize keyword for consistent matching.
        
        Args:
            keyword: The keyword to normalize.
            
        Returns:
            Normalized form of the keyword.
        """
        if keyword is None:
            return ""
        return keyword.lower().strip()
        
    def has_japanese_chars(self, text: str) -> bool:
        """
        Check if text contains Japanese characters.
        
        Args:
            text: The text to check.
            
        Returns:
            True if the text contains Japanese characters, False otherwise.
        """
        # Japanese character ranges
        japanese_ranges = [
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x9FFF),  # Kanji
            (0x3400, 0x4DBF)   # Extended Kanji
        ]
        
        for char in text:
            code = ord(char)
            for start, end in japanese_ranges:
                if start <= code <= end:
                    return True
        return False
    
    def merge_subwords(self, tokens: List[str]) -> List[str]:
        """
        Merge subword tokens into whole words.
        Handles both English and Japanese tokenization patterns.
        
        Args:
            tokens: List of tokens to merge.
            
        Returns:
            List of merged tokens.
        """
        merged = []
        for token in tokens:
            # Handle BERT-style subword tokens (##)
            if token.startswith("##") and merged:
                merged[-1] += token[2:]
            # Handle potential Japanese tokenizer markers
            elif token.startswith("▁") and len(token) > 1:  # Some tokenizers use ▁ for word beginnings
                merged.append(token[1:])
            # Handle regular tokens
            else:
                merged.append(token)
        return merged
    
    def get_meaningful_segments(self, tokens: List[str], is_prefix: bool = True, max_length: int = 2) -> List[str]:
        """
        Extract meaningful segments (prefixes or suffixes) from tokens.
        
        Args:
            tokens: List of merged tokens.
            is_prefix: If True, extract prefixes; if False, extract suffixes.
            max_length: Maximum number of tokens to consider.
            
        Returns:
            List of extracted segments.
        """
        if not tokens:
            return []
        
        segments = []
        
        # Get segments of different lengths
        for i in range(1, min(max_length + 1, len(tokens) + 1)):
            if is_prefix:
                segment = " ".join(tokens[:i])
            else:
                segment = " ".join(tokens[-i:])
            segments.append(segment)
            
        return segments

    def analyze_history_rejected(self, history_list: List[str]) -> Tuple[str, List[str], List[str]]:
        """
        Analyzes common patterns among rejected keywords and returns an analysis report
        along with lists of common prefixes and suffixes.
        
        Returns:
            A tuple (report, common_prefixes, common_suffixes)
        """
        if not history_list:
            return "No rejected keywords to analyze.", [], []
        
        # Clean the keywords
        cleaned_keywords = [kw.strip() for kw in history_list if kw.strip()]
        
        # Count language distribution
        jp_count = sum(1 for kw in cleaned_keywords if self.has_japanese_chars(kw))
        en_count = len(cleaned_keywords) - jp_count
        mixed_count = sum(1 for kw in cleaned_keywords 
                          if self.has_japanese_chars(kw) and re.search(r'[a-zA-Z]', kw))
        
        # Process all keywords
        prefix_counter = Counter()
        suffix_counter = Counter()
        
        for kw in cleaned_keywords:
            # Keep original case for Japanese, lowercase for English parts
            if not self.has_japanese_chars(kw):
                kw = kw.lower()
                
            # Tokenize and merge subwords
            tokens = self.tokenizer.tokenize(kw)
            merged_tokens = self.merge_subwords(tokens)
            
            if not merged_tokens:
                continue
                
            # Get meaningful prefixes and suffixes
            prefixes = self.get_meaningful_segments(merged_tokens, is_prefix=True)
            suffixes = self.get_meaningful_segments(merged_tokens, is_prefix=False)
            
            # Count occurrences
            for prefix in prefixes:
                prefix_counter[prefix] += 1
            for suffix in suffixes:
                suffix_counter[suffix] += 1
        
        # Get most common patterns
        common_prefixes = [prefix for prefix, count in prefix_counter.most_common(3)]
        common_suffixes = [suffix for suffix, count in suffix_counter.most_common(3)]
        
        # Generate report
        report = "Rejected Keywords Analysis:\n"
        report += f"- Total rejected keywords: {len(cleaned_keywords)}\n"
        
        if jp_count > 0 or en_count > 0:
            report += f"- Japanese keywords: {jp_count}\n"
            report += f"- English keywords: {en_count}\n"
            report += f"- Mixed language keywords: {mixed_count}\n"
            
        report += f"- Common prefixes of rejected keywords: {common_prefixes}\n"
        report += f"- Common suffixes of rejected keywords: {common_suffixes}\n"
        
        # Add recommendations based on language distribution
        if jp_count > 0 and en_count > 0:
            report += "- Mixed language patterns detected. Consider language-specific keyword generation.\n"
        
        report += (
            "Recommendation: When generating new keywords, ensure that they do not start with any "
            "of the above common prefixes or end with any of the above common suffixes. "
            "For mixed-language keywords, pay special attention to both Japanese and English patterns. "
            "New keywords should be entirely distinct in vocabulary and structure."
        )
        
        return report, common_prefixes, common_suffixes

    def check_candidate_keyword(self, candidate: str, common_prefixes: List[str], common_suffixes: List[str]) -> bool:
        """
        Checks whether the candidate keyword matches any of the common patterns to avoid.
        
        Args:
            candidate: The candidate keyword.
            common_prefixes: List of common prefixes to avoid.
            common_suffixes: List of common suffixes to avoid.
            
        Returns:
            True if the candidate should be rejected, False otherwise.
        """
        if not candidate or not candidate.strip():
            return False
            
        # Normalize candidate based on language content
        candidate = candidate.strip()
        if not self.has_japanese_chars(candidate):
            candidate = candidate.lower()
            
        # Tokenize and merge
        tokens = self.tokenizer.tokenize(candidate)
        merged_tokens = self.merge_subwords(tokens)
        
        if not merged_tokens:
            return False
            
        # Get candidate's segments
        candidate_prefixes = self.get_meaningful_segments(merged_tokens, is_prefix=True)
        candidate_suffixes = self.get_meaningful_segments(merged_tokens, is_prefix=False)
        
        # Check for overlap
        for prefix in candidate_prefixes:
            if prefix in common_prefixes:
                return True
                
        for suffix in candidate_suffixes:
            if suffix in common_suffixes:
                return True
                
        return False

    def generate_feedback_json(self, rejected_kw_list: List[str], analysis_report: str,
                               common_prefixes: List[str], common_suffixes: List[str]) -> str:
        """
        Generates a structured JSON string containing feedback information for the LLM agent.
        """
        feedback = {
            "number of retries": self.retry_counter,
            "rejected_keywords": rejected_kw_list,
            "common_prefixes": common_prefixes,
            "common_suffixes": common_suffixes,
            "instruction": (
                "Please regenerate new keywords that do not start with any of the common prefixes or "
                "end with any of the common suffixes indicated above. Use chain-of-thought reasoning to "
                "ensure the new keywords are entirely distinct in vocabulary and structure, with zero token "
                "overlap and zero similarity with the rejected keywords. You should have a space to sperate English words and Japanese words like PC 移行."
            )
            #,"analysis_report": analysis_report
        }
        return json.dumps(feedback, indent=2, ensure_ascii=False)
    
    def _run(self, tool_input: str, **kwargs) -> str:
        try:
            # Clean the input
            tool_input = tool_input.replace("```", "")
            if "json" in tool_input:
                tool_input = tool_input.replace("json", "")
            tool_input = tool_input.replace("```", "")
            tool_input = tool_input.strip()
            result_dict = ast.literal_eval(tool_input)
            if not isinstance(result_dict, dict):
                return "Error: Format Error. Input cannot be parsed."
        except Exception as e:
            return f"Error: Cannot parse input. Please refer: {e}"
        
        # Process based on the BRANDED_KEYWORD configuration
        if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
            all_non_branded = []
            for category, keywords in result_dict['Non-Branded'].items():
                all_non_branded += keywords
            metrics = self.get_keyword_planner_metrics(
                'local',
                all_non_branded,
                '2023-06',
                '2024-06'
            )
        elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
            all_branded = []
            for category, keywords in result_dict['Branded'].items():
                all_branded += keywords
            metrics = self.get_keyword_planner_metrics(
                'local',
                all_branded,
                '2023-06',
                '2024-06'
            )
        else:
            raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
        
        # Normalize search volume metrics
        if str(self.config["EXE"]["S3_DEPLOY"]) == "True":
            keyword_to_search_volume = {self.normalize_keyword(metric['keyword']): metric['avg_monthly_searches'] for metric in metrics}
        else:
            keyword_to_search_volume = {
                self.normalize_keyword(metric['Keyword']): metric['Avg. monthly searches'] for metric in metrics
            }
        
        keyword_list_needed = all_non_branded if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False' else all_branded
        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword in keyword_list_needed]
        if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
            return "All keywords meet the search volume threshold. Output current keywords as the final answer using the predefined format."
        # If all keywords meet the threshold, output them as final answer.
        if all(x >= int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']) for x in new_words_check):
            return "All keywords meet the search volume threshold. Output current keywords as the final answer using the predefined format."
        else:
            self.retry_counter += 1
            rejected_kw_list = []
            for i in range(len(new_words_check)):
                if new_words_check[i] < int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']):
                    rejected_kw_list.append(str(keyword_list_needed[i]))
            if rejected_kw_list == self.last_rejeceted_cache:
                # If so, instruct regeneration directly without analysis.
                self.repeat_counter += 1
                if self.repeat_counter > 10:
                    return (f"Should-not-generate-list:{str(rejected_kw_list)}\n"
                            "You should delete those rejected keywords in the Should-not-generat-list from the generated results and output other keywords using the predefined format previously as the final answer."      
                            "\nNext step: Delete those keywords and output the final answer.")
                else:
                    return (f"Should-not-generate-list:{str(rejected_kw_list)}\n"
                            "You must regenerate the keywords in this should-not-generate list, as they do not have sufficient search volume. Generate the same number of keywords as in this Should-not-generate-list, then replace the keywords in the should-not-generate with the newly generated ones while maintaining the predefined dictionary structure. You can ignore prefix or suffix requirements if you think it is hard to regerate new keywords while kepping those requirements."
                            "\nNext step: Must regenerate those rejected keywords and then re-enter Step 4.")
            else:
                self.last_rejeceted_cache = rejected_kw_list

            base_results = f"Rejected keywords list:{str(rejected_kw_list)}\n "
    
            # Add the currently rejected keywords to history
            self.history_rejected_keyword.extend(rejected_kw_list)
            # Analyze the history of rejected keywords, obtaining the analysis report and common patterns.
            analysis_report, common_prefixes, common_suffixes = self.analyze_history_rejected(self.history_rejected_keyword)
                
            # Generate structured feedback JSON that the LLM agent can use for regeneration.
            feedback_json = self.generate_feedback_json(
                rejected_kw_list, analysis_report, common_prefixes, common_suffixes
            )
            
            if self.retry_counter >10 and len(rejected_kw_list) == 1:
                return feedback_json + "\nNext step: You have tried 10 times and still the keyword does not pass the search volume validator. You should just delete it from the generated results and output other keywords using the predefined format previously as the final answer."
            elif self.retry_counter >20:
                return feedback_json + "\nNext step: You have tried 20 times and still the keyword does not pass the search volume validator. You should just delete all those rejeceted keywords from the generated results and output other keywords using the predefined format previously as the final answer."
            else: 
                return feedback_json + "\nYou must regenerate new keywords **only** for keywords in the Rejected keywords list and replace the keywords in the Rejected keywords list with the newly generated ones. Next step: After regenarting new keywords for rejected keywords and replace them, next step remains Step 4 to check the search volume of those generated keywords."

class EvaluateAndSortKeywordsTool(BaseTool):
    name: str = "evaluate_and_sort_keywords"
    description: str = """This tool loads historical performance data from a CSV file and computes TOPSIS scores for each historical keyword 
        using a provided function. Then it sorts the keywords within each category by TOPSIS score from high to low and, for each category, 
        labels the top 25% (based on the distribution within that category) as 'Good' and the remaining 75% as 'Bad'. 
        It also computes each category's average TOPSIS score and finally sorts the categories in descending order of this average. 
        The final output is a dictionary with key 'Sorted_Categories', which contains the category names, their average TOPSIS scores, 
        and the sorted keyword list with labels. This information is intended to inform the model which categories and keywords are good.
    """

    def __init__(self, topsis_func=None, historical_csv_path="historical_keywords.csv"):
        """
        topsis_func: A function to compute TOPSIS scores on a DataFrame.
        historical_csv_path: Path to a CSV file containing historical keywords and performance data.
        """
        super().__init__()
        self.topsis_func = topsis_func
        self.historical_csv_path = historical_csv_path

    def _run(self, tool_input: str, **kwargs) -> str:
        # Note: tool_input is ignored since this tool uses historical data directly.
        try:
            df_hist = pd.read_csv(self.historical_csv_path)
            if self.topsis_func:
                df_hist = self.topsis_func(df_hist)
            else:
                df_hist["Topsis Score"] = np.random.uniform(0, 1, len(df_hist))
        except Exception as e:
            return f"Error: {e}"

        # Group historical data by Category.
        category_dict = {}
        grouped = df_hist.groupby("Category")
        for cat, group in grouped:
            group_sorted = group.sort_values("Topsis Score", ascending=False)
            avg_topsis = group_sorted["Topsis Score"].mean()
            threshold = np.quantile(group_sorted["Topsis Score"].values, 0.75)
            keywords_list = []
            for _, row in group_sorted.iterrows():
                rank_label = "Good" if row["Topsis Score"] >= threshold else "Bad"
                keywords_list.append(
                    {"Keyword": row["Keyword"], "Topsis Score": row["Topsis Score"], "Rank": rank_label})
            category_dict[cat] = {"Average Topsis Score": avg_topsis, "Keywords": keywords_list}

        sorted_categories = sorted(category_dict.items(), key=lambda x: x[1]["Average Topsis Score"], reverse=True)
        result_dict = {"Sorted_Categories": []}
        for cat, data in sorted_categories:
            result_dict["Sorted_Categories"].append({
                "Category": cat,
                "Average Topsis Score": data["Average Topsis Score"],
                "Keywords": data["Keywords"]
            })
        return str(result_dict)

    async def _arun(self, tool_input: str, **kwargs) -> str:
        return self._run(tool_input, **kwargs)


class ClusterAnalysisTool(BaseTool):
    name: str = "analyze_clusters"
    description: str = """
        This tool receives clustering information from historical keyword data. The input should be a string representation of a dictionary containing keys such as 'cluster_centers_keywords', 'cluster_stats', and 'category_list'. 
        It then analyzes each cluster and provides a concise reasoning for why users might search for the keywords in that cluster. 
        For each cluster, the output includes the center keyword, the average TOPSIS score, and a brief analysis of potential user intent.
    """

    def _run(self, tool_input: str, **kwargs) -> str:
        try:
            cluster_info = ast.literal_eval(tool_input)
        except Exception as e:
            return f"Error: Could not parse clustering info: {e}"

        cluster_centers = cluster_info.get("cluster_centers_keywords", [])
        cluster_stats = cluster_info.get("cluster_stats", [])

        if not cluster_stats:
            return "Error: No cluster statistics provided."

        analysis_result = {}
        for i, stat in enumerate(cluster_stats):
            center = cluster_centers[i] if i < len(cluster_centers) else "N/A"
            avg_score = stat.get("Topsis Score", 0)
            desc = stat.get("description", "No description available.")
            analysis_result[f"Cluster {i + 1}"] = (
                f"Center Keyword: {center}. Average TOPSIS Score: {avg_score:.2f}. "
                f"User engagement may be inferred from metrics (e.g. Avg. CPC, Clicks). Analysis: {desc}"
            )
        return str(analysis_result)

    async def _arun(self, tool_input: str, **kwargs) -> str:
        return self._run(tool_input, **kwargs)

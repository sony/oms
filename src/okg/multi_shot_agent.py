import configparser
import os
import pdb
import time

from langchain import hub
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_react_agent
import numpy as np
import pandas as pd

import random
import importlib.util

# from google_util.google_ads_api import get_keyword_planner_metrics
import copy
import io
from okg.prompts import (first_prompt, dynamic_multi_shot_multi_query_prompt,
                     dynamic_multi_shot_single_query_prompt, dynamic_multi_shot_single_query_prompt_cluster,
                     naive_multi_shot_prompt, react, finalizing_prompt)
from okg.utils import concatenate_llm_parts, concatenate_reflection_beginning,run_with_retries, create_clustering, predict_cluster, merge_cluster_stats
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


# tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
# bert_model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
# scorer = BERTScorer(lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')
# scorer = BERTScorer(model_type="cl-tohoku/bert-base-japanese", lang="ja", device='cuda' if torch.cuda.is_available() else 'cpu')

# from okg.load_and_embed import customized_trend_retriever
# from okg.utils import run_with_retries
# from tools import getSerpTool, OutputTool, ClickAggregator

from langchain_community.agent_toolkits.load_tools import load_tools
import os

# import sys
# sys.path.append('../')


class multi_shot_agent:
    def __init__(self, s3_bucket, s3_path, config_path, setting_day, code_in=0):

        # 0. Read the configuration file
        self.config = configparser.ConfigParser()
        try:
            self.config.read(config_path)
            # self.config.read('./config_base.ini')
        except Exception as e:
            raise ValueError("Failed to read the configuration file: " + str(e))

        self.observation_period = int(self.config['SETTING']['OBSERVATION_PERIOD_DAY'])

        # self.csv_file_path = self.config['FILE']['CSV_FILE']

        # self.setting_day = pd.to_datetime(self.config['SYSTEM']['SETTING_DAY'])
        self.setting_day = pd.to_datetime(setting_day)

        # self.dataframe = pd.read_csv(str(self.config['FILE']['CSV_FILE']))

        if str(self.config["EXE"]["S3_DEPLOY"]) == "True":
            self.save_dir = s3_path
            base_path = 'local_policy'
            self.initial_keyword_data = self.adjust_path_for_local_policy(
                self.config["SETTING"]["initial_keyword_data"])
            self.generated_keyword_data = self.adjust_path_for_local_policy(
                self.config["SETTING"]["generated_keyword_data"])
            self.rule_data = self.adjust_path_for_local_policy(self.config["SETTING"]["rule_data"])
            self.rejected_keyword_list = self.adjust_path_for_local_policy(
                self.config["SETTING"]["rejected_keyword_list"])
            self.history_df_data = self.adjust_path_for_local_policy(self.config["SETTING"]["history_data"])

        elif str(self.config["EXE"]["S3_DEPLOY"]) == "False":
            self.save_dir = './'
            base_path = '.'
            self.initial_keyword_data = self.config["SETTING"]["initial_keyword_data"]
            self.generated_keyword_data = self.config["SETTING"]["generated_keyword_data"]
            self.rule_data = self.config["SETTING"]["rule_data"]
            self.rejected_keyword_list = self.config["SETTING"]["rejected_keyword_list"]
            # Google Ads API removed: provide local stub
            def _stub_get_keyword_planner_metrics(account_id, keywords, start_month, end_month):
                return [{
                    'Keyword': kw,
                    'Avg. monthly searches': 100
                } for kw in keywords]
            self.get_keyword_planner_metrics = _stub_get_keyword_planner_metrics
            self.history_df_data = self.config["SETTING"]["history_data"]

        # Dynamic import based on calculated base path
        self.customized_trend_retriever = self.dynamic_import(f"{base_path}/okg/load_and_embed",
                                                              "customized_trend_retriever")
        self.run_with_retries = self.dynamic_import(f"{base_path}/okg/utils", "run_with_retries")
        self.normalize_keyword = self.dynamic_import(f"{base_path}/okg/utils", "normalize_keyword")
        self.getSerpTool = self.dynamic_import(f"{base_path}/tools", "getSerpTool")
        self.OutputTool = self.dynamic_import(f"{base_path}/tools", "OutputTool")
        self.ClickAggregator = self.dynamic_import(f"{base_path}/tools", "ClickAggregator")
        self.WordFilter = self.dynamic_import(f"{base_path}/tools", "FilterGeneratedKeywordsTool")
        self.RepeatKeywordFilter = self.dynamic_import(f"{base_path}/tools", "FilterRepeatedKeywords")
        self.CoherenceReflector = self.dynamic_import(f"{base_path}/tools", "CoherenceRefelctionTool")
        self.RejectedKeywordsReflector = self.dynamic_import(f"{base_path}/tools", "RejectReflextionTool")
        self.SearchVolumeReflector = self.dynamic_import(f"{base_path}/tools", "KeywordSearchVolumeTool")

        self.s3_path = s3_path
        self.s3_bucket = s3_bucket
        self.good_kw_list = []
        self.attempt = 0
        self.successfull_pass = False
        self.running_week_day = tuple(int(day.strip()) for day in self.config['EXE']['RUNNING_WEEK_DAY'].split(','))

        self.code = code_in

        os.environ['SERPAPI_API_KEY'] = self.config['KEYS']['SERPAPI_API_KEY']
    def pre_process_csv(self, df):
        # Clean the Keyword column by stripping extra quotes
        df["Keyword"] = df["Keyword"].astype(str).str.strip('"').replace('"', '')
        df = df[(df["Keyword status"].str.upper() == "ENABLED") | (df["Keyword status"].str.upper() == "ELIGIBLE")]
        df = df.groupby("Keyword", group_keys=False).apply(lambda x: x.sample(1)).reset_index(drop=True)

         # Convert CTR column: Remove '%' and convert to a float fraction (if it exists)
        if "CTR" in df.columns:
            def process_ctr(x):
                try:
                    return float(x.rstrip("%")) / 100
                except Exception:
                    return None
            df["CTR"] = df["CTR"].astype(str).apply(process_ctr)
        
        # Convert numeric columns, e.g., Impr., Cost, Clicks, Conversions, Avg. CPC, Cost / conv.
        numeric_columns = ["Clicks", "Impr.", "Conversions", "Avg. CPC", "Cost", "Cost / conv."]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str)
                    .replace(['--', ''], np.nan)
                    .str.replace("[^0-9.]", "", regex=True),
                    errors="coerce"
                )
        
        # Clean column headers (remove extra spaces)
        df.columns = df.columns.str.strip()
        df["Conversion Rate"] = df["Conversions"] / df["Impr."]
        # Fill missing numeric values for selected columns
        df[["Clicks", "Impr.", "Conversions", "Conversion Rate", "Avg. CPC", "Cost", "Cost / conv."]] = df[["Clicks", "Impr.", "Conversions",
                                                                                "Conversion Rate", "Avg. CPC", "Cost", "Cost / conv."]].fillna(0)
        df["Avg. CPC"] = df["Avg. CPC"].replace(0, 1000)
        # For Avg. CPC, replace 0 values with 1000
        if str(self.config["KEYWORD"]["FILTERING_KEYWORDS"]) == "True":
            # Retrieve configuration values.
            boundary = int(self.config["KEYWORD"]["BOUNDARY_TARGET_CPA"])
            boundary_2 = boundary * 2
            out_1 = df[(5.0 <= df["Conversions"]) & (boundary_2 <= df["Cost / conv."])]
            out_2 = df[(1.0 <= df["Conversions"]) & (df["Conversions"] < 5.0) & (boundary_2 <= df["Cost / conv."]) & (50 <= df["Clicks"])]
            out_3 = df[(df["Conversions"] < 1.0) & (boundary <= df["Cost"]) & (30 <= df["Clicks"])]
            print(f'### {out_1=}', "Rule 1: コンバージョン数が5以上の場合、キーワードののこれまでのコンバージョン単価が目標の2倍以上")
            print(f'### {out_2=}', "Rule 2: コンバージョン数が１以上の場合、キーワードの、１：これまでのコンバージョン単価が目標コンバージョン単価の二倍以上、かつ、クリック数が５０以上の場合")
            print(f'### {out_3=}', "Rule 3: コンバージョン数が０の場合、キーワードの、1:これまでの消化金額が目標コンバージョン単価の一倍以上、かつ、クリック数が３０以上の場合、このルールは早期に判断するため作られたルールです")
            zero_cpc = df[df["Avg. CPC"] == 0]
            valid_df = df.drop(out_1.index).drop(out_2.index).drop(out_3.index).copy()
            filtered_out_df = df.drop(valid_df.index).copy()
            print(f'### {valid_df=}')
            print(f'### {filtered_out_df=}')
            return valid_df, filtered_out_df, out_1, out_2, out_3, zero_cpc
        else:
            # When filtering is disabled, use the default conversion condition.
            zero_cpc = df[df["Avg. CPC"] == 0]
            valid_df = df[df["Conversions"] >= 0].copy()
            filtered_out_df = df.drop(valid_df.index).copy()   
            # Extract unique keywords from the filtered-out rows
            #filtered_out_keywords = filtered_out_df["Keyword"].unique().tolist()
            # Return both the valid DataFrame and the list of filtered-out keywords
            print(f'### {valid_df=}')
            print(f'### {filtered_out_df=}')
            return valid_df, filtered_out_df, None, None, None, zero_cpc
    
    def topsis_score(self, updated_df, weights=[0.1, 0.1, 0.0, 0.5, 0.3],
                keys=["Clicks", "Impr.", "Conversions", "Conversion Rate", "Avg. CPC"],
                entropy_balance=False, threshold_non_data=0.0, cpa_threshold=1500):
        """
        Enhanced TOPSIS scoring function that properly prioritizes untested keywords (zero metrics)
        over inefficient keywords that waste ad spend (high clicks/impressions but low conversions).
        Now includes CPA calculation and penalty for keywords with CPA above threshold.
        
        Parameters:
        - updated_df: DataFrame containing keywords data
        - weights: Weights for each metric
        - keys: The metric column names
        - entropy_balance: Whether to use entropy for weight balancing
        - threshold_non_data: Minimum threshold for considering data as non-zero
        - cpa_threshold: Threshold for Cost Per Acquisition penalty (default: 1500)
        """
        def compute_entropy_weights(data):
            """
            Computes entropy-based weights for a (rows × metrics) numpy array.
            Rows = items (e.g., keywords), Columns = metrics (e.g., Clicks, CPC, etc.)
            """
            eps = 1e-12
            # 1) Normalize each column so their sum is 1
            col_sum = np.sum(data, axis=0) + eps
            p = data / col_sum

            # 2) Compute entropy e_j for each column j
            m = data.shape[0]  # number of items
            k = 1.0 / np.log(m + eps)
            # Avoid log(0) by adding eps and ensure p_ij * log(p_ij) = 0 if p_ij=0
            p_log_p = p * np.log(p + eps)
            e_j = -k * np.sum(p_log_p, axis=0)  # entropy for each metric

            # 3) Compute degree of divergence d_j = 1 - e_j
            d_j = 1.0 - e_j

            # 4) Entropy-based weights = d_j / Σ(d_j)
            weights = d_j / (np.sum(d_j) + eps)
            return weights
        
        # Create a copy to avoid modifying the original DataFrame
        df = updated_df.copy()
        
        # Replace "-" with 0.0 and convert to float
        df[keys] = df[keys].apply(lambda x: x.replace("-", "0.0") if isinstance(x, str) else x).astype(float)
        
        # Calculate CPA (Cost Per Acquisition)
        # Handle possible division by zero by using np.where
        df["Cost"] = df["Cost"].astype(float)
        df["Conversions"] = df[keys[2]].astype(float)
        
        # Add CPA calculation (Cost per Acquisition)
        eps = 1e-12  # Small epsilon to avoid division by zero
        # Use a numeric sentinel value (np.nan) instead of None for consistent numeric type
        df["CPA"] = np.where(df["Conversions"] > 0, 
                            df["Cost"] / df["Conversions"], 
                            np.nan)  # Set to NaN if no conversions to avoid penalizing untested keywords
        
        # Identify zero-metric keywords (no clicks, impressions, conversions)
        zero_metrics = (df[keys[0]] <= threshold_non_data) & \
                    (df[keys[1]] <= threshold_non_data) & \
                    (df[keys[2]] <= threshold_non_data)
        
        # Identify wasteful keywords (high clicks/impressions, low conversions)
        has_conversions = df[keys[2]] > threshold_non_data
        low_conv_rate = df[keys[3]] < np.median(df.loc[has_conversions, keys[3]]) if any(has_conversions) else False
        wasteful = has_conversions & low_conv_rate
        
        # Extract data from DataFrame
        data = df[keys].to_numpy(dtype=float)
        
        # Update global min and max values
        self.global_max_click = df[keys[0]].max()
        self.global_min_click = df[keys[0]].min()
        self.global_max_search_volume = df[keys[1]].max()
        self.global_min_search_volume = df[keys[1]].min()
        self.global_max_conversion = df[keys[2]].max()
        self.global_min_conversion = df[keys[2]].min()
        self.global_max_converstion_rate = df[keys[3]].max()
        self.global_min_converstion_rate = df[keys[3]].min()
        self.global_max_cpc = df[keys[4]].max()
        self.global_min_cpc = df[keys[4]].min()
        
        # Store min/max values
        self.global_min_list = [
            self.global_min_click, 
            self.global_min_search_volume, 
            self.global_min_conversion, 
            self.global_min_converstion_rate, 
            self.global_min_cpc
        ]
        
        self.global_max_list = [
            self.global_max_click, 
            self.global_max_search_volume, 
            self.global_max_conversion, 
            self.global_max_converstion_rate, 
            self.global_max_cpc
        ]
        
        global_min_arr = np.array(self.global_min_list)
        global_max_arr = np.array(self.global_max_list)
        
        # Avoid division by zero in normalization
        denom = global_max_arr - global_min_arr
        denom[denom == 0] = 1e-12
        
        # Normalize the data
        data_norm = (data - global_min_arr) / denom
        
        # Define which metrics are beneficial (higher is better)
        beneficial_arr = np.array([True, True, True, True, False], dtype=bool)
        
        # Flip non-beneficial metrics
        for j, is_beneficial in enumerate(beneficial_arr):
            if not is_beneficial:
                data_norm[:, j] = 1 - data_norm[:, j]
        
        # Special handling for clicks and impressions for wasteful keywords
        # For rows with conversions > 0 but poor conversion rates, high clicks is BAD
        if any(wasteful):
            # Flip the normalization for clicks and impressions for wasteful keywords
            # This makes high clicks/impressions BAD for wasteful keywords
            data_norm[wasteful, 0] = 1 - data_norm[wasteful, 0]  # Clicks
            data_norm[wasteful, 1] = 1 - data_norm[wasteful, 1]  # Impressions
        
        # Compute weights
        if entropy_balance:
            weights_arr = compute_entropy_weights(data)
        else:
            weights_arr = np.array(weights, dtype=float)
        
        # Apply weights to normalized data
        weighted_data = data_norm * weights_arr
        
        # Get ideal best and worst
        ideal_best = weighted_data.max(axis=0)
        ideal_worst = weighted_data.min(axis=0)
        
        # Calculate distances
        dist_to_best = np.sqrt(np.sum((weighted_data - ideal_best) ** 2, axis=1))
        dist_to_worst = np.sqrt(np.sum((weighted_data - ideal_worst) ** 2, axis=1))
        
        # Calculate TOPSIS scores
        scores = dist_to_worst / (dist_to_best + dist_to_worst + 1)
        
        # Special adjustment for zero-metric keywords
        # We want them to score better than wasteful keywords
        if any(zero_metrics) and any(wasteful):
            # Find the median score of wasteful keywords
            wasteful_scores = scores[wasteful]
            
            # Calculate a boosted score for zero-metric keywords
            # Set it to be slightly higher than the median wasteful score
            median_wasteful = np.median(wasteful_scores)
            boost_factor = 1.2  # Adjust as needed
            
            # Only boost zero-metric keywords that would otherwise score lower than wasteful keywords
            boost_needed = zero_metrics & (scores < median_wasteful * boost_factor)
            if any(boost_needed):
                scores[boost_needed] = median_wasteful * boost_factor
        
        # Apply CPA penalty to keywords with CPA above threshold
        # Identify keywords with high CPA (above threshold)
        high_cpa = (df["CPA"] > cpa_threshold) & (df["CPA"] != float('inf'))
        
        # Apply penalty to high CPA keywords
        if any(high_cpa):
            # Calculate penalty factor based on how much CPA exceeds threshold
            # For example: 50% over threshold = 0.5 penalty
            penalty_factor = np.minimum(1.0, np.maximum(0.2, (df["CPA"] - cpa_threshold) / cpa_threshold))
            scores[high_cpa] = scores[high_cpa] * (1 - penalty_factor[high_cpa])
        
        # Add scores to the original DataFrame
        updated_df["Topsis Score"] = scores
        updated_df["CPA"] = df["CPA"]
        
        return updated_df

    def dynamic_import(self, module_name, function_name):
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, function_name)
        return cls

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
            [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in
             action_int_dic[branded].items() for kw in kws],
            columns=['Category', 'Keyword', 'Branded']
        )

        # Filter out keywords that meet the threshold
        new_keywords_df = new_keywords_df[new_keywords_df['Keyword'].apply(
            lambda kw: int(kw) >= int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']))]

        if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
            new_keywords_df.to_csv(self.generated_keyword_data, index=False)
        else:
            # full_path = self.save_dir + self.generated_keyword_data[2:]
            buffer = io.BytesIO()
            new_keywords_df.to_csv(buffer, index=False)
            buffer.seek(0)
            client = boto3.client('s3')
            client.put_object(Bucket=self.s3_bucket, Key=self.save_dir + self.generated_keyword_data,
                              Body=buffer.getvalue())

        return new_keywords_df

    def multi_shot_prompt_create(self, num_good_categories,num_keywords_per_category, num_new_categories, num_keywords_per_new_category,
                                 rejected_keywords_string, good_kw_string,
                                 good_keywords, bad_keyword, generated_category, bad_category,
                                 step=0):
        def format_few_shot_examples(cluster_centers_keywords, cluster_stats):
            few_shot_examples = []
            for index, (keyword, stats) in enumerate(zip(cluster_centers_keywords, cluster_stats)):
                highest_keyword, highest_click = stats["keyword_with_max_estimated_traffic"]
                lowest_keyword, lowest_click = stats["keyword_with_min_estimated_traffic"]
                example_str = \
                    f"""
                          {index + 1}. {keyword}:
                          - Center Keyword is {keyword} and average Topsis Score of this cluster: {stats.get('Topsis Score'):,.3f}
                          - The Keyword with highest Topsis Score in this cluster is {highest_keyword} with {highest_click:,.3f} score
                          - The Keyword with lowest Topsis Score in this cluster is {lowest_keyword} with {lowest_click:,.3f} score
                          - Description: {stats.get('description')}
                    """
                few_shot_examples.append(example_str)
            return "\n".join(few_shot_examples)


        def generate_keyword_clicks(keywords, good_keywords):
            click_indicator = "low" if good_keywords is False else "high"
            keyword_str = f"\nThe following keywords are identified with {click_indicator} Topsis Score."
            for keyword in keywords["Topsis Score"]:
                keyword_str += f"\nKeyword: {keyword[0]}, Topsis Score: {keyword[1]:,.3f}, Category: {keyword[2]}"
            if good_keywords is False:
                keyword_str += "\nYou should try to avoid generating those keywords again.\n"
            else:
                keyword_str += "\nYou are required to do not generate bad keywords with low Topsis Score. You should maintain those good keywords in this round keyword generations meaning that those good keyword should be kept identitcally. \n"
            return keyword_str

        def keyword_analysis(new_words_stats, sorted_cluster_indices, sorted_clusters):
            generated_keyword_str = ""
            for i in range (len(new_words_stats)):
                for new_keyword in new_words_stats[i].keys():
                    new_keyword_click_rank = sorted_cluster_indices.index(new_words_stats[i][new_keyword]["idx"]) + 1
                    generated_keyword_str += f"\nFor previously generated keywords [{new_keyword}] at round {step}, they are close to the cluster index {new_words_stats[i][new_keyword]["idx"]}, their average Topsis Score is at rank {new_keyword_click_rank} with average Topsis Score of {sorted_clusters[new_keyword_click_rank-1][1]:,.3f}, we computed its similarity with those aformementioned clusters above and seems it belongs to the keyword cluster followed:\n "
                    for keyword, click in zip(new_words_stats[i][new_keyword]["cluster_keyword_list"], new_words_stats[i][new_keyword]["click_list"]):
                        generated_keyword_str += \
                            f"""
                        {keyword}:
                        - Topsis Score: {click:,.3f}\n
                        """
                    else:
                        generated_keyword_str += f"\nFor previously generated keywords [{new_keyword}] at round {step} from custer index {new_words_stats[i][new_keyword]["idx"]}, their similarities with existing clusters are beyond two standard deviations so they are considered as new keywords that does not belong to any cluster.\n"
            return generated_keyword_str

        def generate_initial_categories(category_list):
            initial_categories = ""
            for i, keyword in enumerate(category_list):
                if i == len(category_list) - 1:
                    initial_categories += keyword
                else:
                    initial_categories += keyword + ", "
            return initial_categories
        if good_keywords is None and bad_keyword is None and generated_category is None and bad_category is None:
            few_shot_examples = "Though we wanted to provide you some keywords example, after filtering those previous keyword, we found they are not good keywords which you do not have to study them, so you have generate keywords without any reference this time. For the good categories that we will ask you to generate in following prompt, change it to explore category."
            initial_categories = "No initial categories as all keywords in previous data are bad."
            good_kw = "No intial good keywords as all keywords in previous data are bad."
            generated_category_str = "No initial generate cateogries as all keywords in previous data are bad."
        else:
            few_shot_examples = format_few_shot_examples(self.cluster_centers_keywords, self.cluster_stats)
            good_kw = ""
            for data in good_keywords.iterrows():
                idx, row = data
                good_kw += f"\nKeyword {row["Keyword"]},  Topsis Score: {row["Topsis Score"]:,.3f}, Category: {row['Category']}\n"
            generated_category_str = ""
            total_category = generated_category
            total_category = total_category.extend(bad_category)
            for idx, category_info in enumerate(generated_category):
                generated_category_str += f"""
                                Rank {idx + 1}: {category_info["Category"]}
                                - Average Topsis Score: {category_info["Total Topsis Score"]:,.3f}
                                """
                for rank_in_cluster, keyword_info in enumerate(category_info["Keywords Details"]):
                    generated_category_str += f"""
                                    - {keyword_info["Keyword"]}: Topsis Score: {keyword_info["Topsis Score"]:,.3f}, rank: {rank_in_cluster + 1}
                                """
            generated_category_str += f"\nDo not generate keywords that are identified with low Topsis Score for category {generate_initial_categories([x for x in bad_keyword['Keyword'].to_list()])}, keep generate keywords that are identified with good Topsis Score in this category.\n"
            generated_category_str += f"Do not generate keywords for category {generate_initial_categories([x["Category"] for x in bad_category])} as even their average Topsis Score is low, so you should discard all keywords in this category.\n"
            initial_categories = generate_initial_categories([x["Category"] for x in generated_category])
        input_prompt = dynamic_multi_shot_single_query_prompt.format(
                good_kw_string = ", ".join(self.good_kw_list),
                num_good_categories = num_good_categories,
                num_keywords_per_category=num_keywords_per_category,
                num_new_categories=num_new_categories,
                num_keywords_per_new_category=num_keywords_per_new_category,
                rejected_keywords_string=rejected_keywords_string,
                product_name=str(self.config['SETTING']['product']),
                few_shot_examples=few_shot_examples,
                initial_categories=initial_categories,
                kept_good_keywords=good_kw,
                generated_category=generated_category_str,
                language=self.config["SETTING"]["language"])
        return input_prompt

    def sort_topsis(self, predicted_df):
        # ---------------------------
        
        zero_keywords_df = predicted_df[predicted_df["Topsis Score"] == 0].copy()
        
        # Filter out zero TOPSIS scores to compute the threshold
        non_zero_df = predicted_df[predicted_df["Topsis Score"] > 0]

        # Calculate the threshold for the top 25% of non-zero TOPSIS scores
        keyword_threshold = non_zero_df["Topsis Score"].quantile(0.75)

        # Select top and bad keywords from the non-zero keywords
        top_keywords_df = non_zero_df[non_zero_df["Topsis Score"] >= keyword_threshold].copy()
        top_keywords_df.sort_values("Topsis Score", ascending=False, inplace=True)

        bad_keywords_df = non_zero_df[non_zero_df["Topsis Score"] < keyword_threshold].copy()
        bad_keywords_df = pd.concat([zero_keywords_df, bad_keywords_df]).copy()
        bad_keywords_df.sort_values("Topsis Score", ascending=False, inplace=True)

        category_agg = predicted_df.groupby("Category", as_index=False)["Topsis Score"].mean()

        category_threshold = category_agg["Topsis Score"].quantile(0.75)
        top_categories_df = category_agg[category_agg["Topsis Score"] >= category_threshold].copy()
        top_categories_df.sort_values("Topsis Score", ascending=False, inplace=True)

        bad_categories_df = category_agg[category_agg["Topsis Score"] < category_threshold].copy()
        bad_categories_df.sort_values("Topsis Score", ascending=False, inplace=True)

        # ---------------------------
        top_category_details = []
        for idx, row in top_categories_df.iterrows():
            cat = row["Category"]
            total_score = row["Topsis Score"]

            keywords_in_cat = predicted_df[predicted_df["Category"] == cat].copy()
            keywords_in_cat.sort_values("Topsis Score", ascending=False, inplace=True)

            keyword_list = keywords_in_cat["Keyword"].tolist()
            top_category_details.append({
                "Category": cat,
                "Total Topsis Score": total_score,
                "Keywords": keyword_list,
                "Keywords Details": keywords_in_cat.to_dict(orient="records")
            })

        # ---------------------------
        bad_category_details = []
        for idx, row in bad_categories_df.iterrows():
            cat = row["Category"]
            total_score = row["Topsis Score"]

            keywords_in_cat = predicted_df[predicted_df["Category"] == cat].copy()
            keywords_in_cat.sort_values("Topsis Score", ascending=False, inplace=True)

            keyword_list = keywords_in_cat["Keyword"].tolist()
            bad_category_details.append({
                "Category": cat,
                "Total Topsis Score": total_score,
                "Keywords": keyword_list,
                "Keywords Details": keywords_in_cat.to_dict(orient="records")
            })

        return top_keywords_df, top_category_details, bad_keywords_df, bad_category_details

    # def run(self, step = 0, start_month = '2023-06', end_month = '2024-09', google_account_id='1252903913'):]
    #stopped keywrod list neab
    def run(self, cb_get_kw_metrics, cb_exec_kw_plan, current_keyword_list, computing=True):
        if computing == True:
            self.filtered_df = {
                "df_rule1": None,
                "df_rule2": None,
                "df_rule3": None,
                "df_non_data": None,
            }
        if self.code == 0 and self.setting_day.weekday() not in self.running_week_day:
            return [], {}, int(2), []
        if self.setting_day.day <= 7:
            is_init = True
        else:
            is_init = False

        if ((self.setting_day.weekday() in self.running_week_day) and is_init and self.code == 0) or (
                self.code == 1 and is_init):
            # if (self.setting_day.weekday() == 6 and self.setting_day.day <= 7 and self.code == 0) or (self.code == 1 and self.setting_day.day <= 7) :

            with open(self.rejected_keyword_list, 'r') as file:
                rejected_kw_list = [line.strip() for line in file]
            rejected_kw_list = list(set(rejected_kw_list))
            Aggrerator_init_flag = True

            
            original_df = pd.read_csv(self.history_df_data, encoding="utf-16", sep="\t").drop(columns=["Final URL",
                                                                                                      "Mobile final URL",
                                                                                                      "Status reasons"
                                                                                                      ], errors='ignore')
            #KW_loader = CSVLoader(df)
            df, invalid_keyword, filtered_df_rule1, filtered_df_rule2, filtered_df_rule3, zero_cpc = self.pre_process_csv(original_df)
            self.filtered_df = {
                "df_rule1": filtered_df_rule1,
                "df_rule2": filtered_df_rule2,
                "df_rule3": filtered_df_rule3,
                "df_non_data": zero_cpc,
            }
            self.history_df = original_df
            if len(df) == 0:
                df = self.history_df
                empty = True
            else:
                df = self.topsis_score(df)
                self.history_df = df
                empty = False
            if computing == True:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                self.clustering, self.cluster_centers_keywords, self.cluster_centers_indices, self.cluster_stats,  self.embeddings, self.category_list = create_clustering(self.history_df, self.config, base_dir, num_shot=len(self.history_df))
            if empty == True:
                sorted_by_keyword_good, sorted_by_category_good, sorted_by_keyword_bad, sorted_by_category_bad = None, None, None, None
            else:
                sorted_by_keyword_good, sorted_by_category_good, sorted_by_keyword_bad, sorted_by_category_bad = self.sort_topsis(df)
            original_keywords = [item["keyword"] for item in current_keyword_list]
            #KW_retriever = self.customized_trend_retriever(KW_loader,
            #                                               str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']), str(
            #        self.config['KEYS']['OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

            # 2. define a retriever_tool
           # KW_retriever_tool = create_retriever_tool(
           #     KW_retriever,
            #    str(self.config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
           #     str(self.config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
           # )

            # 4. exampler tool
            exampler_loader = TextLoader(str(self.rule_data))
            exampler_retriever = self.customized_trend_retriever(exampler_loader,
                                                                 str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),
                                                                 str(self.config['KEYS'][
                                                                         'OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

            # define a retriever_tool
            exampler_retriever_tool = create_retriever_tool(
                exampler_retriever,
                str(self.config['TOOL']['RULE_RETRIEVAL_NAME']),
                str(self.config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
            )

            search_tool = load_tools(["serpapi"])
            search_tool[0].name = "google_search"

            # 3. Initilize LLM and the agent chain
            if self.config['LLM']['MODEL'] == 'GPT-4o':
                llm = AzureChatOpenAI(
                    deployment_name=str(self.config['LLM']['gpt4o_deployment_name']),
                    openai_api_version=str(self.config['LLM']['gpt4o_openai_api_version']),
                    openai_api_key=str(self.config['KEYS']['OPENAI_GPT4O_API_KEY']),
                    azure_endpoint=str(self.config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']),
                    temperature=float(self.config['LLM']['temperature'])
                )
            elif self.config['LLM']['MODEL'] == 'o3-mini':
                llm = AzureChatOpenAI(
                    deployment_name=str(self.config['LLM']['o3_deployment_name']),
                    openai_api_version=str(self.config['LLM']['o3_openai_api_version']),
                    openai_api_key = str(self.config['KEYS']['OPENAI_o3mini_API_KEY']),
                    azure_endpoint=str(self.config['KEYS']['OPENAI_o3mini_AZURE_OPENAI_ENDPOINT']),
                )

            
            prompt = hub.pull("hwchase17/react")
            prompt.template = react
            self.WordFilter.rejected_keywords = rejected_kw_list
            self.RepeatKeywordFilter.history_keywords =  original_df["Keyword"].tolist()
            search_volume_function = self.get_keyword_planner_metrics
            tools = [
                self.getSerpTool(region="us" if self.config["SETTING"]["language"] == 'English' else "jp",
                                 language="us" if self.config["SETTING"]["language"] == 'English' else "jp"),
                self.ClickAggregator(self.initial_keyword_data, self.config, self.s3_bucket, self.s3_path,
                                     Aggrerator_init_flag),
                self.RepeatKeywordFilter(history_keywords=original_df["Keyword"].tolist()),
                self.WordFilter(rejected_keywords=rejected_kw_list),
                self.CoherenceReflector(product=str(self.config["SETTING"]['product']), config=self.config),
                self.RejectedKeywordsReflector(product=str(self.config["SETTING"]['product']), config=self.config, reject_keywords=rejected_kw_list),
                self.SearchVolumeReflector(config=self.config, s3_bucket=self.s3_bucket, save_dir=self.save_dir, rejected_keyword_list=self.rejected_keyword_list, get_keyword_planner_metrics=search_volume_function)]
            agent_chain = create_react_agent(
                llm,
                tools,
                prompt
            )
            agent_executor = AgentExecutor(agent=agent_chain, tools=tools, return_intermediate_steps=True, verbose=True, max_iterations=50,
                                           handle_parsing_errors='Check your output and make sure it either uses: Correct Action/Action Input syntax, or Outputs the Final Answer directly after the thought in the form of  "Thought: I now know the final answer\nFinal Answer: the final answer to the original input question"')

            # print("Reflection is unabled")
            print("the first step")

            # Define the hyperparameters
            num_good_categories = int(self.config['KEYWORD']['NUM_GOOD_CATEGORIES'])
            num_keywords_per_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY'])
            num_new_categories = int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])
            num_keywords_per_new_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_NEW_CATEGORY'])

            self.good_kw_list = [s for s in self.good_kw_list if s not in df["Keyword"].tolist()]
            rejected_kw_list = list(set(rejected_kw_list))

            rejected_keywords_string = ", ".join(rejected_kw_list[-30:0])  # Converts list to string
            good_kw_string = ", ".join(self.good_kw_list)  # Converts list to string

            # 4. Process the first prompt
            # Define the prompt with placeholders for the hyperparameters

            # Format the prompt with the hyperparameters
            #first_prompt = prompt_template.format(num_keywords_per_category, num_new_categories,
            #                                      num_keywords_per_new_category, rejected_keywords_string,
            #                                      good_kw_string, str(self.config["SETTING"]['product']),
            #                                      self.config["SETTING"]["language"]) 
            first_prompt = self.multi_shot_prompt_create(num_good_categories,
                                                         num_keywords_per_category, num_new_categories,
                                                         num_keywords_per_new_category,
                                                         rejected_keywords_string, good_kw_string, sorted_by_keyword_good,
                                                         sorted_by_keyword_bad, sorted_by_category_good, sorted_by_category_bad)
            # 5. Output the first qustion and Run the agent chain

            print("Question: " + first_prompt)

            action_int_dic, scratch_pad, code_out = self.run_with_retries(agent_executor, first_prompt,
                                                                          int(self.config['LLM']['max_attempts']))

            if code_out == 4:
                return [], {}, int(4), []

            if str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"]) == 'True':
                all_branded = []
                for key in action_int_dic["Branded"]:
                    all_branded.extend(action_int_dic["Branded"][key])

                # Create a list of all strings in 'Non-Branded'
                all_non_branded = []
                for key in action_int_dic["Non-Branded"]:
                    all_non_branded.extend(action_int_dic["Non-Branded"][key])

                if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                    keyword_list_needed = all_non_branded
                else:
                    keyword_list_needed = all_branded


                if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                    keywords_info_needed = all_non_branded
                else:
                    keywords_info_needed = all_branded
                #keyword_list_needed = [info["keyword"] for info in keywords_info_needed]
                # this should be replaced by the func. of Ascade san
                if True:
                    metrics = self.get_keyword_planner_metrics(
                        '1252903913',
                        keyword_list_needed,
                        '2023-06',
                        '2024-06',
                    )
                    # Wait for 10 seconds
                    time.sleep(10)
                    # Normalize keyword casing for accurate matching
                    # keyword_data_normalized = {k['Keyword'].lower(): k['Avg. monthly searches'] for k in metrics}
                    keyword_to_search_volume = {
                        self.normalize_keyword(metric['Keyword']): metric['Avg. monthly searches'] for metric in
                        metrics}
                    # Extract 'Avg. monthly searches' for keywords in the list
                    # new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in keyword_list_needed]
                    new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword in
                                       keyword_list_needed]




            elif str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"]) == 'False':
                new_words_check = [60, 70, 80, 90, 100, 100, 100, 100, 100]

            else:
                raise ValueError(
                    "Invalid value for SEARCH_VOLUMN_CHECK: " + str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"]))


            while self.attempt < int(self.config['LLM']['max_attempts']):

                # if all the element in new_words_check is over 50, break the loop
                if all(x >= int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']) for x in new_words_check):
                    self.successfull_pass = True
                    # add the new generated keywords to the /data/initial_KW.csv
                    # 1. covert the dic to dataphrame
                    new_keywords_df = pd.DataFrame(
                        [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in
                         action_int_dic[branded].items() for kw in kws],
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
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        df = df[df['Branded'] == False]
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        df = df[df['Branded'] == True]
                    else:
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))

                    # df['Clicks'] = [random.randint(1,100) for k in df.index]

                    with open(self.rejected_keyword_list, 'w') as file:
                        for item in list(set(rejected_kw_list)):
                            file.write("%s\n" % item)

                    df.to_csv(self.generated_keyword_data, index=False)

                    break


                else:
                    for i in range(len(new_words_check)):
                        if new_words_check[i] < int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']):
                            # add the low search check new words to the tried_kw_list
                            rejected_kw_list.append(str(keyword_list_needed[i]))
                            print("The new words whose search check is less than 5 is: " + str(keyword_list_needed[i]))
                            # save the rejected_kw_list to a file
                            with open(self.rejected_keyword_list, 'w') as file:
                                for item in rejected_kw_list:
                                    file.write("%s\n" % item)

                        else:
                            # add keywords to the self.good_kw_list
                            self.good_kw_list.append(str(keyword_list_needed[i]))
                            self.good_kw_list = list(set(self.good_kw_list))

                    self.attempt += 1

                    return self.run(cb_get_kw_metrics, cb_exec_kw_plan, current_keyword_list, computing=False)

                    # print("Next Round, the rejected low search keywords are: " + str(rejected_kw_list))
            # response = agent_chain ({"input":  first_prompt})
            if not self.successfull_pass:
                # Flatten the dictionary into a list of keywords without considering the category
                # keywords_flat_list = [kw for sublist in action_int_dic["Non-Branded"].values() for kw in sublist]
                # Flatten the dictionary into a list of tuples (category, keyword)
                if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                    keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Non-Branded"].items() for kw
                                                in kws]
                else:
                    keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Branded"].items() for kw in
                                                kws]
                # keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Non-Branded"].items() for kw in kws]

                # Filter keywords by matching them with their corresponding search amount
                # good_keywords = [kw for kw, check in zip(keywords_flat_list, new_words_check) if check > int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD'])]
                good_keywords = [(cat, kw) for (cat, kw), check in zip(keywords_with_categories, new_words_check) if
                                 check > int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD'])]

                if good_keywords:
                    # Create DataFrame from good keywords
                    new_keywords_df = pd.DataFrame(good_keywords, columns=['Category', 'Keyword'])
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
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))

                    # Remove rows with duplicate values in the 'Keyword' column, keeping the first occurrence
                    df = df.drop_duplicates(subset='Keyword', keep='first')

                    df['Clicks'] = df['Clicks'].fillna(0)
                    # df['Clicks'] = [random.randint(1, 100) for _ in df.index]  # Optional, simulate clicks

                df.to_csv(self.generated_keyword_data, index=False)

                self.good_kw_list = []

        elif ((self.setting_day.weekday() in self.running_week_day) and (not is_init) and self.code == 0) or (
                (not is_init) and self.code == 1):
            # elif (self.setting_day.weekday() == 6 and self.setting_day.day > 7 and self.code == 0 ) or (self.setting_day.day > 7 and self.code == 1):

            # get data from call back function
            # df = cb_get_kw_metrics(self.observation_period)

            if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                with open(self.rejected_keyword_list, 'r') as file:
                    rejected_kw_list = [line.strip() for line in file]
                rejected_kw_list = list(set(rejected_kw_list))

            elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                # Initialize the S3 client
                s3_client = boto3.client('s3')

                # Get the object from the bucket
                obj = s3_client.get_object(Bucket=self.s3_bucket, Key=self.save_dir + self.rejected_keyword_list)

                # Read data from the S3 object
                data = obj['Body'].read().decode('utf-8')

                # Split data into lines and strip whitespace
                rejected_kw_list = [line.strip() for line in data.splitlines()]
                rejected_kw_list = list(set(rejected_kw_list))

            Aggrerator_init_flag = False

            if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                df = pd.read_csv(self.config["SETTING"]["history_data"], encoding="utf-16", sep="\t").drop(columns=["Final URL",
                                                                                                        "Mobile final URL",
                                                                                                        "Status reasons"
                                                                                                        ], errors='ignore')
                #self.history_df
                df, invalid_keyword, filtered_df_rule1, filtered_df_rule2, filtered_df_rule3, zero_cpc = self.pre_process_csv(df)
                self.filtered_df = {
                "df_rule1": filtered_df_rule1,
                "df_rule2": filtered_df_rule2,
                "df_rule3": filtered_df_rule3,
                "df_non_data": zero_cpc,
                }
                df = self.topsis_score(df)
                sorted_by_keyword_good, sorted_by_category_good, sorted_by_keyword_bad, sorted_by_category_bad = self.sort_topsis(
                    df)
                self.history_df = df
                if computing == True:
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    self.clustering, self.cluster_centers_keywords, self.cluster_centers_indices, self.cluster_stats,  self.embeddings, self.category_list = create_clustering(self.history_df, self.config, base_dir, num_shot=len(self.history_df))
                    
                KW_loader = CSVLoader(self.history_df_data, encoding="utf-16")
                KW_retriever = self.customized_trend_retriever(KW_loader,str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),str(self.config['KEYS'][ 'OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))
                original_keywords = df["Keyword"].tolist()
            elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                # get the dataphrame
                df = cb_get_kw_metrics(self.observation_period)
                if 0 < len(df):
                    df['Avg. CPC'] = df['cost'] / df['clicks']
                    df['Cost / conv.'] = df['cost'] / df['conversions']
                    df = df.replace([np.inf, -np.inf, np.nan], 0.0)
                    # only keep the colomn with name  'keyword' and 'clicks'
                    kw_df = df[['keyword', 'status', 'clicks', "impressions", "conversions", "cost", "Avg. CPC", "Cost / conv."]]
                    original_keywords = [item["keyword"] for item in current_keyword_list]
                    # original_keywords = df["keyword"].tolist()
                    # Initialize S3 client
                    s3_read_client = boto3.client('s3')

                    # Get the object
                    response = s3_read_client.get_object(Bucket=self.s3_bucket,
                                                        Key=self.save_dir + self.generated_keyword_data)

                    # Read the data into a pandas DataFrame
                    df_ref = pd.read_csv(io.BytesIO(response['Body'].read()))

                    # merge the two dataframes on 'Keyword' and 'Category'
                    merged_df = kw_df.merge(df_ref[['Keyword', 'Category']], left_on="keyword", right_on="Keyword", how="left")
                    #only need keyword from df rather than merged df
                    merged_df = merged_df[merged_df['keyword'].isin(df['keyword'])]
                    df = merged_df[['keyword', 'status', 'clicks', "impressions", "conversions", "cost", "Avg. CPC", "Cost / conv.", 'Category']]

                    # change the colomn name
                    df = df.rename(columns={'keyword': 'Keyword'})
                    df = df.rename(columns={'status': 'Keyword status'})
                    df = df.rename(columns={'clicks': 'Clicks'})
                    df = df.rename(columns={'impressions': 'Impr.'})
                    df = df.rename(columns={'conversions': 'Conversions'})
                    df = df.rename(columns={'cost': 'Cost'})
                else:
                    original_keywords = []
                    df = pd.read_csv(
                        self.config["SETTING"]["history_data"],
                        encoding="utf-16",
                        sep="\t"
                        ).drop(columns=["Final URL", "Mobile final URL", "Status reasons"])
                df, invalid_keyword, filtered_df_rule1, filtered_df_rule2, filtered_df_rule3, zero_cpc = self.pre_process_csv(df)
                    
                original_df = copy.copy(df)
                self.filtered_df = {
                "df_rule1": filtered_df_rule1,
                "df_rule2": filtered_df_rule2,
                "df_rule3": filtered_df_rule3,
                "df_non_data": zero_cpc,
                }
                self.history_df = original_df
                if len(df) == 0:
                    df = self.history_df
                    empty = True
                else:
                    df = self.topsis_score(df)
                    self.history_df = df
                    empty = False
                if computing == True:
                    self.clustering, self.cluster_centers_keywords, self.cluster_centers_indices, self.cluster_stats,  self.embeddings, self.category_list = create_clustering(self.history_df, self.config, os.path.expanduser("~/LLMAgent4AdText"), num_shot=len(self.history_df))
                if empty == True:
                    sorted_by_keyword_good, sorted_by_category_good, sorted_by_keyword_bad, sorted_by_category_bad = None, None, None, None
                else:
                    sorted_by_keyword_good, sorted_by_category_good, sorted_by_keyword_bad, sorted_by_category_bad = self.sort_topsis(df)
                #df, invalid_keyword = self.pre_process_csv(df)
                # df = self.topsis_score(df )
                # self.history_df = df
                # if computing == True:
                #     self.clustering, self.cluster_centers_keywords, self.cluster_centers_indices, self.cluster_stats,  self.embeddings, self.category_list = create_clustering(self.history_df, self.config, os.path.expanduser("~/LLMAgent4AdText") ,num_shot=len(self.history_df))
                # sorted_by_keyword_good, sorted_by_category_good, sorted_by_keyword_bad, sorted_by_category_bad =self.sort_topsis(df)
                # save it to a new csv file in the s3
                full_path = self.save_dir + self.generated_keyword_data[2:]
                buffer = io.BytesIO()
                df.to_csv(buffer, index=False)
                body = buffer.getvalue()
                client = boto3.client('s3')
                client.put_object(Bucket=self.s3_bucket, Key=self.save_dir + self.generated_keyword_data, Body=body)
                # KW_loader = CSVLoader(self.generated_keyword_data)
                KW_loader = S3FileLoader(bucket=self.s3_bucket, key=self.save_dir + self.generated_keyword_data)
                KW_retriever = self.customized_trend_retriever(KW_loader,
                                                               str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),
                                                               str(self.config['KEYS'][
                                                                       'OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

            # 2. define a retriever_tool
            KW_retriever_tool = create_retriever_tool(
                KW_retriever,
                str(self.config['TOOL']['GOOD_KW_RETRIEVAL_NAME']),
                # 'Search',
                str(self.config['TOOL']['GOOD_KW_RETRIEVAL_DISCRPTION']),
            )

            # 3. rule tool
            # rule_loader = TextLoader(str(self.config['FILE']['DOMAIN_KNOWLEDGE_FILE']))

            # 4. exampler tool
            exampler_loader = TextLoader(str(self.rule_data))
            exampler_retriever = self.customized_trend_retriever(exampler_loader,
                                                                 str(self.config['KEYS']['OPENAI_EMBEDDING_API_KEY']),
                                                                 str(self.config['KEYS'][
                                                                         'OPENAI_EMBEDDING_AZURE_OPENAI_ENDPOINT']))

            # define a retriever_tool
            exampler_retriever_tool = create_retriever_tool(
                exampler_retriever,
                str(self.config['TOOL']['RULE_RETRIEVAL_NAME']),
                # 'Search',
                str(self.config['TOOL']['RULE_RETRIEVAL_DISCRPTION']),
            )

            search_tool = load_tools(["serpapi"])
            search = SerpAPIWrapper()
            # ロードしたツールの中から一番目のものの名前を変更
            # https://book.st-hakky.com/data-science/agents-of-langchain/
            search_tool[0].name = "google_search"

            # 3. Initilize LLM and the agent chain
            if self.config['LLM']['MODEL'] == 'GPT-4o':
                llm = AzureChatOpenAI(
                    deployment_name=str(self.config['LLM']['gpt4o_deployment_name']),
                    openai_api_version=str(self.config['LLM']['gpt4o_openai_api_version']),
                    openai_api_key=str(self.config['KEYS']['OPENAI_GPT4O_API_KEY']),
                    azure_endpoint=str(self.config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']),
                    temperature=float(self.config['LLM']['temperature'])
                )
            elif self.config['LLM']['MODEL'] == 'o3-mini':
                llm = AzureChatOpenAI(
                    deployment_name=str(self.config['LLM']['o3_deployment_name']),
                    openai_api_version=str(self.config['LLM']['o3_openai_api_version']),
                    openai_api_key = str(self.config['KEYS']['OPENAI_o3mini_API_KEY']),
                    azure_endpoint=str(self.config['KEYS']['OPENAI_o3mini_AZURE_OPENAI_ENDPOINT']),
                )
    

            prompt = hub.pull("hwchase17/react")
            prompt.template = react
            # from tools import CustomCounterTool2
            self.WordFilter.rejected_keywords = rejected_kw_list
            self.RepeatKeywordFilter.history_keywords =  df["Keyword"].tolist()
            self.CoherenceReflector.product = str(self.config["SETTING"]['product'])
            search_volume_function = self.get_keyword_planner_metrics if str(self.config["EXE"]["S3_DEPLOY"]) == 'False' else cb_exec_kw_plan
            tools = [
                self.getSerpTool(region="us" if self.config["SETTING"]["language"] == 'English' else "jp",
                                    language="us" if self.config["SETTING"]["language"] == 'English' else "jp"),
                self.ClickAggregator(self.generated_keyword_data, self.config, self.s3_bucket, self.s3_path,
                                Aggrerator_init_flag),
                self.RepeatKeywordFilter(history_keywords=original_df["Keyword"].tolist()),
                self.WordFilter(rejected_keywords=rejected_kw_list),
                self.CoherenceReflector(product=str(self.config["SETTING"]['product']), config=self.config),
                self.RejectedKeywordsReflector(product=str(self.config["SETTING"]['product']), config=self.config, reject_keywords=rejected_kw_list),
                self.SearchVolumeReflector(config=self.config, s3_bucket=self.s3_bucket, save_dir=self.save_dir, rejected_keyword_list=self.rejected_keyword_list, get_keyword_planner_metrics=search_volume_function)]
            agent_chain = create_react_agent(
                llm,
                tools,
                prompt
            )
            agent_executor = AgentExecutor(agent=agent_chain, tools=tools, return_intermediate_steps=True, verbose=True, max_iterations=50,
                                           handle_parsing_errors='Check your output and make sure it either uses: Correct Action/Action Input syntax, or Outputs the Final Answer directly after the thought in the form of  "Thought: I now know the final answer\nFinal Answer: the final answer to the original input question"')

            # need to find the click growth of each keyword
            # read the whole_kw.csv
            # df_whole = pd.read_csv('/home/ubuntu/reflexion/New_LLM_Agent_4_Ad_Keyword_and_Text/preprocessing/data/whole_kw.csv')

            # Merge the two dataframes on 'Keyword' and 'Category'
            # merged_df = pd.merge(df, df_whole, on=['Keyword', 'Category'], suffixes=('_df1', '_df2'))

            # Calculate the difference in Clicks
            # merged_df['Clicks Difference'] = merged_df['Clicks_df2'] - merged_df['Clicks_df1']

            # Group by 'Category Status' and calculate the mean Clicks Difference
            # category_status_mean_difference = merged_df.groupby('Category Status')['Clicks Difference'].mean().reset_index()

            # Cast 'Category Status' colomn to type string

            # Filtering and summing clicks for 'wider' and 'deeper' categories


            # Filter the dataframe to get the click difference for 'deeper'
            # deeper_click_difference = category_status_mean_difference[category_status_mean_difference['Category Status'] == 'deeper']['Clicks Difference'].values[0]

            # Filter the dataframe to get the click difference for 'wider'
            # wider_click_difference = category_status_mean_difference[category_status_mean_difference['Category Status'] == 'wider']['Clicks Difference'].values[0]



            num_good_categories = int(self.config['KEYWORD']['NUM_GOOD_CATEGORIES'])
            num_keywords_per_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_CATEGORY'])
            num_new_categories = int(self.config['KEYWORD']['NUM_NEW_CATEGORIES'])

            # Adjust for rounding errors and maintain the sum
            current_sum = num_keywords_per_category + num_new_categories
            num_keywords_per_new_category = int(self.config['KEYWORD']['NUM_KEYWORDS_PER_NEW_CATEGORY'])

            self.good_kw_list = [s for s in self.good_kw_list if s not in df["Keyword"].tolist()]
            rejected_kw_list = list(set(rejected_kw_list))

            rejected_keywords_string = ", ".join(rejected_kw_list[-30:])  # Converts list to string
            good_kw_string = ", ".join(self.good_kw_list)  # Converts list to string

            # 4. Process the first prompt
            # Define the prompt with placeholders for the hyperparameters

            # Format the prompt with the hyperparameters
            # first_prompt = prompt_template.format(num_keywords_per_category, num_new_categories,
            #                                       num_keywords_per_new_category, rejected_keywords_string,
            #                                       good_kw_string, str(self.config["SETTING"]['product']),
            #                                       self.config["SETTING"]["language"])
            first_prompt = self.multi_shot_prompt_create(num_good_categories,
                    num_keywords_per_category, num_new_categories,
                                                         num_keywords_per_new_category,
                                                         rejected_keywords_string, good_kw_string, sorted_by_keyword_good,
                                                         sorted_by_keyword_bad, sorted_by_category_good, sorted_by_category_bad)
            # 5. Output the first qustion and Run the agent chain

            print("Question: " + first_prompt)

            action_int_dic, _, code_out = self.run_with_retries(agent_executor, first_prompt,
                                                                int(self.config['LLM']['max_attempts']))

            if code_out == 4:
                return [], {}, int(4), []
            # transfer the dic to list by dumping the key
            # new_words_list = list(action_int_dic.values())

            # Initialize an empty list to hold all values
            new_words_list = []
            # Iterate over the dictionary and extend the list with each value list
            for key in action_int_dic:
                new_words_list.extend(action_int_dic[key])

            # this should be replaced by the func. of Ascade san
            if str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"]) == 'True':
                # Create a list of all strings in 'Branded'
                all_branded = []
                for key in action_int_dic["Branded"]:
                    all_branded.extend(action_int_dic["Branded"][key])

                # Create a list of all strings in 'Non-Branded'
                all_non_branded = []
                for key in action_int_dic["Non-Branded"]:
                    all_non_branded.extend(action_int_dic["Non-Branded"][key])

                # this should be replaced by the func. of Ascade san
                if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        metrics = self.get_keyword_planner_metrics(
                            '1252903913',
                            all_non_branded,
                            '2023-06',
                            '2024-06',
                        )
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        metrics = self.get_keyword_planner_metrics(
                            '1252903913',
                            all_branded,
                            '2023-06',
                            '2024-06',
                        )
                    else:
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # Wait for 10 seconds
                    time.sleep(10)

                    # Normalize keyword casing for accurate matching
                    # keyword_data_normalized = {k['Keyword'].lower(): k['Avg. monthly searches'] for k in metrics}
                    # Create a dictionary for lookup with normalized keys
                    keyword_to_search_volume = {
                        self.normalize_keyword(metric['Keyword']): metric['Avg. monthly searches'] for metric in
                        metrics}

                    # Extract 'Avg. monthly searches' for keywords in the list
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        # new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]
                        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword
                                           in all_non_branded]

                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        # new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_branded]
                        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword
                                           in all_branded]
                    else:
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]

                elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        search_result = cb_exec_kw_plan(all_non_branded, 12)
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        search_result = cb_exec_kw_plan(all_branded, 12)
                    else:
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # search_result = cb_exec_kw_plan(all_non_branded, 12)
                    # Wait for 10 seconds
                    time.sleep(10)
                    # Normalize keyword casing for accurate matching
                    # keyword_data_normalized = {k['keyword'].lower(): k['avg_monthly_searches'] for k in search_result}
                    keyword_to_search_volume = {self.normalize_keyword(k['keyword']): k['avg_monthly_searches'] for k in
                                                search_result}

                    # Extract 'Avg. monthly searches' for keywords in the list
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        # new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]
                        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword
                                           in all_non_branded]

                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        # new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_branded]
                        new_words_check = [keyword_to_search_volume.get(self.normalize_keyword(keyword), 0) for keyword
                                           in all_branded]
                    else:
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]





            elif str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"]) == 'False':
                new_words_check = [60, 70, 80, 90, 100, 100, 100, 100, 100]

            else:
                raise ValueError(
                    "Invalid value for SEARCH_VOLUMN_CHECK: " + str(self.config["KEYWORD"]["SEARCH_VOLUMN_CHECK"]))

            while self.attempt < int(self.config['LLM']['max_attempts']):
                # if all the element in new_words_check is over 50, break the loop
                if all(x >= int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']) for x in new_words_check):
                    self.successfull_pass = True
                    # add the new generated keywords to the /data/initial_KW.csv
                    # 1. covert the dic to dataphrame
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        new_keywords_df = pd.DataFrame(
                            [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in
                             action_int_dic[branded].items() for kw in kws],
                            columns=['Category', 'Keyword', 'Branded']
                        )
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        new_keywords_df = pd.DataFrame(
                            [(k, kw, branded == "Branded") for branded in ["Branded", "Non-Branded"] for k, kws in
                             action_int_dic[branded].items() for kw in kws],
                            columns=['Category', 'Keyword', 'Branded']
                        )
                    else:
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))

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
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        df = df[df['Branded'] == False]
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        df = df[df['Branded'] == True]
                    else:
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # df = df[df['Branded'] == False]
                    df = df.drop_duplicates(subset='Keyword', keep='first')

                    if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                        with open(self.rejected_keyword_list, 'w') as file:
                            for item in list(set(rejected_kw_list)):
                                file.write("%s\n" % item)
                    elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                        s3_client = boto3.client('s3')
                        buffer = io.BytesIO()
                        # Write each keyword to the buffer as a new line
                        for item in rejected_kw_list:
                            buffer.write(f"{item}\n".encode("utf-8"))
                        # Move to the start of the StringIO buffer
                        buffer.seek(0)
                        # Upload the buffer content to S3
                        s3_client.put_object(Bucket=self.s3_bucket, Key=self.save_dir + self.rejected_keyword_list,
                                             Body=buffer.getvalue())

                    if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                        df.to_csv(self.generated_keyword_data, index=False)
                    else:
                        # full_path = self.save_dir + self.generated_keyword_data[2:]
                        buffer = io.BytesIO()
                        df.to_csv(buffer, index=False)
                        body = buffer.getvalue()
                        client = boto3.client('s3')
                        client.put_object(Bucket=self.s3_bucket, Key=self.save_dir + self.generated_keyword_data,
                                          Body=body)

                    break


                else:
                    for i in range(len(new_words_check)):
                        if new_words_check[i] < int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD']):
                            # add the low search check new words to the tried_kw_list
                            if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                                rejected_kw_list.append(str(all_non_branded[i]))
                                print("The new words whose search check is less than 5 is: " + str(all_non_branded[i]))
                            elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                                rejected_kw_list.append(str(all_branded[i]))
                                print("The new words whose search check is less than 5 is: " + str(all_branded[i]))
                            else:
                                raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(
                                    self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                            # rejected_kw_list.append(str (all_non_branded[i]))

                            # save the rejected_kw_list to a file
                            if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                                with open(self.rejected_keyword_list, 'w') as file:
                                    for item in rejected_kw_list:
                                        file.write("%s\n" % item)
                            elif str(self.config["EXE"]["S3_DEPLOY"]) == 'True':
                                s3_client = boto3.client('s3')
                                buffer = io.BytesIO()
                                # Write each keyword to the buffer as a new line
                                for item in rejected_kw_list:
                                    buffer.write(f"{item}\n".encode("utf-8"))
                                # Move to the start of the StringIO buffer
                                buffer.seek(0)
                                # Upload the buffer content to S3
                                s3_client.put_object(Bucket=self.s3_bucket,
                                                     Key=self.save_dir + self.rejected_keyword_list,
                                                     Body=buffer.getvalue())

                                # Clear the buffer after uploading to reuse it
                                buffer.seek(0)
                                buffer.truncate()

                        else:
                            # add keywords to the self.good_kw_list
                            if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                                self.good_kw_list.append(str(all_non_branded[i]))
                                self.good_kw_list = list(set(self.good_kw_list))
                            elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                                self.good_kw_list.append(str(all_branded[i]))
                                self.good_kw_list = list(set(self.good_kw_list))
                            else:
                                raise ValueError("Invalid value for BRANDED_KEYWORD: " + str(
                                    self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                            # self.good_kw_list.append(str (all_non_branded[i]))

                    self.attempt += 1

                    return self.run(cb_get_kw_metrics, cb_exec_kw_plan, current_keyword_list, computing=False)

                    # print("Next Round, the rejected low search keywords are: " + str(rejected_kw_list))
            # response = agent_chain ({"input":  first_prompt})

            if not self.successfull_pass:
                # Flatten the dictionary into a list of keywords without considering the category
                # keywords_flat_list = [kw for sublist in action_int_dic["Non-Branded"].values() for kw in sublist]
                # Flatten the dictionary into a list of tuples (category, keyword)
                if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                    keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Non-Branded"].items() for kw
                                                in kws]
                elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                    keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Branded"].items() for kw in
                                                kws]
                else:
                    raise ValueError(
                        "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                # keywords_with_categories = [(cat, kw) for cat, kws in action_int_dic["Non-Branded"].items() for kw in kws]

                # Filter keywords by matching them with their corresponding search amount
                # good_keywords = [kw for kw, check in zip(keywords_flat_list, new_words_check) if check > int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD'])]
                good_keywords = [(cat, kw) for (cat, kw), check in zip(keywords_with_categories, new_words_check) if
                                 check > int(self.config['KEYWORD']['SEARCH_CHECK_THRESHOLD'])]

                if good_keywords:
                    # Create DataFrame from good keywords
                    new_keywords_df = pd.DataFrame(good_keywords, columns=['Category', 'Keyword'])
                    if str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'False':
                        new_keywords_df['Branded'] = False
                    elif str(self.config["KEYWORD"]["BRANDED_KEYWORD"]) == 'True':
                        new_keywords_df['Branded'] = True
                    else:
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # new_keywords_df['Branded'] = "Non-Branded"  # Adding a constant column for branding

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
                        raise ValueError(
                            "Invalid value for BRANDED_KEYWORD: " + str(self.config["KEYWORD"]["BRANDED_KEYWORD"]))
                    # Remove rows with duplicate values in the 'Keyword' column, keeping the first occurrence
                    df = df.drop_duplicates(subset='Keyword', keep='first')

                    df['Clicks'] = df['Clicks'].fillna(0)
                    # df['Clicks'] = [random.randint(1, 100) for _ in df.index]  # Optional, simulate clicks

                if str(self.config["EXE"]["S3_DEPLOY"]) == 'False':
                    df.to_csv(self.generated_keyword_data, index=False)
                else:
                    # full_path = self.save_dir + self.generated_keyword_data[2:]
                    buffer = io.BytesIO()
                    df.to_csv(buffer, index=False)
                    body = buffer.getvalue()
                    client = boto3.client('s3')
                    client.put_object(Bucket=self.s3_bucket, Key=self.save_dir + self.generated_keyword_data, Body=body)

                self.good_kw_list = []


            print("Next Round, the rejected low search keywords are: " + str(rejected_kw_list))
            # response = agent_chain ({"input":  first_prompt})


        else:
            return [], {}, int(2), []
        # df = df[df['Branded'] == False]
        keywords_list = df["Keyword"].tolist()
        # Find the intersection of both lists
        common_items = set(keywords_list) & set(original_keywords)

        # Remove common items from both lists
        new_keyword_list = [item for item in keywords_list if item not in common_items]

        unique_keywords = list(set(new_keyword_list))
        category_list = df["Category"].tolist()
        return unique_keywords, category_list, code_out, invalid_keyword["Keyword"].tolist()

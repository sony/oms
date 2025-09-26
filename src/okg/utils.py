import pandas as pd
import datetime
import json
import time
import requests
import ast
from langchain_openai import AzureChatOpenAI
import pickle
import os
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import re
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pdb

def concatenate_llm_parts(config, section, setting_day, observation_period):
    """
    Concatenate all values in the specified section whose keys start with 'PART'.
    parameters:
    - config: the configuration file
    - section: the section in the configuration file
    returns:
    - concatenated_string: the concatenated string of all values in the section whose keys start with 'PART'
    """

    # Retrieve all items (key-value pairs) in the section
    items = config.items(section)
    # Sort items by key to ensure correct order
    sorted_items = sorted(items, key=lambda x: x[0])
    # replace the date named '3/1 and 3/2' with the corresponding value
    setting_day = pd.to_datetime (setting_day)
    observation_period = int(observation_period)
    date_str = ''
    for i in range(1, observation_period + 1):
        # generate the date string
        date_str += str((setting_day - pd.DateOffset(days=i)).strftime('%m/%d')) + ', '
    # delete the last ', '
    date_str = date_str[:-2]
    
    if str(config['EXE']['CPA_MODEL']) == 'False':
        # Concatenate values whose keys start with 'PART'
        if str(config['KEYWORD']['TYPE']) == 'MULTI':
            original_string = "".join(value for key, value in sorted_items if key.startswith('part')and key != 'part2')
        elif str(config['KEYWORD']['TYPE']) == 'SINGLE':
            original_string = "".join(value for key, value in sorted_items if key.startswith('part')and key != 'part3')
        else:
            raise ValueError("The value of 'TYPE' in the 'KEYWORD' section must be either 'MULTI' or 'SINGLE'.")
    else:
        original_string = " ".join(value for key, value in sorted_items if key.startswith('cpa_part'))
    
    
    # replace the date named '3/1 and 3/2' win concatenated_string
    concatenated_string = original_string.replace('3/1 and 3/2', date_str)

    ## replace the date named '3/3' with target date
    concatenated_string = concatenated_string.replace('3/3', str( pd.to_datetime (setting_day).strftime('%m/%d')))

    concatenated_string = concatenated_string.replace('Neural Network Console of Sony', str(config['CAMPAIGN']['PRODUCT_NAME']))

    return concatenated_string, original_string

def concatenate_reflection_beginning(config, section, setting_day, observation_period, setting_times):
    """
    Concatenate all values in the specified section whose keys start with 'PART'.
    parameters:
    - config: the configuration file
    - section: the section in the configuration file
    returns:
    - concatenated_string: the concatenated string of all values in the section whose keys start with 'PART'
    """

    # Retrieve all items (key-value pairs) in the section
    items = config.items(section)
    # Sort items by key to ensure correct order
    sorted_items = sorted(items, key=lambda x: x[0])
    # replace the date named '3/1 and 3/2' with the corresponding value
    setting_day = pd.to_datetime (setting_day)
    observation_period = int(observation_period)
    date_str = ''
    for i in range(1, observation_period + 1):
        # generate the date string
        date_str += str((setting_day - pd.DateOffset(days=i)).strftime('%m/%d')) + ', '
    # delete the last ', '
    date_str = date_str[:-2]
    
    # Concatenate values whose keys start with 'PART'
    original_string = "".join(value for key, value in sorted_items if key.startswith('part'))
    
    # replace the date named '3/1 and 3/2' win concatenated_string
    concatenated_string = original_string.replace('3/1 and 3/2', date_str)

    ## replace the date named '3/3' with target date
    concatenated_string = original_string.replace('3/3', str( pd.to_datetime (setting_day).strftime('%m/%d')))

    return concatenated_string

def parse_output(output):
    """
    Parse the output from the agent chain into a list of integers.
    Parameters:
    - output: the output from the agent chain

    Returns:
    - action_int_list: the list of integers parsed from the output
    """
    # Attempt to parse the output into a list of integers
    return list(map(int, output.split(', ')))


# def parse_dic_output(output):
#     """
#     Parse the output from the agent chain into a list of integers.
#     Parameters:
#     - output: the output from the agent chain

#     Returns:
#     - action_int_list: the list of integers parsed from the output
#     """
#     # Attempt to parse the output into a list of integers
#     # Replace single quotes with double quotes to make it valid JSON
#     output = output.replace("````", "").replace("```", "").replace("`", "").replace("json", "").strip()
#     output = output.replace('\'', '\"')
#     # delete /n if exists
#     output = output.replace('\n', '')
#     # Convert the string to a dictionary
#     output = json.loads(output)
    
#     if type(output) != dict:
#         raise ValueError("The output is not a dictionary.")
#     return output

def parse_dic_output(output):
    """
    Parse the output from the agent chain into a dictionary.
    """
    # First, try to find JSON in code blocks
    pattern = r"```(?:json)?(.*?)```"
    match = re.search(pattern, output, re.DOTALL)
    print("received output dictionary for parsing is ", output)
    if match:
        # Found content in code blocks
        json_content = match.group(1).strip()
    else:
        # If no code blocks, try to extract the entire output
        json_content = output.strip()
    
    # Clean up the content
    json_content = json_content.replace('\'', '"')
    json_content = re.sub(r'\n\s*', ' ', json_content)  # Handle multi-line formatting
    
    # Try to parse with json
    try:
        result = json.loads(json_content)
        return result
    except json.JSONDecodeError:
        # If standard JSON parsing fails, try with ast.literal_eval
        try:
            import ast
            result = ast.literal_eval(json_content.replace('"', "'"))
            if isinstance(result, dict):
                return result
        except:
            pass
        
        # If still failing, try to extract only the dictionary part
        dict_pattern = r'\{.*\}'
        match = re.search(dict_pattern, output, re.DOTALL)
        if match:
            try:
                dict_content = match.group(0)
                return json.loads(dict_content.replace('\'', '"'))
            except:
                pass
                
    # If all attempts fail, raise an error
    raise ValueError("Could not parse output as a dictionary.")


def parse_list_output_old(output):
    """
    Parse the output from the agent chain into a list of integers.
    Parameters:
    - output: the output from the agent chain

    Returns:
    - action_int_list: the list of integers parsed from the output
    """
    # Attempt to parse the output into a list of integers
    # Replace single quotes with double quotes to make it valid JSON
    output = output.replace("```", "").replace("`", "").replace("json", "").replace("python", "").strip()
    #output = output.replace(''', '')
    output = output.replace('\'', '\"')
    # delete /n if exists
    output = output.replace('\n', '')
    # Convert the string to a dictionary
    output = ast.literal_eval(output)
    
    if not isinstance(output, list):
        raise TypeError("Expected output to be a list, but got a different type.")
    return output

def parse_list_output(output):
    """
    Robust parser for extracting and processing a 2D list from agent output.
    Handles various formatting issues and ensures the final structure is a 2D list.

    Parameters:
    - output: a string representing the list data, potentially with extra text
    
    Returns:
    - A 2D list of strings (or raises an error if extraction/parsing fails)
    """
    # First try to extract just the list part from the output
    # Common patterns include "Final Answer: [...", "Here is the final output: [...", etc.
    list_pattern = r'(?:\[\s*\[|\[\n\s*\[)'  # Match the beginning of a 2D list structure
    match = re.search(list_pattern, output)
    
    if match:
        # Extract from the start of the list pattern to the end
        start_idx = match.start()
        extracted_text = output[start_idx:]
        
        # Try to find the closing brackets of the outermost list
        bracket_count = 0
        end_idx = 0
        
        for i, char in enumerate(extracted_text):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > 0:
            extracted_text = extracted_text[:end_idx]
        
        # Clean up the extracted text
        output = extracted_text
    
    # Clean up the formatting issues
    output = output.replace("```", "").replace("`", "").replace("json", "").replace("python", "").strip()
    
    # Make sure commas are present between list items (common error in LLM outputs)
    output = re.sub(r'"\s*\n\s*"', '",\n"', output)  # Fix missing commas between strings
    output = re.sub(r'"\s*\]\s*\[', '"],\n[', output)  # Fix missing commas between sublists
    
    # Replace single quotes with double quotes for JSON compatibility
    output = output.replace('\'', '"')
    
    # Try multiple parsing strategies
    parsed_output = None
    
    # Strategy 1: Direct literal_eval
    try:
        parsed_output = ast.literal_eval(output)
    except Exception:
        pass
    
    # Strategy 2: Try JSON parsing
    if parsed_output is None:
        try:
            parsed_output = json.loads(output)
        except Exception:
            pass
    
    # Strategy 3: More aggressive cleaning and then try literal_eval
    if parsed_output is None:
        # Remove any text before the first '[' and after the last ']'
        cleaned = re.sub(r'^[^\[]*', '', output)
        cleaned = re.sub(r'[^\]]*$', '', cleaned)
        
        try:
            parsed_output = ast.literal_eval(cleaned)
        except Exception:
            pass
    
    # Strategy 4: Extract lists line by line as a last resort
    if parsed_output is None:
        try:
            # Find all quoted strings and reconstruct the lists
            items = re.findall(r'"([^"]*)"', output)
            
            # Determine the structure by looking at the original text
            splits = []
            current_group = []
            
            for item in items:
                current_group.append(item)
                # Look for patterns that indicate a group break
                if len(current_group) > 0 and (
                    re.search(r"\],\s*\[", output) is not None or
                    re.search(r"\]\s*\[", output) is not None
                ):
                    # Heuristic: if we see more than 10 items, assume first group is done
                    if len(current_group) >= 10:
                        splits.append(current_group)
                        current_group = []
            
            # Add any remaining items
            if current_group:
                splits.append(current_group)
            
            if splits:
                parsed_output = splits
        except Exception:
            pass
    
    # If all parsing strategies failed, raise an error
    if parsed_output is None:
        raise ValueError("Failed to parse the output after multiple attempts. Check the format.")
    
    # Validate that parsed_output is a list
    if not isinstance(parsed_output, list):
        raise TypeError(f"Expected a list at the top level, but got {type(parsed_output).__name__}.")
    
    # Validate that it is a 2D list
    for i, sublist in enumerate(parsed_output):
        if not isinstance(sublist, list):
            # Try to convert single items to lists if needed
            if isinstance(sublist, (str, int, float)):
                parsed_output[i] = [sublist]
            else:
                raise TypeError(f"Expected element {i} to be a list, but got {type(sublist).__name__}.")
        
        # Ensure no further nesting beyond 2D
        for j, item in enumerate(sublist):
            if isinstance(item, list):
                # Flatten if we find a nested list
                flattened = []
                for nested_item in item:
                    if isinstance(nested_item, (str, int, float)):
                        flattened.append(nested_item)
                    elif isinstance(nested_item, list):
                        flattened.extend([x for x in nested_item if not isinstance(x, list)])
                parsed_output[i][j] = flattened[0] if len(flattened) == 1 else flattened
    
    return parsed_output

def run_with_retries_old(agent_chain, input_prompt, max_attempts=20):
    """
    Run the agent chain with retries until the output can be successfully parsed.
    Parameters:
    - agent_chain: the agent chain function
    - input_prompt: the input prompt for the agent chain
    - max_attempts: the maximum number of attempts to run the agent chain
    Returns:
    - action_int_list: the list of integers parsed from the output
    """
    
    attempts = 0
    scratch_pad = ''
    while attempts < max_attempts:
        try:
            # Call the function and attempt to parse the output
            response = agent_chain.invoke({"input" : input_prompt})
            #action_int_list = parse_output(response["output"])            
            action_int_list = parse_dic_output(response["output"])


            # ac_keys = list(action_int_list.keys())
            # output_dict = {}
            # for ac_key in ac_keys:
            #     for ac_key_key in action_int_list[ac_key].keys():
            #         if ac_key_key in output_dict:
            #             output_dict[ac_key_key].extend(action_int_list[ac_key][ac_key_key])
            #         else:
            #             output_dict[ac_key_key] = action_int_list[ac_key][ac_key_key]
            
            
            # Generate and print the scratchpad content
            scratch_pad = extract_scratchpad_as_string(response)

            return action_int_list, scratch_pad  # Return on successful parsing
        except ValueError as e:
            # Handle the specific parsing error
            print(f"Attempt {attempts + 1} failed: {e}. Retrying...")
            attempts += 1
            # Optional: add a delay here if needed
            time.sleep(1)  # Sleep for 1 second between retries
    raise Exception("Maximum retries reached, all attempts failed.")

# Normalize function (removes spaces for consistent matching)
def normalize_keyword(keyword):
    # Remove extra spaces and convert to lowercase
    return keyword.strip().lower().replace(' ', '')

def run_with_retries(agent_chain, input_prompt, max_attempts=20):
    """
    Run the agent chain with retries until the output can be successfully parsed or a request is successful.
    Parameters:
    - agent_chain: the agent chain function
    - input_prompt: the input prompt for the agent chain
    - max_attempts: the maximum number of attempts to run the agent chain
    Returns:
    - action_int_list: the list of integers parsed from the output or an empty dictionary if all attempts fail
    - scratch_pad: the scratch pad content from the last attempt or an empty string if all attempts fail
    """
    
    attempts = 0
    scratch_pad = ''
    while attempts < max_attempts:
        try:
            # Call the function and attempt to parse the output
            response = agent_chain.invoke({"input" : input_prompt})
            action_int_list = parse_dic_output(response["output"])
            scratch_pad = extract_scratchpad_as_string(response)
            return action_int_list, scratch_pad, int(1)  # Return on successful parsing
        except (ValueError, KeyError) as e:
            # Handle the specific parsing error
            print(f"Attempt {attempts + 1} failed due to parsing error: {e}. Retrying...")
        except (requests.exceptions.RequestException, IOError) as e:
            # Handle network errors specifically
            print(f"Attempt {attempts + 1} failed due to network error: {e}. Retrying...")
        except Exception as e:
            # Generic exception handler for any other unexpected issues
            print(f"Attempt {attempts + 1} failed due to unexpected error: {e}. Retrying...")
        
        attempts += 1
        time.sleep(1)  # Sleep for 1 second between retries

    print("Maximum retries reached, returning empty results.")
    return {}, scratch_pad, int(4)  # Return empty results if all attempts fail

def invoke_with_retries(agent_executor, input_text, max_attempts=20):
    """
    Invoke the agent with retries until the output can be successfully retrieved as a string.
    Parameters:
    - agent_executor: the agent executor function or object
    - input_text: the input text for the agent
    - max_attempts: the maximum number of attempts to invoke the agent
    Returns:
    - output_str: the output string from the response or an empty string if all attempts fail
    """
    
    attempts = 0
    while attempts < max_attempts:
        try:
            response = agent_executor.invoke({"input": input_text})
            output_str = response["output"]
            output_list = parse_list_output(output_str)
            return output_list , int(1) # Return on successful retrieval
        except (requests.exceptions.RequestException, IOError) as e:
            # Handle network errors specifically
            print(f"Attempt {attempts + 1} failed due to network error: {e}. Retrying...")
        except Exception as e:
            # Generic exception handler for any other unexpected issues
            print(f"Attempt {attempts + 1} failed due to unexpected error: {e}. Retrying...")
        
        attempts += 1
        time.sleep(1)  # Sleep for 1 second between retries

    print("Maximum retries reached, returning empty result.")
    return "" , int(4) # Return empty string if all attempts fail

def post_process_ads(ad_data):
    # Process headlines
    headlines = [headline.replace('!', '.').replace('！', '.') for headline in ad_data[0]]
    # remove \ in the headlines
    headlines = [headline.replace('\\', '') for headline in headlines]
    
    # Process descriptions
    descriptions = [description.replace('??', '?').replace('？？', '?') for description in ad_data[1]]
    # remove \ in the descriptions
    descriptions = [description.replace('\\', '') for description in descriptions]
    
    # Check the number of the descriptions is smlaler than 5, if so just keep the first 4
    if len(descriptions) < 5:
        descriptions = descriptions
    else:
        descriptions = descriptions[:4]
    
    return [headlines, descriptions]

# Function to extract thoughts, actions, observations and return them as a single string
def extract_scratchpad_as_string(response):
    scratchpad_string = f"Input: {response['input']}\n"

    steps = response['intermediate_steps']
    for i, step in enumerate(steps):
        action, observation = step
        scratchpad_string += f"Thought {i+1}: {action.log}\n"
        scratchpad_string += f"Action {i+1}: Tool - {action.tool}, Tool Input - {action.tool_input}\n"
        scratchpad_string += f"Observation {i+1}: {observation}\n\n"

    scratchpad_string += f"Output: {response['output']}"
    
    
    return scratchpad_string

# Function to extract thoughts, actions, observations and return them as a single string with fewer line breaks
def extract_scratchpad_as_compact_string(response):
    scratchpad_string = f"Input: {response['input']}\n"

    steps = response['intermediate_steps']
    for i, step in enumerate(steps):
        action, observation = step
        scratchpad_string += f"\n• Thought {i+1}: {action.log}"
        scratchpad_string += f"\n• Action {i+1}: Tool - {action.tool}, Tool Input - {action.tool_input}"
        scratchpad_string += f"\n• Observation {i+1}: {observation}"

    scratchpad_string += f"\n\nOutput: {response['output']}"
    
    return scratchpad_string


def calculate_ad_settings(setting_daym, observation_period):
    

    # Calculate the day of the month when the ad was first set
    today_day = setting_daym.day
    
    # Calculate how many days have passed since the ad was first set
    days_passed = today_day - 1
    
    # Calculate the number of times the ad has been set
    count = days_passed // observation_period

    
    return max (0, count -1)


def compare_ad_metrics(config, df_current_day, df_day_after):
    
    if str(config['EXE']['CPA_MODEL']) == 'False':
        # Define the columns to compare
        columns_to_compare = [
            'ad1_cost', 'ad1_clicks', 'ad1_Real_CPC',
            'ad2_cost', 'ad2_clicks', 'ad2_Real_CPC',
            'ad3_cost', 'ad3_clicks', 'ad3_Real_CPC',
            'total_clicks', 'total_cost', 'total_CPC'
        ]
    else:
        # Define the columns to compare
        columns_to_compare = [
            'ad1_cost', 'ad1_conversions', 'ad1_Real_CPA',
            'ad2_cost', 'ad2_conversions', 'ad2_Real_CPA',
            'ad3_cost', 'ad3_conversions', 'ad3_Real_CPA',
            'total_conversions', 'total_cost', 'total_CPA'
        ]
    
    # Initialize an empty list to store the description of changes
    changes_description = []

    # Loop through each column and compare the values
    for column in columns_to_compare:
        # Get the values from both dataframes
        value_day_1 = df_current_day[column].iloc[0]
        value_day_2 = df_day_after[column].iloc[0]
        
        # Determine the change description
        if value_day_2 > value_day_1:
            change = 'increased'
        elif value_day_2 < value_day_1:
            change = 'decreased'
        else:
            change = 'remained the same'

        # Append the description to the list
        changes_description.append(f"{column} {change} from {value_day_1} to {value_day_2}")

    # Join all descriptions into a single string
    return '. '.join(changes_description) + '.'

    

def compare_ad_settings(config, df_current_day, df_day_after):
    
    if str(config['EXE']['CPA_MODEL']) == 'False':
        # Define the columns to compare related to budget and CPC settings
        columns_to_compare = [
            'ad1_budget', 'ad1_Max_CPC',
            'ad2_budget', 'ad2_Max_CPC',
            'ad3_budget', 'ad3_Max_CPC'
        ]
    else:
        # Define the columns to compare related to budget and CPC settings
        columns_to_compare = [
            'ad1_budget', 'ad1_target_CPA',
            'ad2_budget', 'ad2_target_CPA',
            'ad3_budget', 'ad3_target_CPA'
        ]

    # Initialize an empty list to store the changes as integers
    changes = []

    # Loop through each column to compare the values
    for column in columns_to_compare:
        # Get the values from both dataframes
        value_day_1 = df_current_day[column].iloc[0]
        value_day_2 = df_day_after[column].iloc[0]
        
        # Determine the change and append the corresponding integer to the list
        if value_day_2 > value_day_1:
            changes.append(1)  # Indicates an increase
        elif value_day_2 < value_day_1:
            changes.append(-1)  # Indicates a decrease
        else:
            changes.append(0)  # Indicates no change

    return changes


def compute_cosine_similarity_batchwise(embeddings, batch_size=512):
    num_embeddings = embeddings.size(0)
    similarity_matrix = np.zeros((num_embeddings, num_embeddings), dtype=np.float32)

    # Compute pairwise cosine similarity in batches
    
    for i in tqdm(range(0, num_embeddings, batch_size)):
        end_i = min(i + batch_size, num_embeddings)

        for j in range(0, num_embeddings, batch_size):
            end_j = min(j + batch_size, num_embeddings)

            # Calculate cosine similarity for each batch
            batch_similarity = torch.nn.functional.cosine_similarity(
                embeddings[i:end_i].unsqueeze(1),
                embeddings[j:end_j].unsqueeze(0),
                dim=2
            ).cpu().numpy()

            # Fill the respective positions in the total similarity matrix
            similarity_matrix[i:end_i, j:end_j] = batch_similarity

    return similarity_matrix


def get_embedding(keywords):
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')#.cuda()
    print("Model loaded successfully.")
    
    embeddings = []
    for keyword in keywords:
        inputs = tokenizer(keyword, return_tensors='pt', padding=True, truncation=True, max_length=50)#.to('cuda')
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].detach()
        embeddings.append(embedding)
    return torch.cat(embeddings)


def create_clustering(df, config, base_path, num_shot=200):
    product_name = str(config['SETTING']['product'])

    # Determine if 'Keyword' or 'キーワード' is used
    keyword_col = 'Keyword' if 'Keyword' in df.columns else 'キーワード'
    topsis_indicator = True if "Topsis Score" in df.columns else False
    # Get list of keywords
    keywords = df[keyword_col].tolist()

    # Calculate embeddings

    # if f"{product_name}_{num_shot}_clustering_info.pkl" in os.listdir(f"{base_path}/src/results/"):
    #     with open(f'{base_path}/src/results/{product_name}_{num_shot}_clustering_info.pkl', 'rb') as f:
    #         clustering_info = pickle.load(f)
    #     return clustering_info['best_clustering'], clustering_info['cluster_centers_keywords'], clustering_info[
    #         'cluster_centers_indices'], clustering_info['cluster_stats'], clustering_info['embeddings'], \
    #     clustering_info['category_list']
    # if torch.cuda.is_available():
    #     embeddings = get_embedding(keywords).cuda()
    # else:
    embeddings = get_embedding(keywords).cpu()
    # Calculate the similarity matrix
    similarity_matrix = compute_cosine_similarity_batchwise(embeddings)

    # Ensure the similarity matrix is a 2D array
    assert similarity_matrix.ndim == 2, "The similarity matrix must be a 2-dimensional array."
    llm = AzureChatOpenAI(
        deployment_name=str(config['LLM']['gpt4o_deployment_name']),
        openai_api_version=str(config['LLM']['gpt4o_openai_api_version']),
        openai_api_key=str(config['KEYS']['OPENAI_GPT4O_API_KEY']),
        azure_endpoint=str(config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']),
        temperature=float(config['LLM']['temperature']))
    # Perform clustering
    best_clustering = AffinityPropagation(affinity='precomputed').fit(similarity_matrix)
    labels = best_clustering.labels_
    cluster_centers_indices = best_clustering.cluster_centers_indices_
    cluster_centers_keywords = [keywords[i] for i in cluster_centers_indices]
    cluster_stats = []
    for center_idx in cluster_centers_indices:
        center_cluster_data = df[labels == labels[center_idx]]
        center_cluster_data.loc[:, 'Avg. CPC'] = pd.to_numeric(center_cluster_data['Avg. CPC'], errors='coerce')
        center_cluster_data.loc[:, 'Impr.'] = pd.to_numeric(center_cluster_data['Impr.'], errors='coerce')
        center_cluster_data.loc[:, 'Clicks'] = pd.to_numeric(center_cluster_data['Clicks'], errors='coerce')
        if topsis_indicator == True:
            center_cluster_data.loc[:, "Topsis Score"] = pd.to_numeric(center_cluster_data["Topsis Score"], errors='coerce')
        cluster_keywords = center_cluster_data[keyword_col].tolist()
        prompt = (f"You are given a product name {product_name} and the users searched for following keywords that "
                  f"leads to click the website related to this product '{product_name}', you are required to consider why users search for those keywords to click this product, those keywords are: "
                  f"{', '.join(cluster_keywords)}, provide a concise description of the keyword cluster.")
        response = llm.invoke(prompt)
        description = response.content.strip()
        # description = ""


        # Calculate average statistics
        if topsis_indicator == True: 
            min_index = center_cluster_data['Topsis Score'].idxmin()
            max_index = center_cluster_data['Topsis Score'].idxmax()
            min_keyword = center_cluster_data.loc[min_index, keyword_col]
            min_traffic = center_cluster_data.loc[min_index, 'Topsis Score']
            max_keyword = center_cluster_data.loc[max_index, keyword_col]
            max_traffic = center_cluster_data.loc[max_index, 'Topsis Score']
            avg_stats = {
                'Avg. CPC': center_cluster_data['Avg. CPC'].mean(),
                'Impr.': center_cluster_data['Impr.'].mean(),
                'Clicks': center_cluster_data['Clicks'].mean(),
                "Topsis Score": center_cluster_data['Topsis Score'].mean(),
                "description": description,
                "keyword_with_min_estimated_traffic": (min_keyword, min_traffic),
                "keyword_with_max_estimated_traffic": (max_keyword, max_traffic)
            }
        else:
            avg_stats = {
                'Avg. CPC': center_cluster_data['Avg. CPC'].mean(),
                'Impr.': center_cluster_data['Impr.'].mean(),
                'Clicks': center_cluster_data['Clicks'].mean(),
                "description": description,
            } 
        cluster_stats.append(avg_stats)
    category_list = category_initialization(df, cluster_centers_indices, llm, product_name, labels)
    # save all the information in a dictionary
    clustering_info = {
        "best_clustering": best_clustering,
        "cluster_centers_keywords": cluster_centers_keywords,
        "cluster_centers_indices": cluster_centers_indices,
        "cluster_stats": cluster_stats,
        "embeddings": embeddings,
        "category_list": category_list
    }
    # dump the cluster_info
    # with open(f"{base_path}/src/results/{product_name}_clustering_info.pkl", 'wb') as f:
    #      pickle.dump(clustering_info, f)

    return best_clustering, cluster_centers_keywords, cluster_centers_indices, cluster_stats, embeddings, category_list


def parse_categories(text):
    # Find the starting point of the summarized categories section
    start_keyword = "Summarized categories:"
    start_index = text.find(start_keyword)

    if start_index == -1:
        return "Categories section not found."

    # Extract the section containing the categories
    categories_text = text[start_index + len(start_keyword):].strip()

    # Remove the leading and trailing brackets
    categories_text = categories_text.lstrip('[').rstrip(']')

    # Split the text into individual categories
    categories = [category.strip() for category in categories_text.split(',')]

    return categories


def category_initialization(df, cluster_center_indices, llm, product_name, labels, num_of_categories=3):
    prompt = (
        f"product name {product_name} and the users searched for following keywords to reach the website related to this product '{product_name}'."
        f"You are given the following clusters of keywords and their clicks: \n")
    for cluster_label in cluster_center_indices:
        prompt += f"Cluster {cluster_label}:\n "
        cluster_data = df[labels == labels[cluster_label]]
        cluster_keywords = cluster_data['Keyword'].tolist()
        for keyword in cluster_keywords:
            prompt += f"{keyword} has clicks {cluster_data[cluster_data['Keyword'] == keyword]['Clicks'].values[0]}  \n"
    prompt += (
        f"Analyze above cluster and use {num_of_categories} categories to summarize the search behavior of users like General Feature, Specification or etc.\n"
        "The results should be a list of those categories. e.g [Category 1, Category 2, Category 3]\n"
        "Please provide the list of those summarized categories in the last line of your response to make it easy to parse."
        "A output format example is, strictly follow this format: \n"
        "Summarized categories: [Category 1, Category 2, Category 3]")
    response = llm(prompt)
    category = parse_categories(response.content.strip())
    return category




def predict_cluster(config, new_keyword, clustering, cluster_centers_indices, embeddings, dfs, new_cluster_counter):
    # Compute the embedding for the new keyword
    new_embedding = get_embedding([new_keyword]).cuda()
    cluster_center_embeddings = embeddings[cluster_centers_indices]

    # Calculate cosine similarities between the new keyword and cluster centers
    similarities = cosine_similarity(new_embedding.cpu().numpy(), cluster_center_embeddings.cpu().numpy())
    closest_cluster = np.argmax(similarities, axis=1)
    max_similarities = np.max(similarities, axis=1)

    # Calculate the mean and standard deviation of the similarities
    mean_similarity = np.mean(max_similarities)
    std_dev_similarity = np.std(max_similarities)
    # Determine a dynamic threshold based on statistical properties
    # For example, declare a keyword as an outlier if its similarity is less than mean - 2 * std_dev
    dynamic_threshold = mean_similarity - 2 * std_dev_similarity

    # Initialize cluster stats
    llm = AzureChatOpenAI(
        deployment_name=str(config['LLM']['deployment_name']),
        openai_api_version=str(config['LLM']['openai_api_version']),
        openai_api_key=str(config['KEYS']['OPENAI_GPT4O_API_KEY']),
        azure_endpoint=str(config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']),
        temperature=float(config['LLM']['temperature']))
    cluster_stats = {}
    for idx, keyword, max_similarity in zip(closest_cluster, new_keyword, max_similarities):
        # Retrieve words from the closest cluster
        words_in_cluster = dfs[clustering.labels_ == idx]['キーワード'].tolist()

        # Construct prompt for LLM
        prompt = (
            f"Given the new keyword '{keyword}' and the following existing cluster keywords: "
            f"{', '.join(words_in_cluster)}, determine if the new keyword belongs in this cluster. "
            "Respond with 'yes' if it does, or 'no' if it should be in a new cluster."
        )

        try:
            response = llm(prompt)
            decision = response.content.strip().lower()

            if decision == 'yes':
                cluster_stats[keyword] = {
                    "cluster_keyword_list": words_in_cluster,
                    "click_list": dfs[clustering.labels_ == idx]['Topsis Score'].tolist(),
                    "idx": str(idx)
                }
            else:
                # If LLM decides it should be a new cluster
                cluster_stats[keyword] = {
                    "cluster_keyword_list": [],
                    "click_list": [],
                    "idx": f"new cluster {new_cluster_counter}"
                }
                new_cluster_counter += 1

        except Exception as e:
            print(f"Error determining cluster for keyword '{keyword}': {e}")
            # Fallback in case of LLM error
            cluster_stats[keyword] = {
                "cluster_keyword_list": words_in_cluster,
                "click_list": dfs[clustering.labels_ == idx]['Topsis Score'].tolist(),
                "idx": str(idx)
            }
    return cluster_stats, new_cluster_counter


def merge_cluster_stats(cluster_stats_list):
    merged_clusters = defaultdict(lambda: {
        "keyword_list": [],
        "cluster_keyword_list": [],
        "click_list": [],
    })
    merged_list = []
    # Iterate over each cluster_stats dictionary
    for cluster_stats in cluster_stats_list:
        for keyword, stats in cluster_stats.items():
            idx = stats["idx"]
            # Append the keywords and click lists to the cluster identified by idx
            merged_clusters[idx]["keyword_list"].append(keyword)
            merged_clusters[idx]["cluster_keyword_list"] = stats["cluster_keyword_list"]
            merged_clusters[idx]["click_list"] = stats["click_list"]

    # Create a list of dictionaries with concatenated keywords for each cluster
    for idx, details in merged_clusters.items():
        concatenated_keywords = ", ".join(details["keyword_list"])
        temp_cluster_stats = {}
        temp_cluster_stats[concatenated_keywords] = {
            "cluster_keyword_list": details["cluster_keyword_list"],
            "click_list": details["click_list"],
            "idx": idx
        }
        merged_list.append(
            temp_cluster_stats
        )
    return merged_list

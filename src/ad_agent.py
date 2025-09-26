import configparser

#from llm_agent import ReactAgent
#from okg.okg_agent import okg_agent

import importlib.util
import os
import pandas as pd


class Ad_agent:
    def __init__(self, s3_bucket, s3_path, config_path, setting_day, code_in):

        self.config = configparser.ConfigParser()
        try:
            self.config.read(config_path)
        except Exception as e:
            raise ValueError("Failed to read the configuration file: " + str(e))
        # Standalone version: always use local base path
        base_path = '.'
        # Dynamic import based on calculated base path
        #self.okg_agent = self.dynamic_import(f"{base_path}/okg/okg_agent", "okg_agent")
        self.okg_agent = self.dynamic_import(f"{base_path}/okg/multi_shot_agent", "multi_shot_agent")
        self.post_process_ads = self.dynamic_import(f"{base_path}/okg/utils", "post_process_ads")
        
        self.keyword_agent = self.okg_agent(s3_bucket, s3_path, config_path, setting_day, code_in)
        
        self.setting_day = pd.to_datetime(setting_day)
        
        self.code = code_in
        self.running_week_day = tuple(int(day.strip()) for day in self.config['EXE']['RUNNING_WEEK_DAY'].split(','))
        
    def dynamic_import(self, module_name, function_name):
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, function_name)

    def run(self,cb_get_kw_metrics, cb_exec_kw_plan, cb_get_ad_metrics,current_keyword_list ):
        os.environ["LANGCHAIN_API_KEY"] = str(self.config["LANG_SMITH"]["LANGSMITH_API_KEY"])
        os.environ["LANGCHAIN_TRACING_V2"] = str(self.config["LANG_SMITH"]["LANGSMITH_TRACING"])
        os.environ["LANGCHAIN_PROJECT"] = str(self.config["LANG_SMITH"]["LANGSMITH_PROJECT"])
        if self.code == 0 and (self.setting_day.weekday() not in self.running_week_day): 
            return [], {}, int (2), []
        
        keyword_list, _, code_out,stopped_kw_list = self.keyword_agent.run(cb_get_kw_metrics,cb_exec_kw_plan,current_keyword_list )
        
        # if rhe keyword_list is empty, return the emmty list and dictionary
        # the case of API problem of OKG
        if code_out == 4:
            return [], {}, int(4), []
        
        # Create the new format
        #keywords_to_add = [{'keyword': keyword, 'match_type': match_type} 
                            #for keyword in keyword_list 
                            #for match_type in ['PHRASE', 'EXACT']]
        keywords_to_add = [{'keyword': keyword, 'match_type': 'PHRASE'} for keyword in keyword_list]
        
        text_to_add = {}
        return keywords_to_add, text_to_add, code_out, stopped_kw_list

import configparser
import pandas as pd
import time

from ad_agent import Ad_agent

# Note: Google Ads API removed for GitHub version
def get_keyword_planner_metrics(account_id, keywords, start_month, end_month):
    # Return a mocked metric list for offline usage
    return [{
        'Keyword': kw,
        'Avg. monthly searches': 100
    } for kw in keywords]



# only for verification
# delete this for the final version
def cb_get_kw_metrics(days):
    # Local-only fallback; reads from configured CSVs
    if setting_day.day <= 7:
        df = pd.read_csv(config_0["SETTING"]['generated_keyword_data'])
    else:
        df = pd.read_csv(config_0["SETTING"]["initial_keyword_data"])
    # keep only necessary columns
    cols = {c.lower(): c for c in df.columns}
    keyword_col = cols.get('keyword', 'Keyword') if 'Keyword' in df.columns or 'keyword' in cols else df.columns[0]
    clicks_col = cols.get('clicks', 'Clicks') if 'Clicks' in df.columns or 'clicks' in cols else df.columns[-1]
    kw_df = df[[keyword_col, clicks_col]].rename(columns={keyword_col: 'keyword', clicks_col: 'clicks'})

    return kw_df


# only for verification
# delete this for the final version
def cb_exec_kw_plan(kw_list, days):
    metrics = get_keyword_planner_metrics(
        str(config_0["KEYWORD_PLANNER"]["GOOGLE_ACCOUNT_ID"]),
        kw_list,
        str(config_0["KEYWORD_PLANNER"]["START_MONTH"]),
        str(config_0["KEYWORD_PLANNER"]["END_MONTH"]),
    )
    # Wait for 10 seconds
    time.sleep(10)

    # print('=== Keyword Planner metrics:')
    # for i, m in enumerate(metrics):
    # print(f'[{i}]:')
    # print(f'  Keyword: "{m["Keyword"]}"')
    # print(f'  Avg. monthly searches: {m["Avg. monthly searches"]:,}')
    # print(f'  Three month change: {m["Three month change"]}%')
    # print(f'  YoY change: {m["YoY change"]}%')
    # print(f'  Competition: {m["Competition"]}')
    # print(f'  Top of page bid (low range): ¥{m["Top of page bid (low range)"]}')
    # print(f'  Top of page bid (high range): ¥{m["Top of page bid (high range)"]}')
    # Normalize keyword casing for accurate matching
    keyword_data_normalized = {k['Keyword'].lower(): k['Avg. monthly searches'] for k in metrics}

    # Extract 'Avg. monthly searches' for keywords in the list
    new_words_check = [keyword_data_normalized.get(keyword.lower(), 0) for keyword in all_non_branded]

    return new_words_check


# only for verification
# delete this for the final version
def cb_get_ad_metrics(days):
    # not used in this version
    return True


def main():
    ## 0. Read the configuration file
    global config_0, setting_day
    config_0 = configparser.ConfigParser()

    #setting_day = pd.to_datetime(config_0['EXE']['SETTING_DAY'])
    # total_budget = int(config_0['CAMPAIGN']['TOTAL_BUDGET'])
    # total_days = int(config_0['CAMPAIGN']['TOTAL_DAYS'])

    # config_path = '../config/config_branded.ini'
    # config_path = '../config/config_non_branded.ini'
    config_path = '../config/config_non_branded_NuroBiz.ini'
    #config_path = '../config/config_non_branded_Easy_Predictive_Analytics.ini'

    s3_path = 'local/'
    s3_bucket = 'local'
    # stopped_keyword_list = ["PC 管理"] #stopped keyword list means the genearted keywords have a low performance, reasons are not clear so requires LLM analysis
    # code_in = 0: 前回Policy実行／Google広告設定変更は正常終了（PolicyからのCodeで 1, 2, 3が指定されたケース）
    #         = 1：前回Policy実行／Google広告設定変更で異常発生（PolicyからのCodeで4が指定、Policyで予期しないExceptionがraiseされた、キャンペーン設定変更に失敗、などのケース）
    # 設定日判定（曜日による判定）はしていただく前提で、初期化時に渡したCodeが0の場合はその判定に従って、Codeが1の場合は、判定の結果に関わらず強制的に処理を実行
    code_in = 1
    setting_day = pd.to_datetime("2024-11-2")  # for testing")

    current_keyword_list = [{"keyword": 'キッティングセンター', "match_type": 'PHRASE'},
                            {"keyword": 'PCの販売', "match_type": 'PHRASE'},
                            {"keyword": 'キッティング', "match_type": 'PHRASE'}]

    ad_agent = Ad_agent(s3_bucket, s3_path, config_path, setting_day, code_in)
    #keyword_list, ad_text_dic, code_out = ad_agent.run(cb_get_kw_metrics, cb_exec_kw_plan, cb_get_ad_metrics,
                                                      # current_keyword_list, stopped_keyword_list)
    keyword_list, ad_text_dic, code_out, stopped_kw_list = ad_agent.run(cb_get_kw_metrics, cb_exec_kw_plan, cb_get_ad_metrics,
                                                       current_keyword_list)

    print("the keyword list is ", keyword_list)
    print("the ad_text_dic is ", ad_text_dic)
    print("the code is ", code_out)
    print("the stopped_keyword_list is ", stopped_kw_list)


if __name__ == "__main__":
    main()
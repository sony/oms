import configparser
import pandas as pd
from io import StringIO
from langchain_openai import AzureChatOpenAI

def clean_raw_file(filepath: str) -> pd.DataFrame:
    """
    Read the raw search keyword report file, remove initial header text and
    final summary rows (e.g. rows starting with "Total:"), then return a clean DataFrame.
    """
    with open(filepath, "r", encoding="utf-16") as f:
        lines = f.readlines()
    
    # Find the header row by locating the line starting with "Keyword status"
    header_index = None
    for i, line in enumerate(lines):
        if line.startswith("Keyword status"):
            header_index = i
            break
    if header_index is None:
        raise ValueError("Header row starting with 'Keyword status' not found in file.")
    
    # From the header onward, filter out lines that are footer summary rows.
    cleaned_lines = []
    for line in lines[header_index:]:
        if not line.strip():
            continue  # skip empty lines
        # Remove any footer rows that start with "Total:"
        if line.lstrip().startswith("Total:"):
            continue
        cleaned_lines.append(line)
    
    # Use StringIO to load the cleaned string into a DataFrame.
    clean_data = "".join(cleaned_lines)
    df = pd.read_csv(StringIO(clean_data), sep="\t", engine="python", quoting=3)
    
    # Optional: remove extra quotes and whitespace from the 'Keyword' column.
    if "Keyword" in df.columns:
        df["Keyword"] = (
            df["Keyword"]
            .astype(str)
            .str.replace('"""', '', regex=False)
            .str.replace('"', '', regex=False)
            .str.strip()
        )
    return df

def main():
    # -------------------------------
    # 1. Initialize the LLM using your config
    # -------------------------------
    config = configparser.ConfigParser()
    config.read("../config/config_category_initialization.ini")
    
    llm = AzureChatOpenAI(
        deployment_name=str(config['LLM']['deployment_name']),
        openai_api_version=str(config['LLM']['openai_api_version']),
        #openai_api_key=str(config['KEYS']['OPENAI_GPT4O_API_KEY']),
        #azure_endpoint=str(config['KEYS']['OPENAI_GPT4O_AZURE_OPENAI_ENDPOINT']),
        #temperature=float(config['LLM']['temperature']),
        openai_api_key = str(config['KEYS']['OPENAI_03mini_API_KEY']),
        azure_endpoint=str(config['KEYS']['OPENAI_03mini_AZURE_OPENAI_ENDPOINT']),
    )
    
    # -------------------------------
    # 2. Clean the raw file and save as a TSV file
    # -------------------------------
    input_filepath = "./history_data/raw_data/Search keyword report.csv"  # update with your file path
    df = clean_raw_file(input_filepath)
    df.to_csv("./history_data/raw_data/cleaned_keywords.csv", encoding="utf-16", sep="\t", index=False)
    print("Cleaned data saved as 'cleaned_keywords.csv'.")
    
    # -------------------------------
    # 3. Analyze keywords to generate five categories
    # -------------------------------
    # Create a long string of unique keywords.
    all_keywords = df["Keyword"].dropna().unique().tolist()
    keywords_list_str = "\n".join(all_keywords)
    
    categories_prompt = f"""
These are my information for my product.
"PC LCMは新OS Windows 11へのスムーズな切り替えをアシストするNURO Bizのキッティングサービスです。" \
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

Based on this information, please analyze these keywords and suggest exactly five distinct category names that best represent common themes.
Provide the five Japanses category names as a comma-separated list with no additional explanation.
The keywords are:
{keywords_list_str}
"""
    categories_response = llm.predict(categories_prompt)
    print("LLM categories response:", categories_response)
    
    # Parse the categories from the response (assumes comma separation)
    categories = [cat.strip() for cat in categories_response.split(",") if cat.strip()]
    if len(categories) != 5:
        print("Warning: Expected 5 categories, but got", len(categories), ":", categories)
    
    # -------------------------------
    # 4. Function to classify each keyword
    # -------------------------------
    def classify_keyword(keyword: str, categories_list: list) -> str:
        prompt = f"""
Given the following list of categories: {', '.join(categories_list)}.
Classify the following keyword into one of these categories: "{keyword}".
Respond with only the category name.
"""
        response = llm.predict(prompt)
        return response.strip()
    
    # Apply the classification to every row.
    df["Category"] = df["Keyword"].apply(lambda kw: classify_keyword(kw, categories))
    
    # -------------------------------
    # 5. Save the DataFrame including the new category column to a TSV file
    # -------------------------------
    #df.to_csv("./history_data/neuro_biz_category.csv", encoding="utf-16", sep="\t", index=False)
    #print("The final data with categories has been saved.")

if __name__ == "__main__":
    main()
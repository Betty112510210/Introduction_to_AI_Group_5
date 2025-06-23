# Youtube Comments Sentiment Analysis

Emotion Analysis Using LLMs model

## Project Description

Cyberbullying has become an unignorable issue with the rise of the internet, contributing to mental health problems, social hostility, and even retaliatory behavior. In addition to native speakers, language learners often misuse words when expressing their opinions, which can lead to misunderstandings or unintentionally harmful comments. Our model is designed to analyze YouTube user comments to determine whether they are emotionally charged or potentially abusive. It provides feedback and sentiment scores for comments before they are posted. The project collect data using the YouTube API and manual labeling, identifying harmful comments through classification, analysis, and deep learning techniques. Each comment is assigned a sentiment score ranging from -1 to 1, where lower scores indicate more negative or aggressive content. This tool is especially helpful for non-native English speakers to revise and improve their comments before sharing them online.

This project analyzes YouTube comments using sentiment analysis and cyberbullying detection models, collecting over 6,000 YouTube comments as the raw dataset. Using a large language model (GPT-4o, which yielded the most consistent performance in our tests), we labeled comments into three sentiment categories: **negative, neutral, and positive.** A total of 3,000 comments, 1,000 from each category, were then selected to train our sentiment analysis model.

To evaluate labeling strategies, we tested multiple prompting approaches by manually labelling 100 sample comments for comparison:

- Direct sentiment classification
- Contextual framing for cyberbullying detection
- Contextual framing for hate speech detection

## Getting Started

### Step1: Youtube Comments Fetching

### 📦 Prerequisites
Please make sure you have the following:

- Python 3.8 or above
- pip (Python package manager)
- Install required packages:

 ```python
pip install google-api-python-client langdetect emoji
 ```

### 🔑 API Setup
- Go to Google Cloud Console, create a project and enable YouTube Data API v3.
- Generate an API key.
- Store the key securely (e.g., in an environment variable or config file. **Do not hardcode your key directly into public scripts.**)

### 📄 Usage: YouTube Comment Scraper
Use the script below to retrieve up to 200 English YouTube comments and replies with emoji retained:

 ```python

from googleapiclient.discovery import build
from langdetect import detect
import csv
import time

api_key = 'YOUR_YOUTUBE_API_KEY' # Replace with your API key and target video ID
video_id = 'YOUR_VIDEO_ID' # You could find it in the website of the video
youtube = build('youtube', 'v3', developerKey=api_key)

def clean_comment(text): # clear irrelevent space but keep the emoji.
    text = text.replace('<br>', ' ')
    text = text.replace('\n', ' ')
    return text.strip()

def is_english(text): # scratch only English comments for better annalysis.
    try:
        temp = clean_comment(text)
        return detect(temp) == 'en'
    except:
        return False

def get_top_related_comments(video_id, max_results=200): #scratch only 200 comments per video for diverse dataset
    results = []
    next_page_token = None

    while len(results) < max_results:
        response = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id,
            maxResults=100,
            order='relevance',
            textFormat='plainText',
            pageToken=next_page_token
        ).execute()

        for item in response['items']: # main comments
            comment = item['snippet']['topLevelComment']['snippet']
            author = comment.get('authorDisplayName', '')
            text = comment['textDisplay']
            published_at = comment['publishedAt']

            if is_english(text):
                cleaned = clean_comment(text)
                results.append([author, cleaned, published_at, False])

            if 'replies' in item: # reply comments
                for reply in item['replies']['comments']:
                    r = reply['snippet']
                    reply_text = r['textDisplay']
                    reply_author = r.get('authorDisplayName', '')
                    reply_time = r['publishedAt']

                    if is_english(reply_text):
                        cleaned_reply = clean_comment(reply_text)
                        results.append([reply_author, cleaned_reply, reply_time, True])

            if len(results) >= max_results:
                break

        if len(results) >= max_results or 'nextPageToken' not in response:
            break

        next_page_token = response.get('nextPageToken')
        time.sleep(0.5)

    return results[:max_results]

comments_data = get_top_related_comments(video_id, max_results=200)

filename = 'top_200_english_with_emoji.csv'
with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['Author', 'Comment', 'PublishedAt', 'IsReply'])
    writer.writerows(comments_data)

print(f'✔ Saved {len(comments_data)} comments to {filename}')
 ```
### 📂 Output
The output is a CSV file containing:

- Author: Commenter's name
- Comment: Cleaned text (emoji preserved)
- PublishedAt: Timestamp
- IsReply: Whether the comment is a reply

### Step2: Comments Sentiment Label with API

### 📦 Prerequisites
Please make sure the following packages are installed in your environment:
 ```python
pip install openai pandas tqdm
 ```
This script is designed to run on Google Colab or local environments with access to OpenAI API and CSV-based comment data.

### 🔑 API Setup
Sign up for access to the OpenAI API at https://platform.openai.com

Generate your secret API key

Replace the following line in your script with your own key (do not upload your key to public repositories):
 ```python
client = OpenAI(api_key="your-api-key")
 ```

### 🧠 Prompt Engineering
To evaluate the labeling quality of GPT-4o, we designed specialized prompts for different tasks. Rather than relying on minimal instructions, each prompt embeds context, definitions, and examples to simulate expert human judgment.

1. Sentiment Classification Prompt (Label Accuracy:89%)
Model is asked to classify YouTube comments into **positive, neutral, or negative** with the awareness of internet slang, tone and emoji:
 ```python
prompt= f"""
請參考 YouTube 留言與網路流行用字，
你是專業的網路評論分析師，
判斷以下留言的情緒，只回覆一個詞：
positive、neutral 或 negative。
留言：{comment}
"""
 ```

2-1. Hate Speech Detection Prompt (Label Accuracy:65%)
Includes a formal definition of hate speech (based on identity-based attacks and social harm) and asks for binary judgment:
 ```python
prompt= f"""
你是一位專業的網路評論分析師，熟悉 YouTube 留言與網路用語。

仇恨言論（Hate speech）的定義如下：
仇恨言論是指攻擊、貶低、歧視或煽動對特定個人或群體的敵意言論，
特別是針對其種族、性別、宗教、性取向、身心障礙、國籍等身份。
這些言論可能帶有人身攻擊、激起仇恨、煽動暴力或社會對立，
並可能引發閱讀者的負面情緒或社會影響。

請你根據這個定義，判斷下列留言是否屬於仇恨言論。
請只回答「會」或「不會」，不要補充任何說明或理由。

留言：
{comment}
"""
 ```

2-2 Hate Speech Detection Prompt without discription (Label Accuracy: 34%)
Model is asked to classify whether YouTube comments invovled hatred speech with the awareness of internet slang, tone and emoji and no context provided: 
 ```python
prompt= f"""
你是一位專業的網路評論分析師，熟悉 YouTube 留言與網路用語。
請參考 YouTube 留言與網路流行用字，
你是專業的網路評論分析師，
判斷以下留言是否可能造成仇恨言論（Hate speech），
只回覆一個詞：會、不會。
留言：{comment}
"""
 ```

3-1. Cyberbullying Detection Prompt (Label Accuracy:76%)
Provides contextual examples of sarcasm, group mockery, and verbal abuse, asking whether a comment qualifies as cyberbullying:
 ```python
prompt= f"""
你是一位熟悉社群媒體與網路文化的評論分析師。

網路霸凌是指透過文字、語氣、表情、諷刺等形式，在網路上攻擊、羞辱、
貶低、孤立、排擠或嘲笑某人。這些言論可能不是直接罵人，
卻仍造成他人情緒傷害、引發對特定個人的敵意或群體排擠。

- 冷嘲熱諷或語帶攻擊（例如：「好棒棒喔」「真有你的，出來丟臉」）
- 貶低外貌、智商、行為（例如：「看他那德行就知道了」「腦袋有問題吧」）
- 侮辱、攻擊、群嘲（例如：「他這種人活該被罵」「大家都知道他很爛」）
- 用笑話或嘲諷語氣掩飾攻擊意圖

請根據上述定義，判斷以下留言是否屬於網路霸凌。
請只回答「會」或「不會」，不要補充理由或其他文字。

留言：
{comment}
"""
 ```

3-2 CyberBullying Detection Prompt without discription (Label Accuracy: 67%)
Model is asked to classify whether YouTube comments invovled cyberbullying content with the awareness of internet slang, tone and emoji and no context provided: 
 ```python
prompt= f"""
你是一位熟悉社群媒體與網路文化的評論分析師。
請參考 YouTube 留言與網路用語，你是專業的網路評論分析師，判斷以下留言是否可能構成網路霸凌（Cyberbullying），例如冷嘲熱諷、群體攻擊、羞辱或造成情緒傷害。只回覆一個詞：「會」或「不會」。
留言：{comment}
"""
 ```

### Step3: Structure Analysis Model


## File Structure

[Describe the file structure of your project, including how the files are organized and what each file contains. Be sure to explain the purpose of each file and how they are related to one another.]

## Analysis

[Describe your analysis methods and include any visualizations or graphics that you used to present your findings. Explain the insights that you gained from your analysis and how they relate to your research question or problem statement.]

## Results

Due to the low accuracy in detecting hatred, cyberbullying, and sentimental comments—initially yielding 34%, 67%, and 89% under evaluations involving mixed emotions and negative speech—we shifted from intent-based labeling to sentiment-based labeling. Using prompts such as “You are a professional comments analyst, please evaluate the following YouTube comments and identify hatred (cyberbullying and sentimental) content,” results remained inconsistent. Manual labeling allowed us to better differentiate among hateful, cyberbullying, and general sentiment comments. Importantly, not all negative comments contain cyberbullying elements; “sad” comments, often non-aggressive, were filtered separately. Users were also reminded to clarify emotional expressions to avoid political misclassification. Additionally, we extracted 100 negative samples for manual relabeling, and after processing them with a large language model (LLM), accuracy improved to 89%.

## Contributors

| Avatar | Name | Role(s) |
|--------|------|---------|
| <img src="https://github.com/liangli-liu.png" width="40"/> | [柳亮力 Liang-Li Liu](https://github.com/liangli-liu) | Project manager, program writer, analysis model structure |
| <img src="https://github.com/Changtzuan.png" width="40"/> | [張子安 Andy Chang](https://github.com/Changtzuan) | Project manager, program writer, data collecting, data mining |
| <img src="https://github.com/Betty112510210.png" width="40"/> | [陳郁宣 Yuhsuan Chen](https://github.com/Betty112510210) | Project manager, program writer, finetuning, presentation visualizer |


## Acknowledgments

We would like to express our sincere gratitude to **Professor Pien** for the invaluable guidance and support throughout the development of this project. His expertise greatly contributed to shaping our research direction and refining the design of our sentiment analysis framework.

We also acknowledge the use of the following resources:

- **YouTube Data API v3**, for providing access to public comment data used in our dataset.
- **OpenAI GPT-4o (via API access)**, for assisting in large-scale comment labelling and sentiment classification.


## References

This project was built using a variety of tools, libraries, and data sources:

### 🧰 Programming Languages & Libraries
- **Python** – Main language for data collection, processing, and model development
- **Google API Client for Python** – To access YouTube Data API v3
- **langdetect** – For language identification
- **emoji** – For emoji-preserving comment processing
- **OpenAI GPT-4o API** – For comment labeling and LLM-based sentiment evaluation
- **R (dplyr)** – Used during early stages for data wrangling and cleaning

### 📊 Data Sources
- **YouTube Comments** – Collected via YouTube Data API v3  

### 🧪 Analytical Methods
- **LLM Prompt Engineering** – Comparing direct sentiment labeling with hate speech and cyberbullying contextual prompts
- **Manual vs. Automated Label Comparison** – For accuracy benchmarking



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

We provide a standalone script to collect up to 200 English YouTube comments and replies (with emoji retained) from any public video using the YouTube Data API.

📄 Full script: [`youtube_api_scratch.py`](youtube_api_scratch.py)

```python
python scripts/youtube_scraper.py
 ```

### 📂 Output
The output is a CSV file containing:

- `Author`: Commenter's name
- `Comment`: Cleaned text (emoji preserved)
- `PublishedAt`: Timestamp
- `IsReply`: Whether the comment is a reply

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

Replace the following line in your script with your own key (**Do not hardcode your key directly into public scripts.**):
 ```python
client = OpenAI(api_key="your-api-key")
 ```

### 🧠 Prompt Engineering
To evaluate the labeling quality of GPT-4o, we designed different prompts for the tasks. Onw with minimal instructions or context as resoning prompts and the other simple request to test the performance of LLM.

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

Includes a **formal definition of hatred speech** (based on identity-based attacks and social harm) and asks for binary judgment:
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

Model is asked to classify whether YouTube comments **invovled hatred speech** with the awareness of internet slang, tone and emoji and no context provided: 
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

Includes a **formal definition of cyberbullying** (based on contextual examples of sarcasm, group mockery, and verbal abuse) and asks for binary judgment:
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

Model is asked to classify whether YouTube comments **invovled cyberbullying** content with the awareness of internet slang, tone and emoji and no context provided: 
 ```python
prompt= f"""
你是一位熟悉社群媒體與網路文化的評論分析師。
請參考 YouTube 留言與網路用語，你是專業的網路評論分析師，判斷以下留言是否可能構成網路霸凌（Cyberbullying），例如冷嘲熱諷、群體攻擊、羞辱或造成情緒傷害。只回覆一個詞：「會」或「不會」。
留言：{comment}
"""
 ```

### Step3: Structure Analysis Model

### 🔍 Functionality
Given a user-submitted comment, the model will:

- Classify the comment as positive, neutral, or negative
- Return a sentiment score ranging from -1 (very negative) to 1 (very positive)
- If the sentiment is negative, suggest a more neutral, respectful rewrite

### ⚙️ Requirements
Before running the tool, make sure to install the necessary packages:

 ```python
pip install openai gradio pandas
 ```
- Python 3.8 or above
- An OpenAI API key (inserted as your_api_key)
- CSV data file (3000_youtube_sentiment.csv) for offline testing or future fine-tuning

### 🧪 Example Output
Input:

` “This video is ridiculous. I can’t believe people actually like it.” `

Model Output:

 ```python
Classification: negative  
Sentiment Score: -0.87  

Suggested Rewrite:  
"Personally, I didn’t find this video helpful, but I understand others may enjoy it."
 ```

### 💻 Running the Interface
To launch the Gradio web app:

 ```python
python sentiment_model.py
 ```
You’ll see a text input box like:

` “Please enter the YouTube comment you would like to post…” `

The output will display below with sentiment evaluation and, if necessary, rewriting advice.

## File Structure

### 📁 Full Structure
 ```python
Introduction_to_AI_Group_5/
├── data/                            # All intermediate and labeled datasets
│   ├── top_200_english_with_emoji.csv     # Raw scraped comments from YouTube
│   ├── sample_100.csv                     # Human-labeled 100-sample set for benchmark comparison
│   ├── sample_100_label.csv               # GPT-labeled output for sample_100
│   ├── negative_sample.csv                # Filtered negative comments
│   └── negative_sample_bully.csv          # Binary output from cyberbullying classification
├── images/                         # Workflow or analysis visuals
│   └── model_process.png
├── youtube_api_scratch.py          # Step 1: Scrape YouTube comments
├── chatgpt_label.py                # Step 2: Generate GPT labels with various prompt types
├── sentiment_analysis_+_rewriting_suggestions.py             # Step 3: Real-time comment classification + rewrite via Gradio
├── sentiment_analysis_+_rewriting_suggestions.py         # Step 4: Prompt accuracy comparison and evaluation visuals
└── README.md                       # Project overview, instructions, results
 ```

### 🔗 Project Pipeline Overview
| Stage                 | Script                    | Output                                         |
|----------------------|---------------------------|------------------------------------------------|
| 1. YouTube Scraper   | `youtube_api_scratch.py`  | Raw comment data (`.csv`)                      |
| 2. LLM Labeling      | `chatgpt_label.py`        | Labeled data using prompt-based methods        |
| 3. Feedback Assistant| `sentiment_analysis_+_rewriting_suggestions.py`      | Real-time sentiment + rewriting tool           |
| 4. Evaluation        | `sentiment_analysis_+_rewriting_suggestions.py`   | Accuracy comparison and prompt insights        |

### 🔍 File Purpose Summary
`youtube_api_scratch.py`
Retrieves YouTube comments using the YouTube Data API v3, filters English content, and retains emoji.

`chatgpt_label.py`
Applies various prompt strategies with GPT-4o for sentiment, hate speech, and cyberbullying classification.

`sample_100.csv / sample_100_label.csv`
Benchmark dataset for evaluating prompt performance.

`negative_sample.csv / negative_sample_bully.csv`
Refines the distinction between emotionally vulnerable (sad) and harmful (bullying) negative comments.

`sentiment_analysis_+_rewriting_suggestions.py`
Provides an interactive tool via Gradio for real-time sentiment analysis and rewrite suggestions for user comments.

`images`
Contains visuals like workflow diagrams for use in the README.

## Analysis

To evaluate how language and tone contribute to verbal hostility, we conducted a multi-phase analysis combining LLM-based labeling, manual annotation, and performance comparison between different prompt strategies. We labeled 6,000 YouTube comments using GPT-4o and selected 3000 of them, 1000 per class (positive, neutral, negative) for thorugh training and evaluation. Among all tested approaches, **direct sentiment classification** achieved the highest alignment with human labelling (89%), compared to 76% for cyberbullying prompts with context and 65% for hate speech prompts with context.

### 🔍 Sadness Detection: A Critical Distinction
During the labeling process, we discovered that not all negative comments were harmful. Some were expressions of personal sadness or frustration, rather than aggression. To avoid misclassifying these as abusive, we implemented an additional step to distinguish sad emotional content within the negative category.

![sentimental analysis   on youtube comment (59 4 x 84 1 公分)](https://github.com/user-attachments/assets/5c3074c9-bcac-4c12-bfe3-0ffef60020a1)
Figure 1: Comments classified as sad were excluded from moderation flags and instead received comfort-oriented feedback—supportive messages that encourage thoughtful communication rather than punishment. This design promotes a more empathetic environment, especially for emotionally vulnerable or non-native users.

### 💡 Key Process
By differentiating emotional expression from aggression, our system addresses key research questions:
How can AI moderate online discourse without misjudging emotional intent?

![Intro to ai proposal](https://github.com/user-attachments/assets/b4528263-4ab0-4916-ba24-0256197ca46f)
Figure 2: This structure ensures that moderation is both accurate and emotionally aware, helping to prevent unintended harm while supporting constructive expression particularly for non-native English speakers. Our findings emphasize that contextual sentiment understanding is essential for fair and socially sensitive comment moderation.

1. Comment Input – A user-submitted YouTube comment enters the system.
2. Sentiment Classification – The comment is categorized as positive, neutral, or negative.
3. Negative Comment Evaluation – Negative comments are further distinguished as sad or bullying.
4. Feedback Generation – Based on the analysis, the system returns one of three feedback types:
- No change (positive/neutral)
- Supportive message (sad)
- Warning or suggestion (bullying)
5. User Decision – The user receives the feedback before posting and can choose to revise or proceed.


## Results

### 🧠 Ultimate Model
After testing multiple prompts, including hate speech detection, cyberbullying identification as well as direct sentiment classification, we found that **sentiment-based prompts** yielded the most reliable results. When evaluated against a manually labeled dataset of 100 comments, prompt accuracy was 34% for hate speech, 67% for cyberbullying, and 89% for sentiment classification. Though providing context for cyberbullying and hatred speech in the resoning prompt, the accuracy only raised to 76% for cyberbullying label and 65% for hatred speech labelling; thus, **direct sentiment-based label** was choosed as our method.

As a result, we adopted sentiment-based labeling using GPT-4o as our primary approach. Based on manual inspection of negative comments, we identified a recurring subset of emotionally vulnerable `sad` comments. These were manually separated and treated differently in our response strategy: instead of moderation warning, they received comfort-oriented feedback.

This dual-path design helped avoid over-policing emotional expression while still identifying genuinely harmful content. Using the sentiment-based approach that ultimately achieved the highest alignment with human labels; meanwhile, providing a more empathetic, user-sensitive experience for non-native English speakers.

![IMG_1607](https://github.com/user-attachments/assets/f472f191-9490-4ee8-9afd-4668eecec561)
Figure 3: Ultimate model output.

### 🔭 Future Directions
While GPT-4o demonstrated strong performance in sentiment-based labeling, we observed limitations when handling ambiguous specific expressions. To further improve accuracy, particularly for borderline cases such as sarcastic bullying or ironical hate speech, we plan to finetune a language model based on our manually labeled dataset.

Our future goal is to train a task-specific model that better reflects the nuanced definitions of cyberbullying and emotional vulnerability in real-world comments. This fine-tuned LLM would serve as a better classifier for raw comment input and feedback generation, improving consistency, and adapting more effectively to user context.

We believe this next step will allow us to deliver more robust, explainable, and context-aware moderation tools  useful for educational platforms or multilingual digital communities.

## Contributors

| Avatar | Name | Role(s) |
|--------|------|---------|
| <img src="https://github.com/Jessie111508021.png" width="40"/> | [柳亮力 Liang-Li Liu](https://github.com/Jessie111508021) | Project manager, program writer, analysis model structure |
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
- **R (`dplyr`)** – Used during early stages for data wrangling and cleaning

### 📊 Data Sources
- **YouTube Comments** – Collected via YouTube Data API v3  

### 🧪 Analytical Methods
- **LLM Prompt Engineering** – Comparing direct sentiment labeling with hate speech and cyberbullying contextual prompts
- **Manual vs. Automated Label Comparison** – For accuracy benchmarking



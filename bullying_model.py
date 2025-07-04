# -*- coding: utf-8 -*-
"""bullying model

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17hvlP2ConB34e2Z3hKYYIV_r85qjl3_L
"""

# Step 0: Install necessary packages
!pip install openai gradio

# Step 1: Mount Google Drive and load data
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from openai import OpenAI
import gradio as gr

# Step 2: CSV
file_path = '/content/drive/MyDrive/Colab Notebooks/intro AI/期末data/label/3000_youtube_sentiment.csv'
df = pd.read_csv(file_path)
df = df[['Author', 'Comment', 'label']].copy()
df['text'] = df['Author'].fillna('') + ": " + df['Comment']

machine = OpenAI(api_key="yor api key")

# Step 4: Analyze the function
def analyze_comment(comment):
    prompt = f"""
You are a helpful assistant trained to detect cyberbullying in online comments. Analyze the following message and do:

Step 1: Classify it as 'bullying' or 'non-bullying'.
Step 2: Give a sentiment aggression score between -1 (most aggressive) to 1 (very positive).
Step 3: If bullying, provide a more friendly rephrasing suggestion.
Step 4: If the comment sounds sad or depressed, remind the user with a friendly note.

Comment: "{comment}"

Output format:
Bullying Status: ...
Score: ...
Suggestion: ...
Reminder: ...
"""
    response = machine.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a cyberbullying detection assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

for i, row in df.iterrows():
    comment = row['text']

    # Step 1~4: Sentiment analysis and bullying detection
    prompt = f"""
You are a helpful assistant trained to detect cyberbullying in online comments. Analyze the following message and do:

Step 1: Classify it as 'bullying' or 'non-bullying'.
Step 2: Give a sentiment aggression score between -1 (most aggressive) to 1 (very positive).
Step 3: If bullying, provide a more friendly rephrasing suggestion.
Step 4: If the comment sounds sad or depressed, remind the user with a friendly note.

Comment: "{comment}"

Output format:
Bullying Status: ...
Score: ...
Suggestion: ...
Reminder: ...
"""

    try:
        response = machine.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a cyberbullying detection assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        result_text = response.choices[0].message.content.strip()

        # Check if it is bullying, if so rewrite the suggestion
        if "Bullying Status: bullying" in result_text.lower():
            rephrase_prompt = f"""
You are a writing coach. A user wrote the following comment on a YouTube video:

"{comment}"

1. Point out what parts of the comment may sound overly negative, offensive, or inappropriate.
2. Explain why it might be problematic or hurtful to some readers.
3. Provide a more neutral and respectful version of the comment.
"""
            rephrase_response = machine.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You help users rewrite negative comments in a respectful tone."},
                    {"role": "user", "content": rephrase_prompt}
                ]
            )
            suggestion = rephrase_response.choices[0].message.content.strip()
            result_text += f"\n\n🔁 改寫建議（含原因與版本）：\n{suggestion}"

    except Exception as e:
        result_text = f"Error: {e}"

# Step 5: Create the Gradio interface
iface = gr.Interface(
    fn=analyze_comment,
    inputs=gr.Textbox(lines=4, placeholder="請輸入留言..."),
    outputs="text",
    title="🛡️ Comment Sentiment & Cyberbullying Detector",
    description="輸入一段留言，系統將分析其情緒、是否具霸凌傾向，並提供改寫建議或友善提醒。"
)

iface.launch()

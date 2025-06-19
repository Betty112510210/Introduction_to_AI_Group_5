# Youtube Comments Sentiment Analysis

Emotion Analysis Using LLMs model

## Project Description

[Enter a brief description of your project, including the data you used and the analytical methods you applied. Be sure to provide context for your project and explain why it is important.]

## Getting Started

[Provide instructions on how to get started with your project, including any necessary software or data. Include installation instructions and any prerequisites or dependencies that are required.]

This project analyzes YouTube comments using sentiment analysis and cyberbullying detection models. It includes a script to fetch and clean comments (with emoji preserved), filter for English content, and export the data for further analysis using LLMs or deep learning models.

### Youtube Comments Fetching

### Step1: 📦 Prerequisites
Please make sure you have the following:

- Python 3.8 or above
- pip (Python package manager)
- Install required packages:

 ```python
pip install google-api-python-client langdetect emoji
 ```

### Step2: 🔑 API Setup
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
## File Structure

[Describe the file structure of your project, including how the files are organized and what each file contains. Be sure to explain the purpose of each file and how they are related to one another.]

## Analysis

[Describe your analysis methods and include any visualizations or graphics that you used to present your findings. Explain the insights that you gained from your analysis and how they relate to your research question or problem statement.]

## Results

[Provide a summary of your findings and conclusions, including any recommendations or implications for future research. Be sure to explain how your results address your research question or problem statement.]

## Contributors

[List the contributors to your project and describe their roles and responsibilities.]

## Acknowledgments

[Thank any individuals or organizations who provided support or assistance during your project, including funding sources or data providers.]

## References

[List any references or resources that you used during your project, including data sources, analytical methods, and tools.]

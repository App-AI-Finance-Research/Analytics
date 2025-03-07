import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from datetime import datetime, timedelta
import re

# Initialize Hugging Face Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # Run on CPU

def get_et_news(company):
    """Fetches news headlines & links from The Economic Times for the given company from the last 2 weeks."""
    search_url = f"https://economictimes.indiatimes.com/topic/{company.replace(' ', '-')}"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("div", class_="eachStory")  

    # Get the date limit (two weeks ago)
    date_limit = datetime.now() - timedelta(days=14)
    
    news_list = []
    for article in articles:
        # Extract headline and description
        headline = article.find("h3").text.strip()
        description = article.find("p").text.strip() if article.find("p") else ""
        
        # Extract the article date (if available)
        date_text = article.find("time")
        if date_text:
            date_match = re.search(r'(\d{1,2}) (\w+) (\d{4})', date_text.text)  # Extract date like '6 Mar 2024'
            if date_match:
                day, month, year = date_match.groups()
                article_date = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
                
                # Skip articles older than 2 weeks
                if article_date < date_limit:
                    continue
        
        news_list.append(f"{headline} - {description}")
        if len(news_list) >= 3:  # Limit to the latest 3 news articles
            break
    
    return news_list

def summarize_news(news_list):
    """Summarizes news using an LLM model."""
    summaries = []
    for news in news_list:
        summary = summarizer(news, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries

def add_financial_commentary(summary):
    """Adds financial insights based on keywords in the summary."""
    keywords = {
        "revenue": "indicating possible top-line growth.",
        "profit": "which might boost PAT.",
        "loss": "potential negative impact on profitability.",
        "cost": "suggesting operational efficiency or inefficiency.",
        "EBITDA": "impacting operational performance."
    }
    
    commentary = [f"The news mentions {key}, {val}" for key, val in keywords.items() if key in summary.lower()]
    
    return summary + " " + " ".join(commentary)

# Streamlit App UI
st.title("Indian Company News Summarizer")
st.write("Fetch and summarize the latest financial news from The Economic Times (last 2 weeks only).")

company = st.text_input("Enter the Indian company name:")

if st.button("Get News Summary"):
    if not company:
        st.warning("Please enter a company name.")
    else:
        st.info(f"Fetching news for *{company}* (from the last 2 weeks)...")
        news_list = get_et_news(company)
        
        if not news_list:
            st.error("No recent news found in the last two weeks.")
        else:
            summaries = summarize_news(news_list)
            
            for i, summary in enumerate(summaries):
                st.subheader(f"News {i+1} Summary:")
                st.write(add_financial_commentary(summary))

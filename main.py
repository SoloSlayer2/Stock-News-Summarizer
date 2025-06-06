import validators
import streamlit as st
from typing import List,TypedDict
from Stock_News_Summarizer import NewsSummarizer
from gtts import gTTS
import os

class URL(TypedDict):
    url:List[str]
    url_id:List[int]

def url_check(url: str) -> bool:  
    if not url or url.strip() == "":
        return False
    if not validators.url(url):
        return False
    return True

def clean_text_for_tts(text: str) -> str:
    """Prepare the result from AI to be used by the TTS"""
    import re
    cleaned = re.sub(r"^[\*\-\•]\s+", "", text, flags=re.MULTILINE)
    cleaned = cleaned.replace("*", "")
    return cleaned


def main():
    st.title("Stock News Summariser")
    st.sidebar.title("News Article URLs")
    summarizer=NewsSummarizer()

    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    x=st.slider(label="Enter number of URL you want to process",min_value=1,max_value=5)

    urls: URL = {"url": [], "url_id": []}
    for i in range(x):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls["url"].append(url)
        urls["url_id"].append(i+1)

    process_url = st.sidebar.button(label="Process URL")

    if process_url:
        valid_urls = [url for url in urls["url"] if url_check(url)]
    
        if not valid_urls:
            st.error("All URLs you provided are incorrect ❌😭")
        else:
            with st.spinner("Processing your given docs..."):
                try:
                    docs=summarizer.load_news(urls=valid_urls)

                    chunks=summarizer.create_chunks(docs=docs)

                    st.session_state.vectordb=summarizer.create_vectordb(chunks=chunks)

                    st.session_state.processed=True
                    st.success("URLs processed successfully! ✅")
                
                except Exception as e:
                    st.error(f"Error processing URLs: {str(e)}")
            
    query = st.text_input(
            label="Enter your question ❓",
            placeholder="What is the news on ONGC?",
            disabled=not st.session_state.processed
            )
    if query and query.strip() and st.session_state.processed:
        try:
            with st.spinner("Generating Summary..."):
                context, sources = summarizer.retrieve_query(query=query, vectordb=st.session_state.vectordb)
                results = summarizer.SummaryRAG(query=query, context=context)
        
            st.header("📝 Summary")
            st.write(results)

            if st.button("🔊 Listen to Summary"):
                try:
                    cleaned_text = clean_text_for_tts(text=results)
                    tts = gTTS(text=cleaned_text, lang='en')
                    tts.save("summary.mp3")
                    with open("summary.mp3", "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")
                    os.remove("summary.mp3")

                except Exception as e:
                    st.error(f"Error in generating or playing audio: {str(e)}")


            if sources:
                st.markdown("#### 🔗 Citations / Sources")
                for i, src in enumerate(sources, 1):
                    st.markdown(f"{i}. [{src}]({src})")
            else:
                st.info("No source links available for this summary.")
    
        except Exception as e:
            st.error(f"Error generating answer: {type(e).__name__}: {e}")
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset App"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    main()
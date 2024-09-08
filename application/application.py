# Import libraries
import streamlit as st
import pandas as pd
import os
import numpy as np
import faiss   
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from datetime import datetime  
import altair as alt



# Load the pre-built FAISS index and model
index = faiss.read_index('retrieval_index.index')
model = SentenceTransformer('saved_model/')

# Load your dataframe for summaries
df_summary = pd.read_excel('summaries.xlsx')
df_summary['Video Name'] = [' '.join(i.split('_')) for i in df_summary['Video Name']]

# Load your dataframe prepared for RAG - Index built on this dataframe
df_rag = pd.read_excel('prepared_data.xlsx')

# Authenticate OpenAI - Make sure your key is in OpenAI Key.txt
with open('OpenAI Key.txt', 'r') as file:
    key = file.read()
    
client = OpenAI(
    api_key=key,
)

# Function to make LLM requests
def llm_requests(prompt, client, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                    "role": "user", 
                    "content": prompt
            }
                ]
    )
    return response

# Prompt for RAG reponse
def prompt(question, context):
    return """You are tasked with answering questions based on transcripts from Indian Parliament discussions, particularly focused on the question hour and budget sessions. The question hour is an essential mechanism in the Lok Sabha where Members of Parliament question ministers on various topics, particularly on accountability and implementation of policies.

In the context of these discussions, several ministers and Members of Parliament are involved. Key figures include Rahul Gandhi, a leading opposition member from the Indian National Congress, often questioning the government’s policies, and Nirmala Sitharaman, the Finance Minister, responsible for defending the budget and outlining the government’s economic policies. However, many other ministers across various departments are also central to these debates, either presenting or defending specific proposals.

The texts still contain errors related to grammar, regional accents, incorrect numbers, and out-of-context words. Although the questions were generated from corrected text, your answers should be based on the provided transcripts, taking care to verify and correct any statistics or specific details mentioned. Aim to provide a clear, accurate, and well-informed response. Include any relevant web results if they help substantiate your answer.

Question: """ + question + """

Text for Context: """ + context

# Query rewriting - Utility functions
# Summary of summaries we will use as context to create a better question/query
summary = """### Consolidated Summary of Parliamentary Discussions

- **Budget and Economic Development**: The Finance Minister presented the budget aimed at achieving a developed India by 2047, emphasizing economic growth and welfare through substantial government expenditure, which is projected at ₹48.2 lakh crore, and increased capital expenditure for infrastructure development.

- **Fiscal Management and Sector Allocations**: The budget reflects positive trends in fiscal deficit management, aiming for a reduction below 4.5% of GDP by 2025. Key sector allocations are increased for agriculture, education, health, and women's welfare, debunking claims of funding cuts in these areas.

- **Support for Jammu and Kashmir**: The budget allocates ₹17,000 crore for development in Jammu and Kashmir, showing improvements in employment rates and introducing welfare initiatives for tribal communities to enhance healthcare and education.

- **Youth and Employment Initiatives**: New youth empowerment schemes focus on skill training and job creation, aiming to address unemployment rates, with confidence shown from recent statistics indicating improvements in labor market conditions.

- **Water Supply and Infrastructure Improvements**: Discussions on the Jal Jeevan Mission aimed at ensuring clean water access highlighted regional water supply issues and natural disaster management, emphasizing cooperation between central and state governments.

- **Healthcare Initiatives**: The government focused on drug pricing, affordable medicines, and the Ayushman Bharat scheme’s challenges, with efforts to monitor health expenditures and improve healthcare access, especially for vulnerable populations.

- **Agricultural Support and Financial Inclusion**: Initiatives to empower farmers included a significant increase in funding through credit schemes, reforms in crop insurance, and a new emphasis on sustainability and technology in agriculture.

- **MSME and Economic Growth**: MSMEs were acknowledged for their role in job creation and economic stability, with various government measures announced to enhance financial access and support innovation.

- **Environmental Concerns and Renewable Energy**: The discussions included strides in renewable energy adoption, addressing pollution, and strategies to manage waste, including updates on e-waste management rules and carbon emissions tracking.

- **Judicial and Legislative Reforms**: Issues surrounding judicial proposals, backlogs, and the need for regional language accessibility in legal proceedings were discussed, emphasizing the importance of reform in the legal system.

- **Child Welfare and Education**: Rising concerns about child welfare were raised, highlighting the need for better support systems for marginalized children, while educational discussions pointed to the urgent need for quality improvements and integrity in examination systems.

- **Infrastructure Development**: Various infrastructure projects, including railway expansions and highway improvements, were questioned for their progress, with calls for enhanced project monitoring and timely completion.

- **Parliamentary Conduct and Opposition Comments**: Observations on maintaining parliamentary decorum and effective opposition engagement were made, with Rahul Gandhi criticizing the budget's neglect of various societal needs, emphasizing a focus on marginalized groups and pressing for legislative changes.

- **Future Outlook**: The discussions conveyed a need for strategic investments in various sectors, greater representation for underserved communities, and an emphasis on compassion in governance amidst current socio-political challenges. 

This consolidated summary encapsulates vital outcomes and concerns from the parliamentary discussions, illustrating the government's initiatives and the pressing needs identified by opposition members and representatives from various sectors."""

# Prompt to modify user question
def modify_question_prompt(summary, question):
    return f"""
    You are tasked with modifying the following question based on the context provided by a summary of discussions. Your goal is to adjust the question without losing its original meaning or semantic value. The modification should incorporate any relevant details or key points from the summary, ensuring the question remains aligned with the context of the discussions.

    Summary:
    {summary}

    Original Question: {question}

    Provide a modified version of the question that takes the summary into account but does not alter the core meaning.
    """

# Function to call rewriting - Returns the modified question
def query_rewriting(summary, question, model_name):
    return llm_requests(modify_question_prompt(summary, question), client, model_name).choices[0].message.content

# Reranking - Utility functions
# Prompt for reranking
def select_top_relevant_matches_prompt(question, matches, top_n):
    return f"""
    You are tasked with selecting the top {top_n} most relevant passages from the following list, based on the question provided. The passages may contain varying levels of relevance, and your goal is to pick only the ones that are directly related to answering the question.

    The question is: "{question}"

    List of matches:
    {matches}

    Select the top {top_n} most relevant passages and output only those passages.
    """

# Define function to rerank the matches - Returns the reranks matches 
def rerank(df, question, model_name, k_value=10):
    question_vector = model.encode([question])
    result = index.search(question_vector, k_value)
    matches = [df['RAG Text'][result_index] for result_index in result[1][0]]
    # Request the LLM to rerank the matches
    return llm_requests(select_top_relevant_matches_prompt(question, matches, 5), client, model_name).choices[0].message.content.splitlines()



# Code for pages starts here



# Set the page title and favicon
st.set_page_config(
    page_title="ParlaMind",
    page_icon="logo.png", # Created with chatgpt
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
st.sidebar.title("Explore ParlaMind")
page = st.sidebar.selectbox("Select a feature", ["ParlaMind Insight", "ParlaMind Query", "ParlaMind Analytics"])

# Add dropdown for selecting the OpenAI model
model_name = st.sidebar.selectbox(
    "Select a GPT Model", 
    ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "gpt-4o"]
)



# Define page 1
def parlamind_insight():
    # Title
    st.title("📜 Welcome to ParlaMind Insight!")

    # Page Summary
    st.markdown(
        """
        <div style='text-align: justify; font-size: 18px;'>
            Curious about the key discussions in India's budget sessions? ParlaMind is here to help you stay informed and ask smarter questions!  
            Select a session below to view a quick summary of the discussions, and use the insights you gather to craft more targeted and relevant questions for ParlaMind's Query feature.  
            Whether you're curious about fiscal policies, government initiatives, or economic strategies, these summaries will give you the knowledge you need to ask better questions and get the answers you're looking for.  
            Ready to explore and ask informed questions? Let’s get started! 🎯
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Add spacing before the dropdown
    st.markdown("<br>", unsafe_allow_html=True)

    # Subheader for Video Selection
    st.markdown("### Select a video to explore key moments from India's budget discussions")

    # Create the dropdown (loading the video names only)
    selected_video = st.selectbox("Choose a session", df_summary["Video Name"])

    # Find the corresponding summary and link based on the selected video
    video_summary = df_summary[df_summary['Video Name'] == selected_video]['Summary'].values[0]
    video_link = df_summary[df_summary['Video Name'] == selected_video]['Video Link'].values[0]

    # Display the summary and a link to the video
    st.markdown(f"#### Summary of: {selected_video}")
    st.write(video_summary)
    st.markdown(f"🔗 [Watch the full video here]({video_link})")



# Define page 2
def parlamind_query():
    # Title
    st.title("🤖 Welcome to ParlaMind Query!")

    # Page Summary
    st.markdown(
        """
        <div style='text-align: justify; font-size: 18px;'>
            Here, you can ask questions about India's budget discussions, and we'll find the most relevant answers based on the video transcripts in our database.  
            Whether you're interested in fiscal policy, government programs, or specific statements, just type in your question, and ParlaMind will do the rest!  
            Ready to ask your question? Let's get started! 🤔💬
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Add spacing before the question input
    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize session state variables for question and answer
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ''
    if 'answer' not in st.session_state:
        st.session_state.answer = ''
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # Input field for user's question
    user_question = st.text_input("Ask your question about India's budget discussions:", st.session_state.user_question)

    # Add checkbox for question enhancement
    enhance_question = st.checkbox("Enhance my question for better results (Optional)")

    # Checkbox to ask user if they want to rerank the matches
    rerank_matches = st.checkbox("Rerank the search results for a better response (Optional)")

    # Initialize video_timestamps to None to avoid the UnboundLocalError
    video_timestamps = None

    if st.button("Submit"):
        st.session_state.user_question = user_question
        if user_question:
            st.session_state.submitted = True  # Mark as submitted
            # If the user opted to enhance the question
            if enhance_question:
                st.write("Enhancing your question...")
                enhanced_question = query_rewriting(summary, user_question, model_name)
                user_question = enhanced_question  # Use the enhanced question for further processing
                # Display the enhanced question
                st.markdown("### Enhanced Question:")
                st.write(enhanced_question)
            else:
                # Display the original question
                st.markdown("### Question:")
                st.write(user_question)

            # Encode the user's (or enhanced) question into a vector
            question_vector = model.encode([user_question])

            # Search the FAISS index
            k = 5  # Number of top results to retrieve
            result = index.search(question_vector, k)[1][0]

            # Retrieve the most relevant transcripts
            matches = [df_rag['RAG Text'][result_index] for result_index in result]

            # Rerank matches if the user opted for it
            if rerank_matches:
                st.write("Reranking the results...")
                matches = rerank(df_rag, user_question, model_name)
                # Do not show video references when reranking is applied
                st.write("Note: Video links are not available for reranked results.")
            else:
                # Retrieve video references
                video_timestamps = [(df_rag['Video Link'][result_index], df_rag['Start Time'][result_index]) for result_index in result]

            # Generate LLM answer based on the matched context
            st.session_state.answer = llm_requests(prompt(user_question, '\n'.join(matches)), client, model_name).choices[0].message.content

    # Check if the question was submitted
    if st.session_state.submitted:
        # Display the answer
        st.markdown("### Answer:")
        st.write(st.session_state.answer)

        # Display video references if not reranked
        if video_timestamps:
            st.markdown("### Video References:")
            video_links = ', '.join([f"[here]({video_link}&t={int(start_time)})" for video_link, start_time in video_timestamps])
            st.markdown(f"See the video at relevant moments {video_links}.")

        # Collect user feedback
        st.write("### Feedback on the Answer")

        # Rating on a scale of 1 to 5
        rating = st.slider("Rate the quality of the answer (1: Poor, 5: Excellent)", 1, 5, 0)

        # Text input for additional feedback
        feedback = st.text_area("Additional feedback (optional)")

        # Submit button for feedback
        if st.button("Submit Feedback"):
            # Prepare feedback data
            feedback_data = {
                'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'Question': [st.session_state.user_question],
                'Answer': [st.session_state.answer],
                'Rating': [rating if rating > 0 else None],  # Only store if rating is provided
                'Feedback': [feedback.strip() if feedback.strip() else None],  # Only store if feedback is provided
                'LLM Model Used': [model_name]  # Automatically store the selected model name
            }
                    
            feedback_df = pd.DataFrame(feedback_data)

            # Append or create feedback CSV
            feedback_file = 'feedback.csv'
            if os.path.exists(feedback_file):
                feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(feedback_file, mode='w', header=True, index=False)

            # Display thank-you message only if rating or feedback is provided
            if rating > 0 or feedback.strip():
                st.write("Thank you! Your feedback has been saved.")

    elif not st.session_state.submitted and user_question == '':
        st.write("Please enter a question before submitting.")



# Define page 3
def parlamind_analytics():
    st.title("📊 Welcome to ParlaMind Analytics!")

    # Add a summary at the top
    st.markdown(
        """
        Here, you can track the performance and interactions of users with the system.  
        The analytics includes:
        - A detailed history of the questions asked and the corresponding answers provided by the system.
        - An exploration of the distribution of user ratings for the quality of responses.
        - An analysis of how many questions were asked on a day-to-day basis.
        - A tracking of the average daily ratings to monitor feedback quality.
        - A comparison of model performance based on average ratings per model.
        
        Use these insights to evaluate how effectively ParlaMind is answering questions and gathering valuable feedback!
        """
    )

    # Add a separator for cleaner look
    st.divider()

    # Load feedback data
    feedback_file = 'feedback.csv'
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        feedback_df['Timestamp'] = pd.to_datetime(feedback_df['Timestamp'])

        # Ensure 'Rating' column is numeric, ignoring non-numeric entries
        feedback_df['Rating'] = pd.to_numeric(feedback_df['Rating'], errors='coerce')

        # Chart 1: History of questions and corresponding answers (occupying entire row/column)
        st.markdown("<h4>Review the History of Questions and Answers</h4>", unsafe_allow_html=True)
        history_table = feedback_df[['Timestamp', 'Question', 'Answer']].copy()
        history_table.columns = ['Timestamp', 'Question', 'Answer']
        st.dataframe(history_table)  # Display the table of questions and answers

        # Create two columns for the other charts
        col1, col2 = st.columns(2)

        # Chart 2: Rating distribution
        with col1:
            st.markdown("<h4>Explore the Distribution of User Ratings</h4>", unsafe_allow_html=True)
            rating_counts = feedback_df['Rating'].value_counts(dropna=True).reset_index()
            rating_counts.columns = ['Rating', 'Count']
            rating_bar_chart = alt.Chart(rating_counts).mark_bar().encode(
                x='Rating:O',
                y='Count:Q',
                tooltip=['Rating', 'Count']
            ).properties(width=350, height=400)
            st.altair_chart(rating_bar_chart)

        # Chart 3: Day-on-day number of questions answered
        with col2:
            st.markdown("<h4>Analyse the Number of Questions Asked Day by Day</h4>", unsafe_allow_html=True)
            feedback_df['Date'] = feedback_df['Timestamp'].dt.date
            daily_questions = feedback_df.groupby('Date').size().reset_index(name='Questions Answered')
            question_trend_chart = alt.Chart(daily_questions).mark_line(point=True).encode(
                x='Date:T',
                y='Questions Answered:Q',
                tooltip=['Date', 'Questions Answered']
            ).properties(width=350, height=400)
            st.altair_chart(question_trend_chart)

        # Chart 4: Day-on-day average rating (side by side in column 1)
        with col1:
            st.markdown("<h4>Track the Daily Average Rating of Responses</h4>", unsafe_allow_html=True)
            daily_avg_rating = feedback_df.groupby('Date')['Rating'].mean().reset_index(name='Average Rating')
            avg_rating_chart = alt.Chart(daily_avg_rating).mark_line(point=True).encode(
                x='Date:T',
                y='Average Rating:Q',
                tooltip=['Date', 'Average Rating']
            ).properties(width=350, height=400)
            st.altair_chart(avg_rating_chart)

        # Chart 5: Model Name-wise Average Rating (side by side in column 2)
        with col2:
            st.markdown("<h4>Compare Model Performance: Average Ratings by Model</h4>", unsafe_allow_html=True)
            model_avg_rating = feedback_df.groupby('LLM Model Used')['Rating'].mean().reset_index(name='Average Rating')
            model_avg_rating_chart = alt.Chart(model_avg_rating).mark_bar().encode(
                x='LLM Model Used:N',  # Model names on x-axis
                y='Average Rating:Q',  # Average ratings on y-axis
                tooltip=['LLM Model Used', 'Average Rating']  # Display model name and average rating on hover
            ).properties(width=350, height=400)
            st.altair_chart(model_avg_rating_chart)

    else:
        st.write("No feedback data available yet.")



# Display the appropriate page based on user selection
if page == "ParlaMind Insight":
    parlamind_insight()
elif page == "ParlaMind Query":
    parlamind_query()
elif page == "ParlaMind Analytics":
    parlamind_analytics()

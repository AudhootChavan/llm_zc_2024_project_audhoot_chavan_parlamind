# ParlaMind: Harnessing LLMs to summarize and answer questions from videos of the Indian Parliament discussions with an end to end RAG system

This project is created by me for the [LLM Zoomcamp 2024](https://github.com/DataTalksClub/llm-zoomcamp).

## Contents
- [Problem statement and project description](#problem-statement-and-project-description)
- [Technologies, tools and data sources used](#technologies-tools-and-data-sources-used)
- [Project flow diagram](#project-flow-diagram)
- [Project flow explanation](#project-flow-explanation)
- [How to replicate](#how-to-replicate)
- [Application demo and features](#application-demo-and-features)
- [Scope for improvement](#scope-for-improvement)
- [ChatGPT experience](#chatgpt-experience)
- [Reviewing criteria](#reviewing-criteria)

## Problem statement and project description

As I'm getting older I'm finding a growing interest in me to follow Politics or atleast be up to date on what is happening in the world of Indian Politics. It is also from a realization that the policies and reforms the Government puts in place have a direct impact on my personal and professional life for the next 5 years (term) or potentially even for the next few decades. I spend time occasionally watching speeches, podcasts, interviews and parliament discussions. Some of these mediums are very lengthy and it is difficult to find time to watch or listen to them in entirety. While watching one of these videos recently, I realized this is a great use case for an LLM to quickly summarise key points and provide actionable insights. So I decided I will select this topic for my LLM zoomcamp project.  
  
As this is a course project, my intention is not to build a full scale product but a generic prototype that can be scaled by whoever interested. For that I'm working only with 15 videos(Each about an hour long) from the Budget sessions 2024 of the Indian Parliament. I'll be creating a knowledge base from those videos using youtube transcripts and build a RAG system on top using Streamlit and OpenAI.  
  
The project/app will help political enthusiasts browse through video summaries and ask relevant questions to get a better understanding of governance and politics in the country. I think this project has a lot of potential for a full scale app and can be a great inspiration for political parties around the world to use during their campaigns as a one stop for increasing engagement with citizens and spread awareness about the initiatives taken, fundamental principles, vision for the future etc. It is a great way to leverage the massive data they would have in the form of speeches/podcasts/debates/promo videos/articles/open letters etc.  
  
## Technologies, tools and data sources used

- [Sansad TV](https://www.youtube.com/channel/UCISgnSNwqQ2i8lhCun3KtQg) YouTube channel - For videos of Parliament discussions.
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) - For downloading the transcripts from videos. 
- [Python](https://www.anaconda.com/download) - To build the project pipeline.
- [Jupyter Notebook](https://jupyter.org/) - For app preparations and evaluations. 
- [Sentence Transformers](https://sbert.net/) - For creating vector embeddings.
- [FAISS](https://ai.meta.com/tools/faiss/) - To perform a vector similarity search and build indices.
- [OpenAI API](https://platform.openai.com/) - To use as LLM and make requests. 
- [Docker](https://www.docker.com/) - For containerization of the pipeline and application.
- [Prefect](https://www.prefect.io/) - Orchestration tool for deployments and monitoring tasks and flows.
- [Streamlit](https://streamlit.io/) - To create a UI. 

## Project flow diagram

## Project flow explanation
The project has 2 parts. Part 1 consists of steps taken to prepare everything we need for our RAG system/application. Part 2 is the application where we put everything together we prepared in part 1.  
  
Part 1 is run entirely on Jupyter notebooks. It consists of the 3 following folders :  
- 1 - Prepare Data For RAG - In this section we will process raw text files and prepare data suitable for our RAG system. The code is in *1 - Prepare Data For RAG.ipynb* notebook. *video_data_to_extract.xlsx* is an excel file manually prepared that consists of a list of videos selected for this project along with their YouTube links. Using the youtube transcript api, english translated transcripts are downloaded for each video and stored in text files in the *data_dumps* folder. The next step is to chunk the transcripts. I tried using ChatGPT for this however I did not have much success with it. It was giving me granular sentences which would not have been useful to capture the entire context for the RAG. So I decided to manually chunk the texts. I did spend a decent amount of time on this and was honestly the most frustrating part of this project. In the future with more time in hand, I'd experiment with audio processing and LLMs together for automated chunking. The chunked text files are in *chunked_data* folder. The final step is to prepare a dataset called *prepared_data.xlsx* that we can use directly for our RAG. We are capturing the video series, video name, video link, chunk start time(We can use this later in our app as a feature - More on this later), chunk text and an additional modified field created by concatenating chunk text with video series. This will be done to help with better retrieval and more context for the RAG system.
- 2 - Retrieval And RAG Evaluations - In this section we will evaluate our retrieval and RAG mechanisms. The code is in *2 - Evaluations.ipynb* notebook. This will help with retrieving more relevant chunks and provide quality responses.
    - Retrieval evaluations - We start by creating a ground truth data set. As we are using OpenAI, we have limited credits and there are [rate limits](https://platform.openai.com/docs/guides/rate-limits) in place. So the best solution is to work on a random sample.  
    For the retrieval evaluation we will work with a sample of 50 records from the dataset we prepared in section 1 and will save the file as *evaluations_data.xlsx*. We then use OpenAI single request to create 5 questions for each chunk in our sample. The prompt to use for generating the questions can be constructed using ChatGPT and I would recommend that as well. The generated response needs some parsing to extract each question and then we save the dataset as *ground_truth_data.xlsx*. This will be our ground truth dataset on which we perform our retrieval evaluations.  
    For the evaluations we will use 2 metrics - [Hit rate and MRR](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems) and **2** approaches. 
        - The **first** approach is to evaluate which text field makes more sense to use for vector search. The original transcript text/chunk or the modified field that consists of video series as well for additional context. For this test we will *use paraphrase-albert-small-v2* sentence transformer model, smallest available to create vector embeddings. We will use K=3 nearest neighbors for *critical* evaluation. All indices are saved in the *indices* folder.
        Results - 
            ```
            Original Text
            Index dimensions: 768
            Number of vectors in the Index: 50
            Hit Rate : 0.744
            MRR : 0.6693333333333333
            ```


            ```
            RAG Text (Modified)
            Index dimensions: 768
            Number of vectors in the Index: 50
            Hit Rate : 0.74
            MRR : 0.6626666666666666
            ```
            Conclusion - My assumption was that the modified text field would give much better performance. But the data shows both have similar performance. We will however use the modified text field for better context for our RAG system.
        - The **second** approach is to try multiple sentence transformer [models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) and select the one that gives best performance. We will test for :
            - [*paraphrase-albert-small-v2*](https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2) - Listed as smallest.  
            - [*all-mpnet-base-v2*](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) - Listed as best performance. 
            - [*paraphrase-MiniLM-L3-v2*](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2) - Listed as fastest.  
            
            Results -  
            ``` 
            paraphrase-albert-small-v2
            Index dimensions: 768
            Number of vectors in the Index: 50
            Hit Rate : 0.74
            MRR : 0.6626666666666666 
            ```  

            ``` 
            all-mpnet-base-v2
            Index dimensions: 768
            Number of vectors in the Index: 50
            Hit Rate : 0.924
            MRR : 0.8633333333333335 
            ```


            ```
            paraphrase-MiniLM-L3-v2
            Index dimensions: 384
            Number of vectors in the Index: 50
            Hit Rate : 0.796
            MRR : 0.7306666666666668
            ```  
            Conclusion - As expected the model listed as best performance has the best hit rate. Great jump of upto ~20%. 
    - RAG evaluations - We start the RAG evaluations by creating a sample from our ground truth dataset. I have taken a random sample of 10 questions. We will use **2** approaches to evaluate our RAG. The first being 2 variations of prompts and the other being 2 different values of K. We will use an LLM for evaluating the quality of response our RAG generates on a scale of 1 to 5 of relevance with 5 being the best. The prompt for evaluation is generated with ChatGPT. 
        - The **first** approach is to create 2 prompt variations for Q & A. This prompt is central to our RAG system as it will generate answers to questions a user inputs to the system. The prompt variations are created with the help of ChatGPT. The first one is more reliable with instructions to search the web and cross reference any stats mentioned in the texts/video transcripts. The second one is more balanced and has instructions to use more intuition as well. We will use the same index we created during the retrieval evaluations stage, *all-mpnet-base-v2_index_option_2* in the *indices* folder.  
        - The **second** approach is to use K=3 and K=5 nearest neighbors and evaluate if more context yields a better quality response. 
        - Both approaches are run in a single loop. The evaluation needs to be done on a numerical scale however OpenAI gpt models often respond with additional info/text data. To solve for this we have used regex. The final outputs are saved in a dataset called *df_rag_evaluation_results.xlsx*.  
        Results - 
        ```
                    sum	mean
        Combination		
        3 | Prompt 1	35	3.5
        3 | Prompt 2	35	3.5
        5 | Prompt 1	34	3.4
        5 | Prompt 2	35	3.5
        ```
        Conclusion - Both prompts are similar in performance. Prompt 2 has 1 point higher so we will use that one. For K, we will use 5 as it will give our RAG system more context to work on. 
- 3 - Preparing Materials For App - In this section we will make final preparations for our app. The code is in *3 - Preparing Materials For App* notebook.
    - 1 - Create summaries of the videos to use in the app - Here we will summarise the entire text/transcript in each video. I plan to use these summaries in my application to let the user browse through so that he/she can ask relevant questions to the RAG. The texts in each video however can often breach the token limit set for the models. So to solve for this we can easily split the texts in to 2 sections, create a summary for each and then create a final summary from those 2. These can be achieved using well constructed prompts for both. I used ChatGPT for this task. The list of summaries are saved in a dataset called *summaries.xlsx*. 
    - 2 - Query rewriting - Here we implement a function to rewrite the query/question the user submits to the RAG. There are many approaches that can be taken but the best that came to my mind was to feed the context of our knowledge base to an LLM along with the query/question from the user and let the LLM reframe it. I created a summary of all summaries we generated in the previous section as context using a prompt constructed by ChatGPT. The query rewriting prompt is also generated using ChatGPT. 
    - 3 - Reranking with LLM - Finally we implement a function to rerank our retrieval search results. In this function, Given a user query/question I retrieve 10 nearest neighbors and let the LLM select top 5 relevant to the question.  

Part 2 of the project is the application where we put together everything we prepared in part 1. The codes are in the *application* folder. The pipeline and application is written in python and is run entirely inside a docker container. The docker image is created using a python base image *python:3.9-slim*. Refer to the Dockerfile for more details and requirements.txt for the python libraries along with their versions used/installed. The 2 main files here are *pipeline.py* and *application.py*.
- *pipeline.py* - This is the main script we use to run the application. First it will prepare the knowledge base of vector embeddings in FAISS and save the index locally. It will also save the sentence transfomer model locally to optimize for runtime. Lastly it will run the streamlit app via a subprocess. Additionally, prefect library/prefect cloud is used for orchestration to create flows/tasks and pipeline deployments to monitor runs. The prefect server is running in the cloud and the prefect agent is running in the docker container. Both parts mentioned above are implemented in prefect flows/tasks. 
- *application.py* - This is our UI that the user will interact with and is built with Streamlit. The UI has 3 pages - Summary page, Q & A page and an analytics page to monitor user feedback. More on this will be discussed in the application demo and features section.  

## How to replicate

### Step 0 - Get the following before starting to replicate

- Create an OpenAI key - https://platform.openai.com/api-keys.
- Create a prefect cloud account, create a workspace and an API key - https://orion-docs.prefect.io/latest/cloud/users/api-keys/#create-an-api-key.
- Install Python/Anaconda - https://www.anaconda.com/download.
- Install Docker - https://docs.docker.com/desktop/.

### Step 1 - Running the Jupyter notebooks 

- You only need to put your own OpenAI key in the OpenAI Key.txt file of the parent folder to run all the Jupyter notebooks. 

### Step 2 - Set up and build the pipeline in Prefect

- Download the entire folder/clone the repo. Navigate inside the application folder. 
- Put your OpenAI key in the OpenAI Key.text file of the application folder. 
- Keep your prefect key handy to login within docker shortly.
- Start Docker desktop and open terminal in your current directory(*application* folder).
- Build Docker image. 
    ```
    docker build -t parlamind-app .
    ```
- Create container from image/start the container.
    ```
    docker run -it -p 8501:8501 --name parlamind parlamind-app
    ```
    The command above is to be used the first time. Going forward you can use the following :
    ```
    docker start -i parlamind
    ```
- Install the dependencies. I have kept this outside of the Dockerfile as it takes a while for me. If your internet is fast you can put it in the Dockerfile itself. 
    ```
    pip install -r requirements.txt
    ```
- Login to Prefect. Use your own key.  
    ```
    prefect cloud login -k prefect_key
    ```
- Build the deployment to make runs via cloud. 
    ```
    python pipeline.py
    ```

### Step 3 - Run the pipeline
- Go to prefect cloud to manage your runs. Your deployment ‘main_flow’ should be created in the deployments section. Do a quick run and monitor tasks on the Prefect dashboard. 

### Step 4(Optional) - Delete project
- Stop prefect agent with Control C.
- Exit from the container.
    ```
    exit
    ```
- Delete container.
    ```
    docker rm parlamind
    ```
- Delete image.
    ```
    docker image parlamind-app
    ```

## Application demo and features

ParlaMind demo and screenshots
  
Features - The app has 3 features.
- 1 - ParlaMind Insight - This is the summary page where a user can browse through summaries of all the videos used in the knowledge base. The objective of creating this page is for the user to ask relevant questions to the RAG. A drop down is implemented so the user can read summaries he/she is interested in. There is also a video link if the user is interested in watching the respective video.  

- 2 - ParlaMind Query - This is the Q & A page which is our main RAG system. This is where the user will ask questions related to parliament discussions and our RAG system will retrieve relevant documents from the knowledge base and respond appropriately. The user also has an option to select the gpt model version he/she wants to run the RAG flow in. There are additionally 2 functions/features implemented for question enhancing(query rewriting) and reranking search results from retrieval. The user can check both or either to ask better questions and or get better responses. Additionally a user also gets video references at the bottom(Except for reranked results) to view exactly at what point in the video the question context was discussed. Finally the user can provide a rating of 1 to 5 for the quality of response received. He/she can also provide custom text feedback for the responses they receive. 

- 3 - ParlaMind Analytics - This is the final page where we visualise the feedback provided by the users. There are 5 charts in total. 
    - 1 - History of questions asked and answers received.
    - 2 - Distribution of user ratings on a scale of 1 to 5.
    - 3 - Day on day number of questions asked. 
    - 4 - Day on day average rating given by the users.
    - 5 - Average rating for each gpt model used. 

## Scope for improvement

- A better search engine can be used. In this project I have used FAISS which was simple to implement but only allows for vector search. For advanced functions like filtering, multi match search, hybrid search etc, Elasticsearch or other alternatives can be used. 
- Chunking - Even though I spent a lot of time manually chunking the video transcripts. The chunks can be enriched with meta data like name of the speaker, party of the speaker, their constituency etc. This can improve the RAG system and will be able to cover a lot more questions and also provide more nuanced answers. 
- LanceDB can be used along with FAISS. 
- The user feedback is stored in csv files. A database like PostgreSQL/SQLite can be used as well. 
- Finally the monitoring dashboard can be separate from the main app and can be built with Grafana as suggested in this course. 


## ChatGPT experience

I have been using the paid version of ChatGPT for a while but I truly realized its power while building this project. I used it for almost every aspect of this project and my experience has been superb. Some of the things I used ChatGPT for are as follows :
- Debugging. 
- Writing test cases and evaluating them.
- Writing content for the app. 
- Creating the logo.
- Reviewing documentation. 
- Writing code. Blocks of it. In fact the analytics page is 90% ChatGPT.  

Very excited for the future. We will defintely build applications while talking to a screen in the future like Iron Man does with JARVIS. 

## Reviewing criteria

* Problem description - *The project and problem statement is described in detail [here](#problem-statement-and-project-description).*
    * 0 points: The problem is not described
    * 1 point: The problem is described but briefly or unclearly
    * 2 points: The problem is well-described and it's clear what problem the project solves
* RAG flow - *RAG flow is fundamental to this project and you can find multiple references in both the Jupyter notebooks and the application file. In the application, the code can be referenced between the lines 223-273 of the application.py file.*
    * 0 points: No knowledge base or LLM is used
    * 1 point: No knowledge base is used, and the LLM is queried directly
    * 2 points: Both a knowledge base and an LLM are used in the RAG flow 
* Retrieval evaluation - *You can find retrieval evaluation implementation in 2 - Retrieval And RAG Evaluations folder and 2 - Evaluations.ipynb notebook. **2** approaches are used.*
    * 0 points: No evaluation of retrieval is provided
    * 1 point: Only one retrieval approach is evaluated
    * 2 points: Multiple retrieval approaches are evaluated, and the best one is used
* RAG evaluation - *You can find RAG evaluation implementation in 2 - Retrieval And RAG Evaluations folder and 2 - Evaluations.ipynb notebook. **2** approaches are used.*
    * 0 points: No evaluation of RAG is provided
    * 1 point: Only one RAG approach (e.g., one prompt) is evaluated
    * 2 points: Multiple RAG approaches are evaluated, and the best one is used
* Interface - *The interface is built in Streamlit and the entire code is in application.py file.*
   * 0 points: No way to interact with the application at all
   * 1 point: Command line interface, a script, or a Jupyter notebook
   * 2 points: UI (e.g., Streamlit), web application (e.g., Django), or an API (e.g., built with FastAPI) 
* Ingestion pipeline - *Prefect is used for orchestration. You can see the tasks and flows implemented in pipeline.py file.*
   * 0 points: No ingestion
   * 1 point: Semi-automated ingestion of the dataset into the knowledge base, e.g., with a Jupyter notebook
   * 2 points: Automated ingestion with a Python script or a special tool (e.g., Mage, dlt, Airflow, Prefect)
* Monitoring - *User feedback is collected in the application.py file. Reference code between lines 287-322.*
   * 0 points: No monitoring
   * 1 point: User feedback is collected OR there's a monitoring dashboard
   * 2 points: User feedback is collected and there's a dashboard with at least 5 charts
* Containerization - *The application and pipeline is run inside the Docker container.*
    * 0 points: No containerization
    * 1 point: Dockerfile is provided for the main application OR there's a docker-compose for the dependencies only
    * 2 points: Everything is in docker-compose
* Reproducibility - *Detailed documentation provided along with a video demo/screenshots and with instructions on how to replicate. The dependencies and their versions are in the requirements.txt file*.
    * 0 points: No instructions on how to run the code, the data is missing, or it's unclear how to access it
    * 1 point: Some instructions are provided but are incomplete, OR instructions are clear and complete, the code works, but the data is missing
    * 2 points: Instructions are clear, the dataset is accessible, it's easy to run the code, and it works. The versions for all dependencies are specified.
* Best practices - *Document reranking and user query rewriting is implemented. Code reference in application.py file between lines 58-129 and 226-285. Unfortunately as I have used FAISS hybrid search could not be implemented.*
    * [ ] Hybrid search: combining both text and vector search (at least evaluating it) (1 point)
    * [ ] Document re-ranking (1 point)
    * [ ] User query rewriting (1 point)
* Bonus points (not covered in the course) - *Not deployed to cloud.*
    * [ ] Deployment to the cloud (2 points)

# Import libraries
import pandas as pd
import os
import numpy as np
import faiss   
from sentence_transformers import SentenceTransformer
# For prefect
import subprocess
from prefect import flow,task



# Prepare for streamlit app
@task(name="app_prep", log_prints=True)
def app_prep():
    # If index and model already available, skip this step and print message
    if os.path.exists('retrieval_index.index') and os.path.exists('saved_model'):
        print('Index already built...')
        print('Sentence transformer model already saved...')
    else:
        # Read the prepared data 
        df = pd.read_excel('prepared_data.xlsx')
        # Build index
        # Create dictionaries to store the text vector embeddings
        print('Creating dictionaries')
        text_dict = {i:'' for i in df['RAG Text']}
        # Load the best model as per retrieval evaluation - 'all-mpnet-base-v2'
        print('Loading model...')
        model = SentenceTransformer('all-mpnet-base-v2')
        # Create embeddings 
        print('Creating embeddings...')
        for text in text_dict.keys():
            text_dict[text] = model.encode(text)
        # Add all vectors together
        all_embeddings = np.vstack(list(text_dict.values()))
        # Get dimensions of the embeddings 
        d = all_embeddings.shape[1]
        # Create the FAISS index - IndexFlatL2 is best for small datasets as it guarantees an exact nearest neighbor search
        index = faiss.IndexFlatL2(d)
        # Add the combined embeddings to the index
        print('Building index...')
        index.add(all_embeddings)
        print('Number of vectors in the index: ' + str(index.ntotal))
        # Save the index
        print('Saving index...')
        faiss.write_index(index,'retrieval_index.index')
        # Save the sentence transformer model to a directory
        save_path = 'saved_model/'
        print('Saving sentence transformer model...')
        model.save(save_path)
    return

# Running streamlit app
@task(name="run_app", log_prints=True)
def run_app():
    try:
        # Run the Streamlit app using subprocess.run() with suppressed output
        # Run with -m to avoid issues
        result = subprocess.run('python -m streamlit run application.py', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Check if the process ran successfully
        if result.returncode == 0:
            print('Streamlit app started successfully...')
            return 
        else:
            print('Streamlit app failed with error code : ' + str(result.returncode))
            return
    except Exception as e:
        print('Error starting Streamlit app : ' + str(e))
        return 
    
# Main flow
@flow(name="main_flow", log_prints=True)
def main_flow():
    # Prepare for streamlit app
    print('Making app preparations...')
    prep_task = app_prep() 
    print('App preparations complete...')
    # Run the streamlit app - Wait for prep_task to complete. This is to ensure the tasks run in sequence. SequentialTaskRunner is not supported in new versions?
    print('Running the app...')
    run_task = run_app(wait_for=[prep_task])
    
# Serve the flow locally instead of deploying
if __name__ == "__main__":
    main_flow.serve()
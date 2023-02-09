from flask import Flask, render_template, url_for, request
import time

from youtube_channel_transcript_api import *
from elasticsearch import Elasticsearch
import os
import pickle
import hashlib
import mmh3
from typing import List, Dict, Optional, Generator, Set, Union
import logging
import numpy as np


class Document:
    def __init__(self, meta, hash_id, title:str, content:str, language:str = 'English', score:float = None, hash_id_keys:List[str] = None):
        self.title = title
        self.content = content
        self.language = language
        self.hash_id_keys = hash_id_keys
        self.meta = meta or {}
        self.embedding = None
        self.score = score

        if hash_id is None: 
            self.hash_id = self.generate_hash(hash_id_keys)
        else:
            self.hash_id = hash_id

    def generate_hash(self, hash_id_keys):
        return "{:02x}".format(mmh3.hash128(str(self.content), signed=False))

    def to_dict(self, field_map = {}):
        inv_field_map = {v:k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}

        for k, v in self.__dict__.items():
            # Exclude other fields (Pydantic, ..) fields from the conversion process
            if k.startswith("__"):
                continue
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        # print(f'_doc in to_dict is {_doc}')
        return _doc

    def __str__(self):
        return (f"Title: {self.title}\nContent: {self.content}\nLanguage: {self.language}\nHash ID: {self.hash_id} \nMetadata: {self.meta}")


from elasticsearch import Elasticsearch

es = Elasticsearch(
    "https://localhost:9200",
    ca_certs="D:\BTP\youtubeQandA\http_ca.crt",
    basic_auth=("elastic", 'vWr8xqxdlmOhj*Q_2yFI')
)

from sentence_transformers import SentenceTransformer

# good for passage search
sBERT = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

sBERT.max_seq_length = 512



app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route('/query', methods=['GET', 'POST'])
def query():
    # apikey = request.form['apikey']
    # playlist_id = request.form['playlist']
    
    API_KEY = request.form['apikey']
    PLAYLIST_ID = request.form['playlist']

    
    # getting playlist data
    channel_getter = YoutubePlaylistTranscripts("Some Gibberish Name",PLAYLIST_ID, API_KEY) #channel getter is a YoutubePlaylistTranscripts Object
    videos_data, videos_errored = channel_getter.get_transcripts(languages=['en'])
    print(f'Number of videos loaded: {len(channel_getter.video)}')
    print(f'Number of videos data fetched: {len(videos_data)}')
    print(f'Number of videos data errored: {len(videos_errored)}')

    # creating backup on storage
    with open(f'./content/{PLAYLIST_ID}_vids_data_processed.pkl', 'wb') as f:
        pickle.dump(videos_data, f)
    with open(f'./content/{PLAYLIST_ID}_vids_data_errored.pkl', 'wb') as f:
        pickle.dump(videos_errored, f)

    print(f'Number of videos loaded from playlist: {len(videos_data)}')
    print('List of loaded videos:')

    print(f'Number of videos loaded from playlist: {len(videos_data)}')
    print('List of non-loaded videos:')
    print(videos_errored)


    # creating caption database on storage
    ROOT_FOLDER = "./content/playlists"
    CHANNEL_DIRECTOR_NAME = PLAYLIST_ID

    SAVE_FOLDER = os.path.join(ROOT_FOLDER, CHANNEL_DIRECTOR_NAME)

    # storing video captions
    for vid_obj in videos_data.values():
        TITLE = vid_obj['title']
        #windows doesn't allow all the special characters to be there in the folder name
        # Let's remove the special characters from the title

        TITLE = TITLE.replace("?",'')   #windows doesn't support '?'
        TITLE = TITLE.replace("|",'')   #windows doesn't support '|'

        VID_FOLDER = os.path.join(SAVE_FOLDER, TITLE)
        # print(f'VID_FOLDER: {VID_FOLDER}')
        vid_exists = os.path.exists(VID_FOLDER)   # checking whether the video directory exists
        # print(f'vid_exists: {vid_exists}')
        os.makedirs(VID_FOLDER) if not vid_exists else None   # if the directory doesn't exist, create one

        vid_captions = vid_obj['captions'] 

        full_vid_captions = [f'Title: {TITLE}']  #This list will have all the captions in the video without the time stamps
        #The below code can be modified to include time
        for caption in vid_captions:
            full_vid_captions.append(caption['text'])   #full video captions is the list of caption strings

        full_vid_captions = " ".join(full_vid_captions)   # this returns a single string of complete video caption

        with open(os.path.join(VID_FOLDER, f'{TITLE}_captions.txt'), 'w') as f:
            f.write(full_vid_captions)


    def clean_document(text:str) -> str:
        # this function tries to clean the text by removing multiple new lines, adding paragraph breaks, and removing empty paragraphs

        # getting rid of all new lines
        while '\n' in text:
            text = text.replace('\n', '')

        # will add some features here in future
        return text


    class Document:
        def __init__(self, meta, hash_id, title:str, content:str, language:str = 'English', score:float = None, hash_id_keys:List[str] = None):
            self.title = title
            self.content = content
            self.language = language
            self.hash_id_keys = hash_id_keys
            self.meta = meta or {}
            self.embedding = None
            self.score = score

            if hash_id is None: 
                self.hash_id = self.generate_hash(hash_id_keys)
            else:
                self.hash_id = hash_id

        def generate_hash(self, hash_id_keys):
            return "{:02x}".format(mmh3.hash128(str(self.content), signed=False))

        def to_dict(self, field_map = {}):
            inv_field_map = {v:k for k, v in field_map.items()}
            _doc: Dict[str, str] = {}

            for k, v in self.__dict__.items():
                # Exclude other fields (Pydantic, ..) fields from the conversion process
                if k.startswith("__"):
                    continue
                k = k if k not in inv_field_map else inv_field_map[k]
                _doc[k] = v
            # print(f'_doc in to_dict is {_doc}')
            return _doc

        def __str__(self):
            return (f"Title: {self.title}\nContent: {self.content}\nLanguage: {self.language}\nHash ID: {self.hash_id} \nMetadata: {self.meta}")


    def split_documents(document:Document, split_length:int = 100):
        text = document.content

        line = ''
        text_chunks = []

        words = text.split(' ')[:-1]

        # print(words)

        for word in words:
            if len(line) >= split_length:
                text_chunks.append(line)
                line = ''

            else:
                line += ' ' + word
                

        documents = []
        for i, txt in enumerate(text_chunks):
            doc = Document(title = document.title, content = txt, hash_id = None, hash_id_keys=None, meta = {'filename': document.meta.copy()} or {})
            # I need to implement meta data here
            doc.meta["_split_id"] = i
            doc.meta["_parent_hash"] = document.hash_id
            documents.append(doc)
            
        
        return documents


    ############################ preprocessor ##################3
    next_folder = os.path.join(SAVE_FOLDER, os.listdir(SAVE_FOLDER)[0])
    file_path = os.path.join(next_folder, f'{os.listdir(SAVE_FOLDER)[0]}_captions.txt')

    parent_document = {}    # storing document objects with the hashid:object 
    document_list = []      # this list stores all the document objects
    split = True

    # crawler
    for folder in os.listdir(SAVE_FOLDER):
        # opening the files
        next_folder = os.path.join(SAVE_FOLDER, folder)
        file_path = os.path.join(next_folder, f'{folder}_captions.txt')
        

        try:
            f = open(file_path, 'r')
        except:
            logging.error(f"The file {file_path} cannot be opened.")
        
        # creating document object 
        content = f.read()
        obj = Document(title = folder, content = content, meta = {'file_name': f'{folder}_captions.txt'} , hash_id = None, hash_id_keys = None)

        # cleaning the object content
        obj.content = clean_document(obj.content)

        # storing the content in the dictionary
        parent_document[obj.hash_id] = obj


        # if split is needed, we split else we directly append to the list
        if split:
            # split_document returns a list of document objects
            documents = split_documents(obj, split_length = 1000)


            # appending the list of document objects to our main list
            for d in documents:
                document_list.append(d)
            
        else:
            document_list.append(obj)

        

    ############# es connection and index creation ##########33
    from elasticsearch import Elasticsearch

    es = Elasticsearch(
        "https://localhost:9200",
        ca_certs="D:\BTP\youtubeQandA\http_ca.crt",
        basic_auth=("elastic", 'vWr8xqxdlmOhj*Q_2yFI')
    )

    if es.ping():
        print("Connected to ES!")
    else:
        print("Could not connect!")


    from elasticsearch.helpers import bulk
    from elasticsearch import Elasticsearch
    import numpy as np

    def bulk_index_documents(documents_to_index, request_timeout = 300, refresh = 'wait_for'):
        try:
            bulk(es, documents_to_index, request_timeout = request_timeout, refresh = refresh)
        except Exception as e:
            logging.error(f"Unable to index batch of {len(documents_to_index)} documents because of too many request response")


    def create_mappings_document(
        index_name = 'document',
        analyzer = 'standard',
        custom_mapping = None,
        name_field = "title",
        content_field = "content",
        embedding_field = 'embedding',
        embedding_dim = 768,
        search_fields = ['content'],
        synonyms = None,
        synonym_type = 'synonym'):
        

        if custom_mapping:
            mapping = custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {name_field : {"type" : "keyword"}, content_field: {"type": "text"}},
                    "dynamic_templates": [
                        {"strings": {"path_match": "*", "match_mapping_type": "string", "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {"analysis": {"analyzer": {"default": {"type": analyzer}}}},
            }

        if synonyms:
            for field in search_fields:
                mapping["mappings"]["properties"].update({field: {"type": "text", "analyzer": "synonym"}})
            mapping["mappings"]["properties"][content_field] = {"type": "text", "analyzer": "synonym"}

            mapping["settings"]["analysis"]["analyzer"]["synonym"] = {
                "tokenizer": "whitespace",
                "filter": ["lowercase", "synonym"],
            }

            mapping["settings"]["analysis"]["filter"] = {
                "synonym": {"type": synonym_type, "synonyms": synonyms}
            }

        else:
            for field in search_fields:
                mapping["mappings"]["properties"].update({field: {"type": "text"}})

        if embedding_field:
            mapping["mappings"]["properties"][embedding_field] = {
                    "type": "dense_vector",
                    "dims": embedding_dim
            }

        es.indices.create(index = index_name, ignore = 400, body = mapping)


    def index_documents(documents, index = 'document', batch_size = 100, refresh_type = 'wait_for'):
        if index and not es.indices.exists(index= index):
            logging.info('Creating mappings for the index as user did not provide any custom mapping...')
            create_mappings_document(index_name = index)

        else:
            logging.info('Using custom mapping...')

        documents_to_index = []

        # Iterating through all the documents and indexing them together
        for i, doc in enumerate(documents):
            
            
            # First we convert the document object into dict to follow ES conventions
            _doc = {
                "_op_type": "index",
                "_index": index,
                **doc.to_dict()
            }
            # print(_doc)

            _doc["_id"] = str(i)

            # Cast the embedding type as ES does not support numpy
            if _doc['embedding'] is not None:
                if (type(_doc['embedding']) == np.ndarray):
                    _doc['embedding'] = _doc['embedding'].tolist()

            # don't index query score and empty fields
            _ = _doc.pop("score", None)
            _doc = {k: v for k, v in _doc.items() if v is not None}

            # For flat structure generally used in Elastic Search
            # we 'unnest' all value within "meta"
            if "meta" in _doc.keys():
                for k, v in _doc["meta"].items():
                    _doc[k] = v
                _doc.pop("meta")

            documents_to_index.append(_doc)

            if len(documents_to_index) % batch_size == 0:
                logging.info(f'Indexing {len(documents_to_index)} documents')
                bulk_index_documents(documents_to_index, request_timeout=300, refresh = refresh_type)
                documents_to_index = []

            if documents_to_index:
                logging.info(f'Indexing {len(documents_to_index)} documents')
                bulk_index_documents(documents_to_index, request_timeout=300, refresh = refresh_type)

    def clear_indices(index_name = ''):
        
        if index_name == '':
            indices = list(es.indices.get_alias("*").keys())
        else:
            indices = list(es.indices.get_alias(index_name).keys())

        if indices:
            logging.info(f'Wiping out all the documents belonging to the following indices: {indices}')

            es.delete_by_query(index = indices, body = {"query": {"match_all": {}}})

            logging.info(f"Deleting Indices")
            for index in indices:
                es.indices.delete(index= index, ignore= [400, 404])

            print(list(es.indices.get_alias("*".keys())))

        else:
            logging.info("No Indices are present in storage")
            

        # Constructing the query for the default BM25 retriever

    def construct_query(query, top_k, filters = None, all_terms_must_match = False):

        # We choose if all terms must match or not (this can be used to enforce more strict rules)
        operator = "AND" if all_terms_must_match else "OR"

    # There are multiple options for keyword based serach such as:
    # -> match: It directly matches all the keywords in any order
    # -> match_phrase: It directly matches all the keywords in specific order( so that sentences are bound to make sense)
    # -> multi_match: It directly matches all the keywords in any order but can match on multiple fields

        body = {
            "size": str(top_k),
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content", "title"],
                                "operator": operator,
                            }
                        }
                    ]
                }
            }
        }

        return body


    # This constructs query for dense retrieval using various scores
    def construct_query_dense(query_vector, top_k, similarity = "cosinse"):

        if similarity == "cosine":
            similarity_fn_name = "cosineSimilarity"
        elif similarity == "dot_product":
            similarity_fn_name = "dotProduct"
        elif similarity == "12":
            similarity_fn_name = "12norm"
        else:
            raise Exception(
                "Invalid value for similarity in ElasticSearchDocumentStore\nChoose between 'cosine', 'dot_product', and '12"
            )
        
        logging.info(f'Using the following similarity metric : {similarity}')

        if (type(query_vector) == np.ndarray):
            query_vector = query_vector.tolist()
        
        logging.info(f'The type of query vector is: {type(query_vector)}')

        body = {
            "size": str(top_k),
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": f"{similarity_fn_name}(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

        return body


    # This function processes the hit results obtained after elastic search
    def convert_es_dict(es_dict, return_embedding = False, scale_score = None):

        meta_data = {k : v for k,v in es_dict['_source'].items() if k not in ('title', 'content', 'language', 'hash_id', 'embedding')}

        # calculate score if using embedding retreival
        score = es_dict['_score']

        # check if name field is present or not
        if es_dict['_source']['title'] is not None:
            title = es_dict['_source']['title']

        document = Document(title = title, content = es_dict['_source']['content'], language = es_dict['_source']['language'], meta = meta_data, score = score, hash_id = es_dict['_source']['hash_id'])
        return document


    ################## creating dense index ################33




    # data_path = './data/tutorial'
    documents_dense, parent_dense = document_list, parent_document

    from sentence_transformers import SentenceTransformer

    # good for passage search
    sBERT = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

    sBERT.max_seq_length = 512

    encoded_data = sBERT.encode([doc.content for doc in documents_dense[0:10]])

    for i, doc in enumerate(documents_dense[0:10]):
        doc.embedding = encoded_data[i]

    index_documents(documents_dense[0:10], index = 'document_dense', batch_size = 1000)


    ### final return statement###
    return render_template('query.html', api= API_KEY, pid = PLAYLIST_ID)



@app.route('/result', methods=['GET', 'POST'])
def result():
    query = request.form['query']
    title = 'hi'

    ############ addding main code ########################

    


    # Constructing the query for the default BM25 retriever

    def construct_query(query, top_k, filters = None, all_terms_must_match = False):

        # We choose if all terms must match or not (this can be used to enforce more strict rules)
        operator = "AND" if all_terms_must_match else "OR"

    # There are multiple options for keyword based serach such as:
    # -> match: It directly matches all the keywords in any order
    # -> match_phrase: It directly matches all the keywords in specific order( so that sentences are bound to make sense)
    # -> multi_match: It directly matches all the keywords in any order but can match on multiple fields

        body = {
            "size": str(top_k),
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content", "title"],
                                "operator": operator,
                            }
                        }
                    ]
                }
            }
        }

        return body


    # This constructs query for dense retrieval using various scores
    def construct_query_dense(query_vector, top_k, similarity = "cosinse"):

        if similarity == "cosine":
            similarity_fn_name = "cosineSimilarity"
        elif similarity == "dot_product":
            similarity_fn_name = "dotProduct"
        elif similarity == "12":
            similarity_fn_name = "12norm"
        else:
            raise Exception(
                "Invalid value for similarity in ElasticSearchDocumentStore\nChoose between 'cosine', 'dot_product', and '12"
            )
        
        logging.info(f'Using the following similarity metric : {similarity}')

        if (type(query_vector) == np.ndarray):
            query_vector = query_vector.tolist()
        
        logging.info(f'The type of query vector is: {type(query_vector)}')

        body = {
            "size": str(top_k),
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": f"{similarity_fn_name}(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

        return body


    # This function processes the hit results obtained after elastic search
    def convert_es_dict(es_dict, return_embedding = False, scale_score = None):

        meta_data = {k : v for k,v in es_dict['_source'].items() if k not in ('title', 'content', 'language', 'hash_id', 'embedding')}

        # calculate score if using embedding retreival
        score = es_dict['_score']

        # check if name field is present or not
        if es_dict['_source']['title'] is not None:
            title = es_dict['_source']['title']

        document = Document(title = title, content = es_dict['_source']['content'], language = es_dict['_source']['language'], meta = meta_data, score = score, hash_id = es_dict['_source']['hash_id'])
        return document

    
    # It is better to define a reader for better results

    from transformers import BertForQuestionAnswering, AutoTokenizer, RobertaForQuestionAnswering

    modelname = 'deepset/roberta-base-squad2'

    reader = RobertaForQuestionAnswering.from_pretrained(modelname)
    reader_tokenizer = AutoTokenizer.from_pretrained(modelname)

    from transformers import pipeline
    nlp = pipeline('question-answering', model = reader, tokenizer = reader_tokenizer)

    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    import time
    import os

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


    def fetch_info(doc):
        meta_dict = dict()
        meta_dict['score'] = doc.score
        meta_dict['title'] = doc.title
        meta_dict['content'] = doc.content
        meta_dict['meta'] = doc.meta
        meta_dict['hash_id'] = doc.hash_id
        
        return meta_dict


    def generate_answer(question, context):
        output = nlp({
            'question' : question,
            'context' : context,
        })

        return output


    def process_documents(query, results):
        answers = []

        for res in results:
            output = generate_answer(query, res['content'])
            answers.append((output, res))

        # Sort the answers according to the scores
        sorted_answers = sorted(answers, key=lambda item: item[1].get('score'), reverse=True)
        return sorted_answers

    def search(query, top_k, index, model, extract_answers:bool = True, all_terms_must_match:bool = True, combine:bool = True):

        ### BM25 serach (lexical search) ###
        t = time.time()

        body = construct_query(query, top_k = top_k, all_terms_must_match= all_terms_must_match)
        # print(f'body is {body}')

        result = es.search(body = body, index = index)["hits"]["hits"]
        # print(f'results is {result}')
        documents_lexical = [
            convert_es_dict(hit, scale_score= None) for hit in result
        ]
        # print(documents_lexical[0])
        documents_lexical = [fetch_info(doc) for doc in documents_lexical]
        documents_lexical = process_documents(query, documents_lexical)

        print('Top-3 lexical search (BM25) hits ')

        for document in documents_lexical[0:3]:
            print("\n###################################")

            if extract_answers == True:
                answers, result = document
                answer = answers['answer']
                print('Answer : ', answer)

            else:
                result = document

        print('\nRetrieval Score : ', result['score'])
        print('Wiki Title : ', result['title'])
        print(result['content'])
        print('\nOriginal Document meta data : ', result['meta'])

        print('#######################')
        print()

        print('BM25 Results took a total of : {} seconds.'.format(time.time()-t))

        #### SBERT Search (Semantic Search) #######

        t = time.time()

        query_vector = sBERT.encode([query])[0].tolist()
        body = construct_query_dense(query_vector, top_k= top_k, similarity = 'cosine')

        result = es.search(body = body, index = index)['hits']['hits']

        documents_semantic = [
            convert_es_dict(hit, scale_score = None)
            for hit in result
        ]

        documents_semantic = [fetch_info(doc) for doc in documents_semantic]
        documents_semantic = process_documents(query, documents_semantic)

        print("Top-3 semantic search (SBERT) hits")

        for document in documents_semantic[0:3]:
            print('\n###############')

            if extract_answers == True:
                answers, result = document
                answer = answers['answer']
                print('Answer : ', answer)

            else:
                result = document

            print('\nRetrieval Score : ', result['score'])
            print('Wiki Title : ', result['title'])
            print(result['content'])
            print('\nOriginal Document meta data : ', result['meta'])

            print('#######################')
            print()

        print('SBERT Results took a total of : {} seconds.'.format(time.time()-t))

        #### Re-Ranking ######

        t = time.time()

        if combine:
            documents_extracted = (documents_lexical + documents_semantic)

            unique_hashes = set()
            documents = []

            for document in documents_extracted:
                hash_id = document[1]['hash_id']

                if hash_id not in unique_hashes:
                    unique_hashes.add(hash_id)
                    documents.append(document)

        else:
            documents = documents_semantic

        cross_inp = [[query, document[1]['content']] for document in documents]
        cross_scores = cross_encoder.predict(cross_inp)

        for idx in range(len(cross_scores)):
            documents[idx][1]['cross-score'] = cross_scores[idx]

        documents = sorted(documents, key = lambda item: item[1].get('cross-score'), reverse = True)

        print(f"Cross Encoder Re-Ranker Scoring of {len(documents)} documents")

        for document in documents[0:3]:
            print("\n#############################")

            if extract_answers == True:
                answers, result = document
                answer = answers['answer']
                print('Answer : ', answer)

            else:
                result = document

            print("Cross Score : ", result['cross-score'])
            print('Wiki Title : ', result['title'])
            print(result['content'])
            print('\nOriginal Document meta data : ', result['meta'])

            print('#######################')
            print()

        print('Cross Decoder Results took a total of : {} seconds.'.format(time.time()-t))

        return documents_lexical[0:3], documents_semantic[0:3] , documents[0:3]

        

    ###### long questions #########3
    extract_answers = True
    all_terms_must_match = False
    index = 'document_dense'
 
    print(f"You Searched : {query}\n")

    lexical, semantic, cross = search(query, top_k = 32, index = index, model= sBERT, extract_answers=extract_answers, all_terms_must_match= all_terms_must_match, combine = True)

    return render_template('result.html', lexical=lexical, semantic=semantic, cross=cross)



if __name__ == '__main__':
    app.run(debug=True)

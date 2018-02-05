from dictionary import MonolingualDictionary
from ir_data_reader import TimeReader
from search_engine import SearchEngine


time_dir = '/Users/xx/Documents/school/kth/thesis/ir-datasets/time/'
doc_file = time_dir + 'TIME.ALL'
query_file = time_dir + 'TIME.QUE'
relevance_file = time_dir + 'TIME.REL'
embed_file = '/Users/xx/Downloads/MUSE-master/trained/vectors-en.txt'

documents, queries, relevance = TimeReader().read(documents=doc_file, queries=query_file, relevance=relevance_file)

mono_dict = MonolingualDictionary(emb_file=embed_file)
search_engine = SearchEngine(dictionary=mono_dict)

search_engine.index_documents(documents.values())
n_results = 10
correct = 0
for i, query in queries.items():
    print()
    print(i, query, relevance[i])
    results = search_engine.query_index(query, n_results=n_results)
    for distance, result in results:
        result_i = list(documents.keys())[list(documents.values()).index(result)]
        print(result_i, distance, result[:300])
        if result_i in relevance[i]:
            print('Correct!')
            correct += 1
print(correct, len(queries)*n_results)

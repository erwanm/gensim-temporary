import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from pprint import pprint
from gensim import corpora, models

import os
import sys, getopt
import random
from collections import defaultdict



PROG_NAME = "gensim-topics.py"

model_type = "lda"
ncores=1
ntopics=10
nworkers=64

def usage(out):
    print("Usage: "+PROG_NAME+" [options] <corpus prefix> <output>",file=out)
    print("",file=out)
#    print("  <input data dir> contains subdirectories, one per document.",file=out)
    print("",file=out)
    print("  Options:")
    print("    -h: print this help message.",file=out)
    print("    -m <model type>. Possible values: lda, ens, lsi. Default: "+model_type,file=out)
    print("    -c <cores>. Default: "+str(ncores),file=out)
    print("    -t <N>. Number of topics. Default: "+str(ntopics),file=out)
    print("    -w <N>. Number of LDA models for ensemble. Default: "+str(nworkers),file=out)
#    print("    -c <cities file>: list of accepted place names.",file=out)
#    print("    -a: no filtering on location at all.",file=out)

    print("",file=out)


# returns tuple (doc_probs_by_topic, sorted_doc_probs_by_doc, top_docs_by_topic, topic_probs)
def topic_per_doc(model, corpus, top_n = None):
    doc_probs_by_topic = defaultdict(list)
    sorted_doc_probs_by_doc = []
    for doc_no, row in enumerate(model[corpus]):
        print(f"\r{doc_no}",end='', file=sys.stderr)
        #        print(row,file=sys.stderr)
        # below because the result format appears to differ between lda and ensemble lda:
        if type(row[0]) == tuple:
            doc_info = row
        else:
            doc_info = row[0]
        for topic_no, prob in doc_info:
            doc_probs_by_topic[topic_no].append((doc_no, prob))
        # sort by descending prob
        doc_probs = sorted(doc_info, key=lambda x: x[1], reverse=True)
        sorted_doc_probs_by_doc.append(doc_probs)
    top_docs_by_topic = []
    topic_probs = []
    #    for topic_no, doc_prob_pairs in enumerate(doc_probs_by_topic):
    for topic_no, doc_prob_pairs in doc_probs_by_topic.items():
        top = sorted(doc_prob_pairs, key=lambda x: x[1], reverse=True)
        if top_n is not None:
            top = top[0:top_n]
        top_docs_by_topic.append(top)
        marginal = sum([ p for doc,p in doc_prob_pairs]) / len(corpus)
        topic_probs.append(marginal)
    return doc_probs_by_topic, sorted_doc_probs_by_doc, top_docs_by_topic, topic_probs



def display_topics(model, corpus, dictionary, output=sys.stdout, with_probs=False, top_n_words = 15, top_n_docs = 5, doc_random_n = 12):
    doc_probs_by_topic, sorted_doc_probs_by_doc, top_docs_by_topic, topic_probs = topic_per_doc(model, corpus, top_n_docs)
    topics = model.get_topics()
    for topic_no,l in enumerate(topics):
        top_indexes = sorted(range(len(l)), key=lambda index: l[index], reverse=True)
        top_indexes = top_indexes[0:top_n_words]
        if with_probs:
            print(topic_no, ':', "{:.2f}".format(topic_probs[topic_no]), ':' , ", ".join([ dictionary[i]+"["+"{:.3f}".format(l[i])+"]" for i in top_indexes ]), file=output)
        else:
            print(topic_no, ':', "{:.2f}".format(topic_probs[topic_no]), ':' , ", ".join([ dictionary[i] for i in top_indexes ]), file=output)
        for top_doc, prob in top_docs_by_topic[topic_no]:
            doc_sample = [ dictionary[w] for w,f in corpus[top_doc] ]
            if len(doc_sample)>doc_random_n:
                doc_sample = random.sample(doc_sample, doc_random_n)
            print('   ',"{:.2f}".format(prob),  doc_sample, file=output)
        print()




def main():
    global model_type
    global ncores
    global ntopics
    global nworkers
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hm:t:c:w:")
    except getopt.GetoptError:
        usage(sys.stderr)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            usage(sys.stdout)
            sys.exit()
        elif opt == "-m":
            model_type = arg
        elif opt == "-c":
            ncores = int(arg)
        elif opt == "-t":
            ntopics = int(arg)
        elif opt == "-w":
            nworkers = int(arg)
#        elif opt == "-M":
#            max_prop = float(arg)

    if len(args) != 2:
        usage(sys.stderr)
        sys.exit(2)

    corpus_prefix = args[0]
    output = args[1]
    with open(output, 'w') as output_stream:

        corpus = corpora.MmCorpus(corpus_prefix)
        dictionary = corpora.Dictionary.load(corpus_prefix+'.dict')
        #    print(len(corpus))
        #    print(len(dictionary))

        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        model = None
        if model_type == "lda":
            model = models.LdaMulticore(corpus, 
                                    id2word=id2word, 
                                    num_topics=ntopics,
                                    workers=ncores,
                                    per_word_topics=True )
            display_topics(model,corpus, dictionary, output_stream)
        elif model_type == "ens":
            model = models.EnsembleLda(corpus=corpus,
                                   id2word=id2word,
                                   num_topics=ntopics,
                                   passes=2,
                                   iterations = 200,
                                   num_models=ncores,
                                   topic_model_class=models.LdaModel,
                                   ensemble_workers=nworkers,
                                   distance_workers=ncores)
            display_topics(model,corpus, dictionary, output_stream)
        elif model_type == "lsi":
            model = models.LsiModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=ntopics,
                                onepass=False)
            display_topics(model,corpus, dictionary, output_stream)
        else:
            raise Exception(f"Invalid model id '{model_type}'")



    

if __name__ == "__main__":
    main()

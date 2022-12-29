import os
import sys, getopt
from collections import defaultdict
from gensim import corpora
from pathlib import Path



PROG_NAME = "create-gensim-corpus.py"

min_freq = 10
max_prop = .5

min_users = 1
min_words = 1

def usage(out):
    print("Usage: "+PROG_NAME+" [options] <input data dir> <output prefix>",file=out)
    print("",file=out)
    print("  <input data dir> contains subdirectories, one per document.",file=out)
    print("",file=out)
    print("  Options:")
    print("    -h: print this help message.",file=out)
    print("    -m <min freq>. default: "+str(min_freq),file=out)
    print("    -M <max prop>. default: "+str(max_prop),file=out)
    print("    -u <min users in conv>. default: "+str(min_users),file=out)
    print("    -w <min words in conv>. default: "+str(min_words),file=out)
#    print("    -c <cities file>: list of accepted place names.",file=out)
#    print("    -a: no filtering on location at all.",file=out)

    print("",file=out)


class TextCorpus:

    def __init__(self, input_dir):
        self.input_dir = input_dir

    def __iter__(self):
        no = 0
        for subdir_no,subdir in enumerate(Path(self.input_dir).iterdir()):
#            print(subdir_no)
            if subdir.is_dir():
                doc = []
                users = set()
                for filename in Path(os.path.join(self.input_dir, subdir.name)).iterdir():
#                    print(f"   {filename}")
                    with open(filename, newline="\n") as infile:
                        for line in infile:
                            fields = line.rstrip().replace("\r", " ").split("\t")
                            if (len(fields)==2):
                                user,tweet = fields
                                users.add(user)
                                doc.extend(tweet.split(' '))
                            else:
                                if len(fields)!=1:
                                    raise Exception(f"Error, too nany fields: {' ; '.join(fields)}")
                if len(users) >= min_users and len(doc) >= min_words:
                    yield no, subdir, doc
                    no += 1


class BOWCorpus:

    def __init__(self, dic, text_corpus):
        self.dic = dic
        self.corpus = text_corpus

    def __iter__(self):
        for no, conv_id,doc in self.corpus:
            yield self.dic.doc2bow(doc)


def save_data(corpus, fname):
    dictionary = corpora.Dictionary(corpus)
    bow = [ dictionary.doc2bow(doc) for doc in corpus ]
    dictionary.save(fname+".dict")

def main():
    global min_freq
    global max_prop
    global min_users
    global min_words
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hm:M:u:w:")
    except getopt.GetoptError:
        usage(sys.stderr)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            usage(sys.stdout)
            sys.exit()
        elif opt == "-m":
            min_freq = int(arg)
        elif opt == "-M":
            max_prop = float(arg)
        elif opt == "-u":
            min_users = float(arg)
        elif opt == "-w":
            min_words = float(arg)

    if len(args) != 2:
        usage(sys.stderr)
        sys.exit(2)

    input_dir = args[0]
    output_prefix = args[1]

    raw_corpus = TextCorpus(input_dir)


    dictionary = corpora.Dictionary()
    with open(output_prefix+".conv-map", "w") as outfile:
        for no, conv_id, doc in raw_corpus:
            #for no, doc in enumerate(raw_corpus):
            #print("\r"+str(no), end='')
            dictionary.add_documents([doc], prune_at = None)
            outfile.write(str(no)+"\t"+str(os.path.basename(conv_id))+"\n")
    print('Number of unique tokens before filter_extremes: %d' % len(dictionary))
    dictionary.filter_extremes(no_below=min_freq, no_above=max_prop, keep_n=None)
    print('Number of unique tokens after filter_extremes: %d' % len(dictionary))
    dictionary.compactify()
    dictionary.save(output_prefix+".dict")

    bow_corpus = BOWCorpus(dictionary, raw_corpus)
    corpora.MmCorpus.serialize(output_prefix, bow_corpus)
    

if __name__ == "__main__":
    main()

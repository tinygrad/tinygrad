# pip install nltk
import os, time, io, hashlib, glob, multiprocessing, random
from pathlib import Path
from typing import List
import nltk
from tinygrad.helpers import getenv

def process_one_file_cleanup(filename:str):
  input_filename = filename + ""
  output_filename = filename + ".1"
  with io.open(input_filename, 'r', encoding='utf-8') as fin:
    with io.open(output_filename, 'w', encoding='utf-8') as fout:
      keep_next_line = True
      for line in fin:
        if not keep_next_line:
          keep_next_line = True
          continue
        if '<doc' in line:
          keep_next_line = False
          fout.write(u'\n')
          continue
        if '</doc>' in line: continue
        if len(line) == 1:
          fout.write(u'\n')
        else:
          sents = nltk.sent_tokenize(line) # sentence tokenize
          for sent in sents:
            sent_str = sent.strip()
            fout.write('%s\n' % sent_str)
        fout.write(u'\n')
  # subprocess.run(clean_up_cmd(output_filename, filename + ".2"), shell=True) # further cleanup
  os.remove(input_filename)

def cleanup_sentence_tokenize_wikipedia(path:str):
  print("Cleaning up files...")
  input_files = sorted(glob.glob(os.path.join(path, '*', '*')))
  num_files = len(input_files)
  nltk.download('punkt')
  print(f'Number of input files to process {num_files}')
  tic = time.time()
  p = multiprocessing.Pool(getenv("NUM_WORKERS", os.cpu_count()))
  p.map(process_one_file_cleanup, input_files)
  toc = time.time()
  print(f'Processed in {toc - tic:.2f} sec')

def process_one_file_for_seperation(file_id: int, input_files:List[str], num_files:int, num_test_articles:int, seed:int):
  input_filename = input_files[file_id]
  output_filename = input_filename[:-2] + ".2"
  num_articles = 0
  num_tests = int((file_id+1) * num_test_articles * 1.0 / num_files) \
      - int(file_id * num_test_articles * 1.0 / num_files)
  file_seed = seed + file_id * 13
  rng = random.Random(file_seed)
  test_articles = []

  with io.open(input_filename, 'r', encoding='utf-8', newline='\n') as fin:
    with io.open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
      lines = fin.read()
      articles = lines.split('\n\n')
      num_articles = len(articles)
      test_article_ids = []
      while len(test_article_ids) < num_tests:
        new_id = int(rng.random() * num_articles)
        if new_id in test_article_ids:
          continue
        test_article_ids.append(new_id)

      for i in range(num_articles):
        article = articles[i]
        if i in test_article_ids:
          # test_articles_in_files[file_id].append(article)
          test_articles.append(article)
        else:
          fout.write(article)
          fout.write('\n\n')

  print(f'Processed {input_filename} => {output_filename}, {num_tests} of {num_articles} articals picked into test set. {test_article_ids}')
  return test_articles

def seperate_test_set(path:str, num_test_articles:int=10000, seed:int=12345):
  os.makedirs(os.path.join(path, "results"), exist_ok=True)
  input_files = sorted(glob.glob(os.path.join(path, '*', '*', '*')))
  num_files = len(input_files)
  file_ids = range(num_files)
  print(f'Number of input files to process {num_files}')
  tic = time.time()
  p = multiprocessing.Pool(getenv("NUM_WORKERS", os.cpu_count()))
  test_articles_in_files = p.starmap(process_one_file_for_seperation, [(file_id, input_files, num_files, num_test_articles, seed) for file_id in file_ids])
  toc = time.time()
  print(f'Processed in {toc - tic:.2f} sec')

  output_filename = os.path.join(path, "results", "eval.txt")
  hash_filename = os.path.join(path, "results", "eval.md5")
  with io.open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
    with io.open(hash_filename, 'w', encoding='utf-8', newline='\n') as hashout:
      for f in test_articles_in_files:
        for article in f:
          fout.write(article)
          fout.write('\n\n')
          article_hash = hashlib.md5(article.rstrip().encode('utf-8')).hexdigest()
          hashout.write(article_hash)
          hashout.write('\n')

def clean_up_cmd(input_filename:str, output_filename:str):
  return """ cat {0} \
  | grep -v '^<doc [^>]*>$' \
  | grep -vE '\[\[Category:[^][]*\]\]' \
  | sed 's/\[\[\([^]|[]*\)\]\]/\1/g' \
  | sed 's/\[\[\([^]|[]*\)\]\]/\1/g' \
  | sed 's/\[\[[^]|[]*|\([^]|[]*\)\]\]/\1/g' \
  | sed 's/\[\[[^]|[]*|\([^]|[]*\)\]\]/\1/g' \
  | sed 's/\[\[[:]*[Ff]ile:[^][]*\]\]//g' \
  | sed 's/\[\[[Mm]edia:[^][]*\]\]//g' \
  | sed 's/\[\[[Ii]mage:[^][]*\]\]//g' \
  | sed 's/\[\([^]|[]*\)\]/\1/g' \
  | sed 's/\[\[\([^][]*\)\]\]//g' \
  | sed 's/alt=//g' \
  | sed 's/<\/doc>/\r/g' \
  | sed 's/<chem\([^<]*\)<\/chem>/\1/g' \
  | sed 's/<ins\([^<]*\)<\/ins>/\1/g' \
  | sed 's/<\, ref \([^<]*\)<\/ref>//g' \
  | sed 's/<includeonly\([^<]*\)<\/includeonly>//g' \
  | sed 's/<graph\([^<]*\)<\/graph>//g' \
  | sed 's/<section\([^\\]*\)\/>//g' \
  | sed 's/<meta\([^\\]*\)\/>//g' \
  | sed 's/<hr\([^\\]*\)\/>//g' \
  | sed 's/<gallery\([^>]*\)>//g' \
  | sed 's/<ref\([^<]*\)<\/ref>//g' \
  | sed 's/<ref\([^>]*\)>//g' \
  | sed 's/<http\([^>]*\)>//g' \
  | sed 's/<Ref\([^>]*\)>//g' \
  | sed 's/<mapframe \([^\/]*\)\/>//g' \
  | sed 's/<mapframe\([^>]*\)>//g' \
  | sed 's/<\/mapframe>//g' \
  | sed 's/<poem>//g' \
  | sed 's/<\/poem>//g' \
  | sed 's/<math>//g' \
  | sed 's/<\/math>//g' \
  | sed 's/<ref>//g' \
  | sed 's/<\/ref>//g' \
  | sed 's/<div\([^>]*\)>//g' \
  | sed 's/<\/div\([^>]*\)>//g' \
  | sed 's/<\/div style>//g' \
  | sed 's/<\/div>//g' \
  | sed 's/<sup>//g' \
  | sed 's/<\/sup>//g' \
  | sed 's/<br>//g' \
  | sed 's/<\/br>//g' \
  | sed 's/<BR>//g' \
  | sed 's/<\/BR>//g' \
  | sed 's/<Br>//g' \
  | sed 's/<\/Br>//g' \
  | sed 's/<del>//g' \
  | sed 's/<\/del>//g' \
  | sed 's/<nowiki>//g' \
  | sed 's/<\/nowiki>//g' \
  | sed 's/<NOWIKI>//g' \
  | sed 's/<\/NOWIKI>//g' \
  | sed 's/<onlyinclude>//g' \
  | sed 's/<\/onlyinclude>//g' \
  | sed 's/<includeonly>//g' \
  | sed 's/<\/includeonly>//g' \
  | sed 's/<small>//g' \
  | sed 's/<\/small>//g' \
  | sed 's/<chem>//g' \
  | sed 's/<\/chem>//g' \
  | sed 's/<noinclude>//g' \
  | sed 's/<\/noinclude>//g' \
  | sed 's/<gallery>//g' \
  | sed 's/<\/gallery>//g' \
  | sed 's/<graph>{{//g' \
  | sed 's/<graph>//g' \
  | sed 's/}}<\/graph>//g' \
  | sed 's/<\/graph>//g' \
  | sed 's/<\/references>//g' \
  | sed 's/<poem \([^>]*\)>//g' \
  > {1} """.format(input_filename, output_filename)

if __name__ == "__main__":
  cleanup_sentence_tokenize_wikipedia(os.path.join(Path(__file__).parent.parents[2] / "extra" / "datasets"/ "wiki"))
  seperate_test_set(os.path.join(Path(__file__).parent.parents[2] / "extra" / "datasets"/ "wiki"))
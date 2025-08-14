# searchwiki.py
# Run a search on the downloaded wikipedia data. The wikipediate data
# should already be downloaded, extracted, and preprocessed by the
# WikipediaEnDownload submodule as well as preprocess.py in this repo.
# Python 3.9
# Windows/MacOS/Linux


import argparse
import os
import json
import random
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

from search import ReRankSearch, TF_IDF, BM25, VectorSearch
from search import print_results
from search import load_data_from_msgpack, load_data_from_json
from search import load_article_text


seed = 1234
random.seed(seed)


def evaluate_search_results(results: List[Tuple[float, str, str, List[int]]], selected_doc: str) -> None:
	'''
	Evaluate whether the target document appears within the search 
		top-n results for various values of n. These results are
		print out to console.
	@param: results (List[Tuple[float, str, str, List[int]]]), the raw
		results list returned from the search engine.
	@param: selected_doc (str), the target document that is expected to
		be within the results.
	@return: returns nothing.
	'''
	# Isolate the documents (file + sha) returned in the results. 
	result_docs = [result[1] for result in results]

	# Acquire the position of the target/expected document from the
	# results documents list. Position is -1 if it does not appear in
	# the results list.
	doc_idx = result_docs.index(selected_doc) if selected_doc in result_docs else -1
	
	# List the top-n values you are using. Iterate through that list
	# and print whether the document is found in the top-n results for
	# that n.
	top_n = [5, 10, 25, len(results)]
	for n in top_n:
		print(f"Top-{n}: {doc_idx >= 0 and doc_idx < n}")


def get_random_paragraph_from_article(article_text: str) -> str:
	'''
	Get a random paragraph from the wikipedia article.
	@param: article_text (str), the text of the wikipedia article.
	@return: returns a random, non-empty paragraph from that article.
	'''
	# Return the article text (empty string) if the article text is 
	# just an empty string.
	if article_text == "":
		return article_text
	
	# Split article text by paragraph and remove all empty string 
	# entries.
	split_text = [
		text for text in article_text.split("\n") if text.strip()
	]

	# Randomly sample a paragraph from the split text. Weigh the 
	# choices by the length of the paragraph.
	return random.choices(
		split_text, 
		weights=[len(text) for text in split_text],
		k=1
	)[0]


def get_article_entries(articles: List[str]) -> List[Tuple[str, str, str, str]]:
	'''
	Extract the article text, file location, and SHA1 for each article.
	@param: articles (List[str]), the list of articles (file + SHA1).
	@return: returns a list containing the article (file + SHA1), 
		article text, file, and SHA1 in a tuple.
	'''
	# Initialize the results list.
	results = []
	
	# Iterate through each article.
	file_sha_map = dict()
	for article in articles:
		# Split the file and SHA1.
		split_string = article.split(".xml")
		if len(split_string) != 2:
			results.append((article, "", "", ""))
			continue

		# Isolate the file, SHA1, and article text.
		file, sha = split_string[0] + ".xml", split_string[1]
		if file in file_sha_map:
			file_sha_map[file].append(sha)
		else:
			file_sha_map[file] = [sha]

	for file, sha_list in file_sha_map.items():
		texts = load_article_text(file, sha_list)

		for idx, text in enumerate(texts):
			sha = sha_list[idx]
			results.append((file + sha, text, file, sha))

	# Return the results.
	return results


def get_all_articles(config: Dict[str, Any]) -> List[str]:
	'''
	Get all of the articles (file + SHA1) from the wikipedia dataset 
		metdata (output from preprocessing). Remove redirect articles
		as well (if the metadata to do so is available).
	@param: config (Dict[str, Any]), the configuration data from the 
		config.json file.
	@return: returns a list of strings which are the articles of the 
		dataset.
	'''
	# Pull all documents accounted for from the doc_to_words metadata.
	doc_to_words_path = config["preprocessing"]["doc_to_words_path"]
	bow_files = os.listdir(doc_to_words_path)

	if any([file.endswith(".parquet") for file in bow_files]):
		files = [
			os.path.join(doc_to_words_path, file) for file in bow_files
			if file.endswith(".parquet")
		]
	elif any([file.endswith(".msgpack") for file in bow_files]):
		files = [
			os.path.join(doc_to_words_path, file) for file in bow_files
			if file.endswith(".msgpack")
		]
	elif any([file.endswith(".json") for file in bow_files]):
		files = [
			os.path.join(doc_to_words_path, file) for file in bow_files
			if file.endswith(".json")
		]
	else:
		print(f"No supported files detected in {doc_to_words_path}")
		exit(1)

	articles = []
	for file in files:
		if file.endswith(".parquet"):
			data = pd.read_parquet(file)
			articles.extend(data["doc"].unique().tolist())
		if file.endswith(".msgpack"):
			data = load_data_from_msgpack(file)
			articles.extend(list(data.keys()))
		if file.endswith(".json"):
			data = load_data_from_json(file)
			articles.extend(list(data.keys()))

	# Deduplication.
	articles = list(set(articles))

	# Identify all redirect articles and filter them from the current 
	# list. This will have required the precompute_sparse_vectors 
	# script to have run.
	redirects_path = config["preprocessing"]["staging_redirect_path"]
	redirect_files = [
		os.path.join(redirects_path, file) 
		for file in os.listdir(redirects_path)
		if file.endswith(".parquet")
	]

	if os.path.exists(redirects_path) and len(redirect_files) > 0:
		data = pd.DataFrame()
		for file in redirect_files:
			if len(data) == 0:
				data = pd.read_parquet(file)
			else:
				data = pd.concat(
					[data, pd.read_parquet(file)], ignore_index=True
				)

		# Explode the articles list so each article gets its own row.
		data_exploded = data.explode("articles", ignore_index=True)

		# Combine file + article into a single string.
		redirect_articles = (
			data_exploded["file"] + data_exploded["articles"]
		).tolist()

		# Remove redirect articles from the list.
		articles = list(set(articles) - set(redirect_articles))

	# Return the list of articles.
	return articles


def test(print_search: bool = False) -> None:
	'''
	Test each of the search processes on the wikipedia dataset.
	@param: print_search (bool), whether to print the search results 
		during the tests. Is False by default.
	@return: returns nothing.
	'''
	# Input values to search engines.
	with open("config.json", "r") as f:
		config = json.load(f)

	bow_dir = "./metadata/bag_of_words"
	index_dir = "./test-temp"

	preprocessing_paths = config["preprocessing"]
	corpus_staging = os.path.join(
		preprocessing_paths["staging_corpus_path"], 
		"corpus_cache",
	)
	corpus_path = os.path.join(corpus_staging, "corpus_stats.json")

	# Load corpus stats from the corpus path JSON if it exists. Use the
	# config JSON if the corpus JSON is not available.
	if os.path.exists(corpus_path):
		with open(corpus_path, "r") as f:
			corpus_stats = json.load(f)
			tfidf_corpus_size = corpus_stats["corpus_size"]
			bm25_corpus_size = corpus_stats["corpus_size"]
			bm25_avg_doc_len = corpus_stats["avg_doc_len"]
	else:
		tfidf_corpus_size = config["tf-idf_config"]["corpus_size"]
		bm25_corpus_size = config["bm25_config"]["corpus_size"]
		bm25_avg_doc_len = config["bm25_config"]["avg_doc_len"]

	model = "bert"
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"

	###################################################################
	# INITIALIZE SEARCH ENGINES
	###################################################################
	search_1_init_start = time.perf_counter()
	tf_idf = TF_IDF(bow_dir, corpus_size=tfidf_corpus_size)
	search_1_init_end = time.perf_counter()
	search_1_init_elapsed = search_1_init_end - search_1_init_start
	print(f"Time to initialize TF-IDF search: {search_1_init_elapsed:.6f} seconds")

	search_2_init_start = time.perf_counter()
	bm25 = BM25(
		bow_dir, corpus_size=bm25_corpus_size, 
		avg_doc_len=bm25_avg_doc_len
	)
	search_2_init_end = time.perf_counter()
	search_2_init_elapsed = search_2_init_end - search_2_init_start
	print(f"Time to initialize BM25 search: {search_2_init_elapsed:.6f} seconds")
	
	# search_3_init_start = time.perf_counter()
	# vector_search = VectorSearch()
	# search_3_init_end = time.perf_counter()
	# search_3_init_elapsed = search_3_init_end - search_3_init_start
	# print(f"Time to initialize Vector search: {search_3_init_elapsed:.6f} seconds")

	# search_4_init_start = time.perf_counter()
	# rerank = ReRankSearch(bow_dir, index_dir, model, device=device)
	# search_4_init_end = time.perf_counter()
	# search_4_init_elapsed = search_4_init_end - search_4_init_start
	# print(f"Time to initialize Rerank search: {search_4_init_elapsed:.6f} seconds")

	search_engines = [
		("tf-idf", tf_idf), 
		("bm25", bm25), 
		# ("vector", vector_search),
		# ("rerank", rerank)
	]

	###################################################################
	# EXACT PASSAGE RECALL
	###################################################################
	# Given passages that are directly pulled from random articles, 
	# determine if the passage each search engine retrieves is correct.

	print("Indexing all articles to sample from for testing:")
	articles = get_all_articles(config)
	selected_docs = random.sample(articles, 5)
	print("Sampled 5 articles.")

	# TODO: Fix this bottleneck. Takes a few minutes. Consider rust
	# extension? Not super critical since I got it down from 6 minutes 
	# to ~3 minutes. Biggest bottleneck is probably the file IO of 
	# parsing through files and getting the exact article texts.
	start = time.perf_counter()
	article_entries = get_article_entries(selected_docs)
	end = time.perf_counter()
	elapsed = end - start
	print(f"Isolated article texts in {elapsed:.6f} seconds.")

	query_passages = [
		get_random_paragraph_from_article(text)
		for _, text, _, _ in article_entries
	]
	print("Isolated query passages.")
	print("=" * 72)
	print("Testing Sparse Vector search engines:")

	# NOTE:
	# Mean search times for each search engine.
	# TF-IDF: ~240s (or 4 min)
	# BM25: ~200s (or 3.5 min)
	# Current bottleneck is the text loading. 

	# Iterate through each search engine.
	# for name, engine in [search_engines[1]]:
	for name, engine in search_engines[:2]:
		# Search engine banner text.
		print(f"Searching with {name}")
		search_times = []

		# Iterate through each passage and run the search with the 
		# search engine.
		for idx, query in enumerate(query_passages):
			# Run the search and track the time it takes to run the 
			# search.
			query_search_start = time.perf_counter()
			results = engine.search(query)
			query_search_end = time.perf_counter()
			query_search_elapsed = query_search_end - query_search_start

			# Print out the search time.
			print(f"Search returned in {query_search_elapsed:.6f} seconds")
			print()

			# Print out the search results if specified.
			if print_search:
				print_results(results, search_type=name)

			# Append the search time to a list.
			search_times.append(query_search_elapsed)

			# Evaluate the search results.
			evaluate_search_results(results, selected_docs[idx])

		# Compute and print the average search time.
		avg_search_time = sum(search_times) / len(search_times)
		print(f"Average search time: {avg_search_time:.6f} seconds")
		print("=" * 72)
	exit()

	###################################################################
	# GENERAL QUERY
	###################################################################
	# Given passages that have some relative connection to random 
	# articles, determine if the passage each search engine retrieves 
	# is correct.
	query_text = [

	]
	
	for name, engine in search_engines:
		pass
	pass


def search_loop() -> None:
	'''
	Run an infinite loop (or until the exit phrase is specified) to
		perform search on wikipedia.
	@param: takes no arguments.
	@return: returns nothing.
	'''

	# Read in the title text (ascii art).
	with open("title.txt", "r") as f:
		title = f.read()

	exit_phrase = "Exit Search"
	print(title)
	print()
	search_query = input("> ")
	pass


def main() -> None:
	'''
	Main method. Will either run search engine tests or interactive
		search depending on the program arguments.
	@param: takes no arguments.
	@return: returns nothing.
	'''
	# Set up argument parser.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--test",
		action="store_true",
		help="Specify whether to run the search engine tests. Default is false/not specified."
	)
	parser.add_argument(
		"--print_results",
		action="store_true",
		help="Specify whether to print the search results during the search engine tests. Default is false/not specified."
	)
	args = parser.parse_args()

	# Depending on the arguments, either run the search tests or just
	# use the general search function.
	if args.test:
		test()
	else:
		search_loop()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
# searchwiki.py
# Run a search on the downloaded wikipedia data. The wikipediate data
# should already be downloaded, extracted, and preprocessed by the
# WikipediaEnDownload submodule as well as preprocess.py in this repo.
# Python 3.9
# Windows/MacOS/Linux


import argparse
import os
import json
import math
import random
import re
import shutil
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import set_seed

from search import ReRankSearch, TF_IDF, BM25, VectorSearch
from search import print_results
from search import load_data_from_msgpack, load_data_from_json
from search import load_article_text


seed = 1234
random.seed(seed)
np.random.seed(seed)
set_seed(seed)


def generate_question(passage: str, model_name: str, device: str) -> str:
	'''
	Use a small LLM to generate a question that can be directly 
		answered by the input passage.
	@param passage (str), the input passage around which the question 
		will be generated.
	@param model_name (str), which LLM to run.
	@param device (str), which hardware accelerator to run the LLM on.
	@return: returns the generated question string.
	'''
	# Set up small LLM model (download if needed). Models are 
	# chat/instruct models to allow for ChatGPT-like prompting instead
	# of text completion.
	model_path = model_name.replace("/", "_")
	cache_path = model_path + "_tmp"

	# Check for path and that path is a directory. Make it if either is
	# not true.
	if not os.path.exists(model_path) or not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)

	# Check for path the be populated with files (weak check). Download
	# the tokenizer and model and clean up files once done.
	if len(os.listdir(model_path)) == 0:
		print(f"Model {model_name} needs to be downloaded.")

		# Check for internet connection (also checks to see that
		# huggingface is online as well). Exit if fails.
		response = requests.get("https://huggingface.co/")
		if response.status_code != 200:
			print(f"Request to huggingface.co returned unexpected status code: {response.status_code}")
			print(f"Unable to download {model_name} model.")
			exit(1)

		# Create cache path folders.
		os.makedirs(cache_path, exist_ok=True)
		os.makedirs(model_path, exist_ok=True)

		# Load tokenizer and model.
		model_id = model_name
		tokenizer = AutoTokenizer.from_pretrained(
			model_id, cache_dir=cache_path, device_map=device
		)
		model = AutoModelForCausalLM.from_pretrained(
			model_id, cache_dir=cache_path, device_map=device
		)

		# Save the tokenizer and model to the save path.
		tokenizer.save_pretrained(model_path)
		model.save_pretrained(model_path)

		# Delete the cache.
		shutil.rmtree(cache_path)
	
	# Load the tokenizer and model.
	tokenizer = AutoTokenizer.from_pretrained(
		model_path, device_map=device
	)
	model = AutoModelForCausalLM.from_pretrained(
		model_path, device_map=device
	)

	# Initialize model pipeline.
	pipe = pipeline(
		"text-generation",
		model=model.to(device),
		tokenizer=tokenizer,
	)

	# Prompt.
	prompt_template = [
		{
			"role": "system",
			"content": "You are a friendly and helpful chatbot who always responds to any query or question asked."
		},
		{
			"role": "user",
			"content": f"Given the following text passage, create a question that can be answered by information directly found in the text (DO NOT supply the answer):\n\n{passage}"
		},
	]
	prompt_str = pipe.tokenizer.apply_chat_template(
		prompt_template, 
		tokenize=False,
		add_generation_prompt=True,
	)

	# Pass prompt to model.
	output = pipe(
		prompt_str,
		num_return_sequences=1,
		do_sample=True,
		temperature=1.25,
		# top_k=8,
		top_k=16,
		top_p=0.90,
		max_new_tokens=pipe.tokenizer.model_max_length
	)

	# Return output.
	return output[0]["generated_text"].replace(prompt_str, "")


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
		text.strip() for text in article_text.split("\n\n") if text.strip()
	]

	# Crude wiki markup cleanup.
	for idx, text in enumerate(split_text):
		text = re.sub(r"\{\{.*?\}\}", "", text)  # remove templates
		# text = re.sub(r"\[\[.*?:.*?\]\]", "", text)  # categories/files
		# text = re.sub(r"==.*?==", "", text)  # headers
		text = re.sub(r"<.*?>", "", text)  # html tags
		split_text[idx] = text

	# Compute weights as lengths of each paragraph.
	weights = []
	for text in split_text:
		weight = len(text)

		# If a paragraph (split along individual newline characters) is
		# too long (indicating some sort of table or other structure),
		# nullify the weight for that paragraph.
		if len(text.split("\n")) > 5:
			weight = math.ceil(weight * 0.01)

		# Append the weight to the list.
		weights.append(weight)

	# Randomly sample a paragraph from the split text. Weigh the 
	# choices by the length of the paragraph.
	return random.choices(
		split_text, 
		# weights=[len(text) for text in split_text],
		weights=weights,
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


def get_article_lengths(config: Dict[str, Any], articles: List[str]) -> List[int]:
	'''
	Retrieve the length of the respective articles provided.
	@param: config (Dict[str, Any]), the configuration data from the 
		config.json file.
	@param: articles (List[str]), the list of articles being targeted.
	@return: returns a list of int values which are the lengths of the 
		respective input articles.
	'''
	# Pull all documents accounted for from the sparse vector metadata.
	sparse_vector_path = config["preprocessing"]["sparse_vector_path"]
	sparse_vector_files = [
		os.path.join(sparse_vector_path, file)
		for file in os.listdir(sparse_vector_path)
		if file.endswith(".parquet")
	]

	# Throw an error if no such files were found.
	if len(sparse_vector_files) == 0:
		print(f"No supported files detected in {sparse_vector_path}")
		exit(1)

	# Iterate through the files and build a mapping of the documents to
	# their lengths.
	doc_len_map = dict()
	for file in sparse_vector_files:
		if file.endswith(".parquet"):
			data = pd.read_parquet(file)
			doc_len_map.update(zip(data["doc"], data["doc_len"]))

	# Iterate over the article list and use the map to get the lengths
	# for each respective document. Return the resulting list.
	article_lengths = [doc_len_map.get(doc, 0) for doc in articles]
	return article_lengths


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
		elif file.endswith(".msgpack"):
			data = load_data_from_msgpack(file)
			articles.extend(list(data.keys()))
		elif file.endswith(".json"):
			data = load_data_from_json(file)
			articles.extend(list(data.keys()))

	# Deduplication.
	articles = sorted(list(set(articles)))

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
	# elif torch.backends.mps.is_available():
	# 	device = "mps"

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
	
	search_3_init_start = time.perf_counter()
	vector_search = VectorSearch(model, index_dir, device)
	search_3_init_end = time.perf_counter()
	search_3_init_elapsed = search_3_init_end - search_3_init_start
	print(f"Time to initialize Vector search: {search_3_init_elapsed:.6f} seconds")

	search_4_init_start = time.perf_counter()
	rerank = ReRankSearch(bow_dir, index_dir, model, device=device)
	search_4_init_end = time.perf_counter()
	search_4_init_elapsed = search_4_init_end - search_4_init_start
	print(f"Time to initialize Rerank search: {search_4_init_elapsed:.6f} seconds")

	search_engines = [
		("tf-idf", tf_idf), 
		("bm25", bm25), 
		("vector", vector_search),
		("rerank", rerank)
	]

	###################################################################
	# EXACT PASSAGE RECALL
	###################################################################
	# Given passages that are directly pulled from random articles, 
	# determine if the passage each search engine retrieves is correct.

	print("Indexing all articles to sample from for testing:")
	articles = get_all_articles(config)
	article_lengths = get_article_lengths(config, articles)
	article_lengths = [math.log(length) for length in article_lengths]
	probabilities = np.array(article_lengths) / np.sum(article_lengths)
	selected_docs = np.random.choice(
		articles, 
		size=5,
		replace=False,
		p=probabilities
	)
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
	print()

	# Print out the isolated query passages.
	print("QUERY PASSAGES:")

	for idx, query_passage in enumerate(query_passages):
		print(article_entries[idx][0])
		print(f"Passage:\n{query_passage}")
		print()

	print("=" * 72)
	print("Testing Sparse Vector search engines:")

	# NOTE:
	# Mean search times for each search engine.
	# TF-IDF: ~360s (or 6 min)
	# BM25: ~390s (or 6.5 min)
	# Current bottleneck is the text loading. 

	# Iterate through each search engine (sparse vector engines only/
	# TF-IDF & BM25).
	sparse_engines = [
		(name, engine) for name, engine in search_engines
		if name in ["tf-idf", "bm25"]
	]
	for name, engine in sparse_engines:
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
			print()

		# Compute and print the average search time.
		avg_search_time = sum(search_times) / len(search_times)
		print(f"Average search time: {avg_search_time:.6f} seconds")
		print("=" * 72)

	###################################################################
	# GENERAL QUERY
	###################################################################
	# Given passages that have some relative connection to random 
	# articles, determine if the passage each search engine retrieves 
	# is correct.
	print("Now testing generic queries")

	model_name = config["test"]["model"]
	valid_models = config["test"]["models"]
	assert model_name in valid_models, \
		f"Expected model in config to be one of {valid_models}. Recieved {model_name}"

	print("Generating abstract questions from the same passages.")
	start = time.perf_counter()
	abstract_query_text = [
		generate_question(query, model_name, device) 
		for query in query_passages
	]
	end = time.perf_counter()
	elapsed = end - start
	print(f"Questions generated in {elapsed:.6f} seconds")
	print()

	# Print out the isolated query passages.
	print("ABSTRACT QUERY PASSAGES:")

	for idx, query_passage in enumerate(abstract_query_text):
		print(article_entries[idx][0])
		print(f"Abstract Question:\n{query_passage}")
		print()
		print(f"Passage:\n{query_passages[idx]}")
		print()

	print("=" * 72)
	print("Testing all Vector search engines:")

	# NOTE:
	# Mean search times for each search engine.
	# TF-IDF: ~s (or  min)
	# BM25: ~s (or  min)
	# ReRank: ~s (or  min)
	# Current bottleneck is the text loading. 
	
	# Iterate through each search engine.
	for name, engine in search_engines:
		# Skip vector search engine. Requires too many resources to
		# generate embeddings.
		if "vector" in name:
			continue

		# Search engine banner text.
		print(f"Searching with {name}")
		search_times = []

		# Iterate through each passage and run the search with the 
		# search engine.
		for idx, query in enumerate(abstract_query_text):
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
			print()

		# Compute and print the average search time.
		avg_search_time = sum(search_times) / len(search_times)
		print(f"Average search time: {avg_search_time:.6f} seconds")
		print("=" * 72)

	shutil.rmtree(index_dir)

	exit(0)


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
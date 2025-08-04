# Simple Wikipedia Search

Description: Similar to the WikipediaEnSearch (which operates on the full Wikipedia dumps), this repository focuses on applying the same search techniques on the Simple Wikipedia dumps.


### Setup

 - Environment
 - Download data
 - Preprocess data


### Minimum Hardware Specifications

 - 64 GB RAM if running download script `download.py`.
     - Data download script (specifically the decompression step) will OOM on 8GB RAM.
 - 16 GB RAM if running preprocessing script with minimal workers.
     - Preprocessing script takes around 4 hours (per document) with 4 workers/processors on bag-of-words, so single worker should take around 64 hours (4 hours x 4 workers x 4 documents) to complete for that stage.
     - Memory overhead under 4 workers (bow or vector) is around 36 GB regardless. 36 / 4 = 9, so 16 GB RAM is recommended at minimum.



### Notes

 - Programmatic decompression of the downloaded articles `.xml.bz2` uses a ridiculous amount of RAM. I'd say it makes sense to keep a copy of the data in a huggingface datasets repository however, the data is updated on the first and twentieth of every month (meaning the data is updated relatively frequently).
     - Onboard/built-in archive utility on 8 GB machine is able to handle decompressing the downloaded file bundle without hitting OOM.
     - Trying to stream with `bz2` with a fixed buffer size also provides no relief on the resources. 
 - I should go back an rework the preprocessing pipeline to use rust or something. The time it is taking to run through the dataset (despite it being around 1.8 GB uncompressed vs 95 GB in the uncompressed full Wikipedia dataset) is not acceptable.
     - Possible optimizations:
         - Rewrite in rust.
         - Cut down/remove/ignore all "redirect" pages.
 - Running the vector database on the entire Simple Wikipedia corpus is still not scalable. Was barely 1/2 way through the first decompressed `.xml` file shard and it had generated over 312 GB of embeddings. A previous napkin calculation for the original full Wikipedia corpus showed that it would need terabytes of storage, so it stands to reason that a similar need would be required for this dataset. 
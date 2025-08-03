# 


### Minimum Hardware Specifications

 - 64GB RAM If running download script `download.py`.
     - Data download script (specifically the decompression step) will OOM on 8GB RAM.
 - 16 GB RAM If running preprocessing script with minimal workers.
     - Preprocessing script takes around 4 hours with 4 workers/processors on bag-of-words, so single worker should take around 16 hours to complete for that stage.
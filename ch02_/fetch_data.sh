#!/bin/bash

curl -X GET \
     "https://datasets-server.huggingface.co/rows?dataset=lm1b&config=plain_text&split=train&offset=0&length=100" | \
jq '.rows[].row.text' | 
sed -e 's/\"//g' > fetched_corpus_data.txt

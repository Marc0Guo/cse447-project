#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Marco Guo,jguo27\nSophia Dai, daiyefan\nBohan Fang, fang0106" > submit/team.txt

# train model
python src/NGram_Model.py train --work_dir work --n 6 --max_train 300000 --multilingual --max_per_lang 30000 --min_count 3

# make predictions on example data submit it in pred.txt
python src/NGram_Model.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file and requirements
cp Dockerfile submit/Dockerfile
cp requirements.txt submit/requirements.txt

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit

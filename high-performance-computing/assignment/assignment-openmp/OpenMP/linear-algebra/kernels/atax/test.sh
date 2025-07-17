#!/bin/bash

./dump.sh "$1"
./validate.py ./dump/mini/"$1" ./dump/mini/SEQUENTIAL
./validate.py ./dump/small/"$1" ./dump/small/SEQUENTIAL
./validate.py ./dump/standard/"$1" ./dump/standard/SEQUENTIAL
./validate.py ./dump/large/"$1" ./dump/large/SEQUENTIAL
./validate.py ./dump/extralarge/"$1" ./dump/extralarge/SEQUENTIAL
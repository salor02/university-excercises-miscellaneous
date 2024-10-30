#!/bin/bash

remote_path=$'/nfs/home/301414\r/hpc-lab-public-sources/'
local_path='./board4'
port=22704
user=301414

sshfs -p $port "$user"@elgamal.mat.unimo.it:"$remote_path" "$local_path"
ssh -p $port "$user"@elgamal.mat.unimo.it
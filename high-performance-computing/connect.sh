#!/bin/bash

remote_path=$'/nfs/home/301414\r/hpc-lab-public-sources/'
local_path='./board4'
port=22704
user=301414

if [ "$1" = "tunnel" ]; then
    remote_host='localhost'
    autossh -f -N -L 22704:"$remote_host":22704 student@wks-marongiu2.mat.unimo.it -v
else
    remote_host='elgamal.mat.unimo.it'
fi

sshfs -p $port "$user"@"$remote_host":"$remote_path" "$local_path"
ssh -p $port "$user"@"$remote_host"

    
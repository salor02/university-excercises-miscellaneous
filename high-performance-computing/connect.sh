#!/bin/bash

remote_path=$'/nfs/home/301414\r/'
local_path='./board4'
port=22700
user=301414

if [ "$1" = "tunnel" ]; then
    remote_host='localhost'
    ssh -f -N -L "$port":"$remote_host":"$port" student@wks-marongiu2.mat.unimo.it
else
    remote_host='elgamal.mat.unimo.it'
fi

sshfs -p $port "$user"@"$remote_host":"$remote_path" "$local_path"
ssh -p $port "$user"@"$remote_host"

    
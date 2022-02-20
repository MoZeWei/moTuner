#!/bin/bash

read -p "File path(without name): " file_path
read -p "File name: " file_name
read -p "dst path(beyond /home/mzw/):" dst_path

scp ${file_path}/${file_name} mzw@nscc-gz.wu-kan.cn:/home/mzw/${dst_path}

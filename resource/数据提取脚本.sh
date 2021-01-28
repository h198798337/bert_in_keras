#!/bin/bash
. /etc/profile
#. ~/.bash_profile
##################################################
# Author: fanqiang
# create date: 2021-01-27
# Content: 数据提取脚本
# parameter:
# desc:
# 20210118:
##################################################
cd "$(dirname "$0")"

in_file=$1
v_pre=$2
v_dim=$3
out_file=$2_$3_tran.txt
echo '参数：'$in_file $v_pre $v_dim $out_file
echo '提取数据...'
cat $in_file|awk -F',' '{print $1,$2}'|awk -F' ' -v pre=$v_pre '{if ($2=="'${v_dim}'") print pre,$1,1; else print pre,$1,0;}'|sort -R > $out_file
echo '提取数据结束...'

echo '计算总行数...'
line_count=`cat $out_file|wc -l`
echo '总行数为'$line_count
train_count=$(($line_count*2/3))
valid_count=$(($line_count-$train_count))

echo '训练集行数为'$train_count
cat $out_file|head -n $train_count > train_data_$2_$3.txt
echo '验证集行数为'$valid_count
cat $out_file|tail -n $valid_count > valid_data_$2_$3.txt

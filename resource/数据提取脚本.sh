cat private_weiduqinggan.txt|awk -F',' '{print $1,$2}'|awk -F' ' -v pre='使用效果' '{if ($2=="功能") print pre,$1,1; else print pre,$1,0;}'|sort -R > 使用效果.txt
cat 使用效果.txt|head -n 1043 > train_data.txt
cat 使用效果.txt|head -n 500 > valid_data.txt
EXP_NAME=heatmap_9_11_600t

#set -x
#trap read debug

python heatmap.py -f "heatmap_data/all_data_heatmap_9_9_25.json" \
                     "heatmap_data/all_data_heatmap_9_9_50.json" \
                     "heatmap_data/all_data_heatmap_9_9_75.json" \
                     "heatmap_data/all_data_heatmap_9_9_10.json"

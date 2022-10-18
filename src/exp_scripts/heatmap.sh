EXP_NAME=heatmap_9_12_600t

#set -x
#trap read debug

python heatmap.py -f "heatmap_data/all_data_${EXP_NAME}_25.json" \
                     "heatmap_data/all_data_${EXP_NAME}_50.json" \
                     "heatmap_data/all_data_${EXP_NAME}_75.json" \
                     "heatmap_data/all_data_${EXP_NAME}_10.json"

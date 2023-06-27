import json
import time

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
eyeblink_threshold = 0.3

results = json.load(open('demo_video/intermediate_results/results.json','r'))

filtered_results = []

for query in results:
    blinks_converted = []
    eyeblink_buffer = []
    for index in range(0, len(query['blink_scores'])):
        if query['blink_scores'][index] >= eyeblink_threshold and eyeblink_buffer == []:
            eyeblink_buffer.append(index)
        if query['blink_scores'][index] < eyeblink_threshold and eyeblink_buffer != []:
            sum = 0
            for i in range(eyeblink_buffer[0],index):     
                sum +=query['blink_scores'][i]
            avg_score = sum/(index-eyeblink_buffer[0])
            blinks_converted.extend([[eyeblink_buffer[0] , index - 1, avg_score]])
            eyeblink_buffer = []
        if (index == len(query['blink_scores']) - 1) and eyeblink_buffer != []: # If it is an end frame
            sum = 0
            for i in range(eyeblink_buffer[0], index+1):  
                sum += query['blink_scores'][i]
            avg_score = sum / (index - eyeblink_buffer[0]+1)
            blinks_converted.extend([[eyeblink_buffer[0], index, avg_score]])
            eyeblink_buffer = []
    query.update({'blinks_converted': blinks_converted})
    filtered_results.append(query)
json.dump(filtered_results, open('demo_video/intermediate_results/results_blink_converted.json', 'w'))
print('Done')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

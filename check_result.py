import sys
import json
import statistics

try:
    with open('example_output.json') as json_file:
        example_result = json.load(json_file)
except (ValueError, Exception):
    print("entry json not loaded")
    sys.exit(0)


try:
    with open('output.json') as json_file:
        my_result = json.load(json_file)
except (ValueError, Exception):
    print("entry json not loaded")
    sys.exit(0)

score = 0
counter = 0
list_for_medians = []
for example_key, example_results_array in example_result.items():
    dict = my_result[example_key]
    print(example_results_array)
    print(dict)
    print('')
    sum_of_img = 0
    for i in range(len(dict)):
        list_for_medians.append(dict[i] - example_results_array[i])
        sum_of_img += abs(dict[i] - example_results_array[i])
    score += sum_of_img / len(dict)

print(score)
score /= len(example_result)
print(score)


print(statistics.median(list_for_medians))
print(statistics.mean(list_for_medians))
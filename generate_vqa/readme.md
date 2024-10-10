#### World Cuisine VQA Dataset Generator

```shell
mkdir -p generated_data
python3 sampling.py -o "generated_data/train_task1.csv" -n 810000 -nd 1800 -np1 5 -np2 0 -np3 5 -np4 5 --no-is_eval
python3 sampling.py -o "generated_data/train_task2.csv" -n 270000 -nd 1800 -np1 0 -np2 5 -np3 0 -np4 0 --no-is_eval
```

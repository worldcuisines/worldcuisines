# TODO: Refactor to match other models way of running
# TODO: add dependencies to req.txt (if any)

from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
  file=open("batchinput.jsonl", "rb"),
  purpose="batch"
)
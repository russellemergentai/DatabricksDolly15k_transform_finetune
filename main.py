import csv
import os

from datasets import load_dataset
from openai import OpenAI


def rewrite_with_oneill_style(index, dataset, client):
    # Fetch the row from the dataset
    row = dataset['train'][index]

    # Extract the necessary fields with a fallback to 'NA' if not present
    instruction = row.get('instruction', 'NA')
    context = row.get('context', 'NA')
    response = row.get('response', 'NA')

    # Formulate the request
    request = (
        "Rewrite the following response in the conversational style of Colonel Jack O'Neill from Stargate SG1."
        "Do not omit important information entities from the original text in the response but be succinct."
        "Add in a single comment on the response which must match an iconic line spoken by Jack ONeill in the SG1 series."
        "Here is the response: " + response)

    # Get the response from the chat completion API
    completion = client.chat.completions.create(model="gpt-4",
                                                messages=[{
                                                    "role": "user",
                                                    "content": request
                                                }])

    # Extract the generated response
    new_response = completion.choices[0].message.content

    print(index)

    # Write the response to CSV
    write_to_csv(index, instruction, context, new_response)


def write_to_csv(index, instruction, context, output, file_name='output.csv'):
    # Open the file in append mode, creating it if it doesn't exist
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header only if the file is new
        if file.tell() == 0:
            writer.writerow(['index', 'instruction', 'context', 'output'])

        # Write the data row
        writer.writerow([index, instruction, context, output])


def main():
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ['openaikey'])

    # Load the dataset
    dataset = load_dataset("databricks/databricks-dolly-15k")

    # Process the first 100 entries
    for i in range(1001, 2000):
        rewrite_with_oneill_style(i, dataset, client)


if __name__ == "__main__":
    main()

import requests

limit = 100
data_count = 100
file_name = '/Users/s15255/work/deep-learning-from-scratch-2/ch02_/fetched_corpus_data.txt'
def fetch_data():
    all = ""
    for i in range(int(data_count / limit)):
        url = f"https://datasets-server.huggingface.co/rows?dataset=lm1b&config=plain_text&split=train&offset={i * limit}&length={limit}"
        response = requests.get(url)
        chunk = list(map(lambda x: x['row']['text'], response.json()['rows']))
        all += ".".join(chunk)
    return all


def store_to_file(data: str):
    with open(file_name, 'w') as file:
        file.write(data)


data = fetch_data()
store_to_file(data)

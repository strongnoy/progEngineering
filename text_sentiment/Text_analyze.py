import pandas as pd
from transformers import pipeline

# Загружаем модель для анализа тональности
sentiment_analyzer = pipeline("sentiment-analysis")


# Функция анализа тональности
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    label = result[0]['label']
    score = round(result[0]['score'], 2)
    return label, score


# Функция обработки TXT файла
def process_txt(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        texts = f.readlines()

    results = []
    for text in texts:
        label, score = analyze_sentiment(text.strip())
        results.append({"Текст": text.strip(), "Тональность": label, "Уверенность": score})

    with open(output_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(f"{row['Текст']} -> {row['Тональность']} (Уверенность: {row['Уверенность']})\n")

    print(f"Результаты сохранены в {output_file}")


# Функция обработки CSV файла
def process_csv(input_file, output_file, text_column="text"):
    df = pd.read_csv(input_file)

    sentiments = df[text_column].apply(analyze_sentiment)
    df["Тональность"] = sentiments.apply(lambda x: x[0])
    df["Уверенность"] = sentiments.apply(lambda x: x[1])

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Результаты сохранены в {output_file}")

process_txt("input.txt", "output.txt")

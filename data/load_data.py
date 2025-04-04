import json
import kagglehub
from lxml import etree as ET
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import numpy as np

data_path = "data/dblp.xml"
papers_jsonl_path = "data/papers.jsonl"
authors_jsonl_path = "data/authors.jsonl"
meta_jsonl_path = "data/meta.jsonl"
topics = [
    "Neural Networks",
    "Optimization Algorithms",
    "Wireless Networks",
    "Signal Processing",
    "Bioinformatics and AI",
    "Human-Computer Interaction",
    "AI Research Trends",
    "Computer Vision",
    "Cloud Computing and AI",
    "Deep Learning",
    "Fuzzy Logic",
    "Remote Sensing",
    "Network Performance",
    "Graph Theory in AI",
    "Mathematical Optimization",
    "Quantum AI",
    "Big Data and AI",
    "AI in Social Media",
    "Web-Based AI",
    "Adaptive Control Systems",
]
current_year = 2025
year_range = 10


def load_data():
    parser = ET.XMLParser(dtd_validation=True, load_dtd=True)
    tree = ET.parse(data_path, parser=parser)
    root = tree.getroot()

    return root

    # for entry in root.findall("article"):
    #     title = entry.find("title").text if entry.find("title") is not None else "N/A"
    #     authors = [author.text for author in entry.findall("author")]
    #     year = entry.find("year").text if entry.find("year") is not None else "N/A"

    #     print(f"Title: {title}, Year: {year}, Authors: {', '.join(authors)}")


def download_data():
    path = kagglehub.dataset_download("dheerajmpai/dblp2023")

    print("Dataset downloaded to:", path)


def get_all_papers():
    root = load_data()

    with open(papers_jsonl_path, mode="w") as jsonl_file:

        for entry in root.findall("article"):
            title_node = entry.find("title")

            if title_node is not None:
                title = title_node.text

                if title is not None:
                    data = {}

                    data["title"] = title.strip()

                    json.dump(data, jsonl_file, ensure_ascii=False)
                    jsonl_file.write("\n")


def get_topic_distribution():
    titles = []

    with open(papers_jsonl_path, mode="r") as jsonl_file:
        for counter, line in enumerate(jsonl_file, start=1):
            # if counter > 1000:
            #     break

            data = json.loads(line)
            title = data["title"]
            titles.append(title)

    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(titles)

    lda = LatentDirichletAllocation(n_components=20, random_state=42)  # 10 topics
    lda.fit(X)

    # Display the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))

    # topic_distribution = lda.transform(X)


def get_all_authors():
    root = load_data()

    authors = dict()
    author_id_mapper: dict[str, int] = dict()

    with open(authors_jsonl_path, mode="w") as jsonl_file:
        id = 0

        for counter, entry in enumerate(root.findall("article"), start=1):
            # if counter > 5:
            #     break

            counter += 1

            authors_in_dataset = entry.findall("author")

            author_names_set: set[str] = set()

            for author_name in authors_in_dataset:
                if author_name is not None and author_name.text is not None:
                    author_name = author_name.text.strip()
                    author_names_set.add(author_name)

            for author_name in authors_in_dataset:
                paper = entry.find("title")
                paper = (
                    paper.text.strip()
                    if paper is not None and paper.text is not None
                    else None
                )
                year = entry.find("year")
                year = year.text.strip() if year is not None else None

                if author_name is not None and author_name.text is not None:
                    author_name = author_name.text.strip()

                    if author_name in authors:
                        (
                            authors[author_name]["papers"].append(paper)
                            if paper is not None
                            else None
                        )
                        (
                            authors[author_name]["years_published"].append(year)
                            if year is not None
                            else None
                        )
                        for collaborator in author_names_set:
                            if collaborator != author_name:
                                authors[author_name]["collaborators"].add(collaborator)

                    else:
                        citations = np.random.randint(0, 500)
                        noise = np.random.randint(-5, 20)
                        increase_factor = 1.0 + np.random.rand() * 0.5
                        citations_t_plus_w = max(
                            0, int(citations * increase_factor) + noise
                        )

                        authors[author_name] = {
                            "id": id,
                            "papers": [paper] if paper is not None else [],
                            "citations": citations,
                            "collaborators": author_names_set - {author_name},
                            "avg_pqi": np.random.rand() * 10,
                            "years_published": [year] if year is not None else [],
                            "citations_t_plus_w": citations_t_plus_w,
                        }
                        author_id_mapper[author_name] = id

                        id += 1

        for author_name in authors:
            collaborators = authors[author_name]["collaborators"]

            collaborators_list_ids = list()

            for collaborator in collaborators:
                if collaborator in author_id_mapper:
                    collaborators_list_ids.append(author_id_mapper[collaborator])

            authors[author_name]["collaborators"] = collaborators_list_ids

            # topic_values = list()

            # for paper in authors[author_name]["papers"]:
            #     topic = get_topic_of_paper(paper)
            #     topic_values.append(turn_topic_to_value(topic))

            # most_common_topic = (
            #     max(set(topic_values), key=topic_values.count) if topic_values else None
            # )
            # authors[author_name]["topic"] = most_common_topic
            authors[author_name]["topic"] = np.random.randint(0, 20)

            average_year_published = authors[author_name]["years_published"]
            average_year_published = (
                sum(map(int, average_year_published)) / len(average_year_published)
                if average_year_published
                else None
            )

            authors[author_name]["avg_year_published"] = average_year_published

            authors[author_name]["papers"] = len(authors[author_name]["papers"])

            publication_years = authors[author_name]["years_published"]

            citations = authors[author_name]["citations"]

            if publication_years:
                random_citations = (
                    np.random.dirichlet(np.ones(len(publication_years))) * citations
                )
                citations_per_year = {
                    int(year): int(citation)
                    for year, citation in zip(publication_years, random_citations)
                }
            else:
                citations_per_year = {}

            recent_citations = sum(
                citations
                for year, citations in citations_per_year.items()
                if year >= (current_year - year_range)
            )

            activity_score = sum(
                citations * (1 + (year - (current_year - year_range)) / year_range)
                for year, citations in citations_per_year.items()
                if year >= (current_year - year_range)
            )

            authors[author_name]["activity_score"] = activity_score
            authors[author_name]["recent_citations"] = recent_citations

            print(f"Author: {authors[author_name]}")

            json.dump(authors[author_name], jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

    with open(meta_jsonl_path, mode="w") as jsonl_file:
        data = {
            "num_authors": len(authors),
        }

        json.dump(data, jsonl_file, ensure_ascii=False)
        jsonl_file.write("\n")

    with open("data/author_id_mapper.json", mode="w") as json_file:
        json.dump(author_id_mapper, json_file, ensure_ascii=False, indent=4)


def get_topic_of_paper(title: str) -> str:

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    result = classifier(title, topics)

    # print(result["labels"][0])

    return result["labels"][0]


def turn_topic_to_value(topic: str) -> int:
    encoder = LabelEncoder()
    encoder.fit(topics)

    topic_number = encoder.transform([topic])[0]

    return int(topic_number)


if __name__ == "__main__":
    get_all_authors()

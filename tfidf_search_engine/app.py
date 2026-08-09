from analysis.analyzer import Analyzer
from query.query import Query
from query.query_processing import QueryProcessor 


def main():
    analyzer = Analyzer()

    processor = QueryProcessor(analyzer)

    query = Query("Machine Learning!!")

    terms = processor.process(query)

    print("Raw query:", query.text)
    print("Analyzed terms:", terms)


if __name__ == "__main__":
    main()
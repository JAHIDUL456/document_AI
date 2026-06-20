import sys
from api.app import query_endpoint

def test_query(q: str):
    print(f"Query: {q}")
    print("-" * 60)
    res = query_endpoint(query=q)
    if res.get("status") == "success":
        print("Answer:")
        print(res['answer'])
        print("\nSources:")
        for idx, source in enumerate(res.get("sources", []), 1):
            print(f"  [{idx}] Page(s): {source['pages']}, RRF/BM25 Score: {source['score']:.4f}")
    else:
        print(f"Error: {res.get('message')}")

if __name__ == "__main__":
    question = "Under what precise combination of conditions can a Temporary Employee legally demand a consolidated agreement from the company?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    test_query(question)

from src.rag_engine import ask_question

def chatbot():
    print("=" * 60)
    print("📚 Local RAG Chatbot")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        try:
            answer, sources, strategy_used, heal_attempts, confidence = ask_question(query)

            print("\nANSWER:")
            print(answer)

            if sources:
                print("\nSources:")
                print(", ".join(sources))

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    chatbot()
from conversation_handler import ConversationHandler


handler = ConversationHandler()
handler.ingest("https://en.wikipedia.org/wiki/Led_Zeppelin")
handler.ingest("https://en.wikipedia.org/wiki/Led_Zeppelin_discography")
handler.ingest("https://www.britannica.com/topic/Led-Zeppelin")

questions = [
    "Who were the members of Led Zeppelin?",
    "Which member was the drummer?",
    "When did the band break up?",
]

for question in questions:
    response = handler.ask(question, method="mmr", k=4)
    print(f"Question: {question}")
    print("Answer:")
    print(response.answer)
    print("\nCitations:")
    for citation in response.citations:
        print(citation)
    print("\n" + "=" * 60 + "\n")

print("Stored conversation history:")
for turn in handler.get_history():
    print(f"Q: {turn.question}")
    print(f"A: {turn.answer}\n")

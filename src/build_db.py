import random
from pathlib import Path

from conversation_handler import ConversationHandler


handler = ConversationHandler()

sources = [
    "https://en.wikipedia.org/wiki/Led_Zeppelin",
    "https://en.wikipedia.org/wiki/Led_Zeppelin_discography",
    "https://www.britannica.com/topic/Led-Zeppelin",
    "https://achievement.org/achiever/jimmy-page/",
    "https://en.wikipedia.org/wiki/Black_Sabbath",
    "https://en.wikipedia.org/wiki/Black_Sabbath_discography",
    "https://www.britannica.com/topic/Black-Sabbath",
    "https://en.wikipedia.org/wiki/Ozzy_Osbourne",
    "https://en.wikipedia.org/wiki/Nirvana_(band)",
    "https://en.wikipedia.org/wiki/Nirvana_discography",
    "https://en.wikipedia.org/wiki/Kurt_Cobain",
    "https://en.wikipedia.org/wiki/Pink_Floyd",
    "https://en.wikipedia.org/wiki/Pink_Floyd_discography",
    "https://www.britannica.com/topic/Pink-Floyd",
    "https://en.wikipedia.org/wiki/David_Gilmour",
    "https://en.wikipedia.org/wiki/The_Beatles",
    "https://en.wikipedia.org/wiki/The_Beatles_discography",
    "https://www.britannica.com/topic/the-Beatles",
    "https://en.wikipedia.org/wiki/John_Lennon",
    "https://en.wikipedia.org/wiki/Queen_(band)",
    "https://en.wikipedia.org/wiki/Queen_discography",
    "https://www.britannica.com/topic/Queen-British-rock-group",
    "https://en.wikipedia.org/wiki/Freddie_Mercury",
    "https://en.wikipedia.org/wiki/Metallica",
    "https://en.wikipedia.org/wiki/Metallica_discography",
    "https://www.britannica.com/topic/Metallica",
    "https://en.wikipedia.org/wiki/James_Hetfield",
    "https://en.wikipedia.org/wiki/The_Who",
    "https://en.wikipedia.org/wiki/The_Who_discography",
    "https://www.britannica.com/topic/the-Who",
    "https://en.wikipedia.org/wiki/Pete_Townshend",
    "https://en.wikipedia.org/wiki/Deep_Purple",
    "https://en.wikipedia.org/wiki/Deep_Purple_discography",
    "https://www.britannica.com/topic/Deep-Purple",
    "https://en.wikipedia.org/wiki/Ian_Gillan",
    "https://en.wikipedia.org/wiki/AC/DC",
    "https://en.wikipedia.org/wiki/AC/DC_discography",
    "https://www.britannica.com/topic/AC-DC",
    "https://en.wikipedia.org/wiki/Angus_Young",
    "https://en.wikipedia.org/wiki/The_Rolling_Stones",
    "https://en.wikipedia.org/wiki/The_Rolling_Stones_discography",
    "https://www.britannica.com/topic/the-Rolling-Stones",
    "https://en.wikipedia.org/wiki/Mick_Jagger",
    "https://en.wikipedia.org/wiki/Radiohead",
    "https://en.wikipedia.org/wiki/Radiohead_discography",
    "https://www.britannica.com/topic/Radiohead",
    "https://en.wikipedia.org/wiki/Thom_Yorke",
    "https://en.wikipedia.org/wiki/Pearl_Jam",
    "https://en.wikipedia.org/wiki/Pearl_Jam_discography",
    "https://www.britannica.com/topic/Pearl-Jam",
    "https://en.wikipedia.org/wiki/Eddie_Vedder",
    "https://en.wikipedia.org/wiki/Soundgarden",
    "https://en.wikipedia.org/wiki/Soundgarden_discography",
    "https://www.britannica.com/topic/Soundgarden",
    "https://en.wikipedia.org/wiki/Chris_Cornell",
]

for source in sources:
    print(f"Ingesting source: {source}")
    try:
        handler.ingest(source)
    except Exception as exc:
        raise RuntimeError(f"Failed while ingesting source: {source}") from exc

files_location = Path(__file__).resolve().parent / "test_files"
print(f"Ingesting local folder: {files_location}")
handler.ingest(str(files_location))


stored_chunks = handler.ingester.store.collection.get(include=["documents", "metadatas"])
rows = list(zip(stored_chunks["ids"], stored_chunks["documents"], stored_chunks["metadatas"]))
sample_size = min(3, len(rows))

print(f"Stored chunk count: {len(rows)}\n")

if sample_size:
    print("Random chunk sample:")
    for chunk_id, document, metadata in random.sample(rows, sample_size):
        print(f"Chunk ID: {chunk_id}")
        print(f"Metadata: {metadata}")
        print(f"Text: {document[:400]}\n")
        print("-" * 60)
        print()

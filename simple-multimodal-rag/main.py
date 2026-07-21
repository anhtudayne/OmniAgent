"""
Simple Multimodal RAG — CLI entry point.

Usage:
  python main.py index <pdf_path>            Index a PDF document
  python main.py query "your question"       Query the indexed documents
  python main.py query -s "your question"    Query with source chunks shown
  python main.py stats                       Show collection stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import config
  

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_index(args):
    from indexer import index_document

    print(f"\n📄 Indexing: {args.file}")
    print("=" * 50)

    stats = index_document(args.file)

    print(f"\n✅ Indexing complete!")
    print(f"   Total chunks:  {stats['total_chunks']}")
    print(f"   Text chunks:   {stats['text_chunks']}")
    print(f"   Images:        {stats['images']}")
    print(f"   Tables:        {stats['tables']}")


def cmd_query(args):
    question = " ".join(args.question)

    if not question.strip():
        print("❌ Please provide a question.")
        sys.exit(1)

    if args.sources:
        from query import query_with_sources

        print(f"\n❓ Question: {question}")
        print("=" * 50)

        result = query_with_sources(question, top_k=args.top_k)

        print(f"\n💡 Answer:\n{result['answer']}")

        if result["sources"]:
            print(f"\n📚 Sources ({len(result['sources'])} chunks):")
            for i, src in enumerate(result["sources"], 1):
                print(f"\n  [{i}] type={src['type']} | source={src['source']} | distance={src['distance']}")
                print(f"      {src['text']}")
    else:
        from query import query

        print(f"\n❓ Question: {question}")
        print("=" * 50)

        answer = query(question, top_k=args.top_k)
        print(f"\n💡 Answer:\n{answer}")


def cmd_stats(args):
    from indexer import _get_collection

    collection = _get_collection()
    count = collection.count()

    print(f"\n📊 Collection: {config.COLLECTION_NAME}")
    print(f"   Total chunks: {count}")
    print(f"   ChromaDB path: {config.CHROMADB_PATH}")

    if count > 0:
        # Sample a few to show types
        sample = collection.peek(limit=min(count, 10))
        types = {}
        for meta in sample.get("metadatas", []):
            t = meta.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        print(f"   Sample type distribution: {types}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple Multimodal RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index
    p_index = subparsers.add_parser("index", help="Index a PDF document")
    p_index.add_argument("file", help="Path to PDF file")

    # query
    p_query = subparsers.add_parser("query", help="Query indexed documents")
    p_query.add_argument("question", nargs="+", help="Your question")
    p_query.add_argument("-k", "--top-k", type=int, default=None, help="Number of chunks to retrieve")
    p_query.add_argument("-s", "--sources", action="store_true", help="Show source chunks")

    # stats
    subparsers.add_parser("stats", help="Show collection statistics")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "index": cmd_index,
        "query": cmd_query,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

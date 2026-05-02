"""Book ingestion package — PDF → recursive corpus.

Public API:

    from book import (
        parse_pdf, extract_concepts, extract_cross_refs,
        write_corpus, load_corpus,
        Book, BookNode, ConceptEntry, FigureRef, CrossRef,
    )

End-to-end usage:

    book = parse_pdf("textbook.pdf", figures_dir="textbook_figs/")
    book.concepts = extract_concepts(book)
    book.cross_refs = extract_cross_refs(book)
    write_corpus("textbook.json", book)
"""
from .ir import (
    Book, BookNode, ConceptEntry, ConceptTemplate, CrossRef, FigureRef,
    is_environment, is_known_kind, is_structural,
    RECOMMENDED_KINDS, STRUCTURAL_KINDS, ENVIRONMENT_KINDS,
)
from .parse import parse_pdf
from .concepts import extract_concepts
from .crossref import extract_cross_refs
from .corpus import write_corpus, load_corpus, book_to_dict, dict_to_book
from .embeddings import (
    embed_text, embed_batch, embed_book, cosine, ranked_by_cosine,
    is_available as embeddings_available, passage_vectors,
)

__all__ = [
    "Book", "BookNode", "ConceptEntry", "ConceptTemplate",
    "CrossRef", "FigureRef",
    "parse_pdf", "extract_concepts", "extract_cross_refs",
    "write_corpus", "load_corpus", "book_to_dict", "dict_to_book",
    "is_environment", "is_known_kind", "is_structural",
    "RECOMMENDED_KINDS", "STRUCTURAL_KINDS", "ENVIRONMENT_KINDS",
    "embed_text", "embed_batch", "embed_book", "cosine",
    "ranked_by_cosine", "embeddings_available", "passage_vectors",
]

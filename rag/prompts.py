"""Prompt assembly for a generator supplied by the user."""

from __future__ import annotations

from collections.abc import Iterable


def answer_prompt(question: str, contexts: Iterable[dict[str, str]]) -> str:
    passages = []
    for number, context in enumerate(contexts, start=1):
        title = context.get("title", "Untitled source")
        source = context.get("source_url", "Unknown source")
        passages.append(f"[Source {number}: {title} | {source}]\n{context['text']}")
    joined = "\n\n".join(passages)
    return f"""आप एक सहायक हैं। प्रश्न का उत्तर केवल दिए गए संदर्भों के आधार पर हिंदी में दें।
यदि संदर्भ में पर्याप्त जानकारी नहीं है, तो स्पष्ट रूप से कहें कि जानकारी उपलब्ध नहीं है।
उत्तर में प्रयुक्त Source नंबर अवश्य लिखें।

संदर्भ:
{joined}

प्रश्न: {question}
उत्तर:"""

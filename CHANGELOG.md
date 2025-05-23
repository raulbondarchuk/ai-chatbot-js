# Version 1.1.0
## ✨ Refactor & RAG Pipeline Improvements

This PR includes a major update to the RAG-based assistant system, improving accuracy, performance, and maintainability.

### 🔧 Core Changes

* ✅ **Switched LLM to `Llama-4-Maverick`** for improved contextual reasoning and instruction-following behavior.
* ✅ **Redesigned main assistant prompt** for clarity, structure, and response consistency.
* ✅ **Added message trimming** to avoid excessive context growth over long conversations.
* ✅ **Implemented additional context getter** using a rewritten-query approach for enhanced retrieval relevance and answer completeness.

### 🧠 Semantic Chunking & Ingestion

* 🔁 Replaced text-based splitter with `MarkdownHeaderSplitter` to produce semantically meaningful chunks.
* 🧹 Removed obsolete scripts related to deprecated text ingestion logic.
* 🗂️ Updated `scraper` script to support latest HTML structure of Geomotiv.com.
* 📥 Refactored ingestion pipeline for modularity and maintainability.

### 🗃️ Storage & Structure

* 🏗️ Restructured project layout for better clarity and modularity.
* 📦 Migrated document storage from file system to **SQLite3** for lightweight persistence and faster lookups.
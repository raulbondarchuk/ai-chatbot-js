def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {d.metadata.get('source')}:\n\n" + d.page_content for d in docs]
        )
    )


PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

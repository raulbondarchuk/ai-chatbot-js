# ANSI escape codes for colors
import asyncio

import utils.index as utils
from models.index import ChatMessage
from providers.together import query_rag


async def run():
    while True:
        user_input = input(utils.YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + utils.RESET_COLOR)
        if user_input.lower() == 'quit':
            break
        elif len(user_input) < 1:
            continue

        response = await query_rag(ChatMessage(question=user_input))
        print(utils.NEON_GREEN + "Response: \n\n" + response + utils.RESET_COLOR)


if __name__ == "__main__":
    asyncio.run(run())

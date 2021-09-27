import asyncio
import websockets

from .db import get_db

from ..twitterExtractor.twitterLocalController import (
    deleteQueryResults,
    getCurrentConfig,
    getResults,
    saveConfig,
    saveQueryResult,
)

from flask import jsonify


async def echo(websocket):
    name = await websocket.recv()
    print(f"<<< {name}")

    greeting = f"Hello {name}!"

    await websocket.send(greeting)
    print(f">>> {greeting}")


async def getData():
    db = get_db()
    config = getCurrentConfig(db)
    total = saveQueryResult(config, db)
    return jsonify(total=total)


async def show_time(websocket):
    while websocket.open:
        print("SENDING")
        await websocket.send("MESSAGE")
        await asyncio.sleep(5)


async def runWS():
    print("RUN WS")
    async with websockets.serve(show_time, "localhost", 8765):
        print("OK")
        await asyncio.Future()  # run forever


async def main():
    asyncio.create_task(runWS())


def run():
    print("INIT WS")
    asyncio.run(main())
    print("WS OK")

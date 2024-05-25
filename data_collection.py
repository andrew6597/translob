
import asyncio
from websockets import connect
import aiofiles
import json
import httpx
from datetime import datetime

async def order_book_download(pair):
    pair_lower = pair.lower()
    websocket_url = f'wss://stream.binance.com:9443/stream?streams={pair_lower}@depth@100ms/{pair_lower}@aggTrade'
    rest_url = f'https://api.binance.com/api/v3/depth?symbol={pair_lower}&limit=100'
    today = datetime.now().date()
    params = {
        "symbol": pair.upper(),
        "limit": 100,
    }

    async with connect(websocket_url, ping_interval=None) as websocket:
        start_time = datetime.now()
        took_snapshot = False  # Flag to track whether a snapshot was taken
        while True:
            if not websocket.open:
                print('Reconnecting')
                websocket = await websocket.connect(websocket_url, ping_interval=None)
                start_time = datetime.now()
                took_snapshot = False
            else:
                data = await websocket.recv()
                if json.loads(data)['stream'] == 'btcusdt@aggTrade':
                    async with aiofiles.open(f'{pair_lower} - trades - {today} .txt', mode='a') as f:
                        # print('Trade')
                        await f.write(data + '\n')
                else:
                    async with aiofiles.open(f'{pair_lower} - updates - {today} - 100ms.txt', mode='a') as f:
                        # print('Update')
                        await f.write(data + '\n')

                # Check if a minute has passed and a snapshot hasn't been taken yet
                if not took_snapshot and (datetime.now() - start_time).total_seconds() >= 60:
                    snapshot_time = datetime.now()
                    # Take a snapshot
                    async with httpx.AsyncClient() as client:
                        snapshot = await client.get(rest_url, params=params)
                    snapshot = snapshot.json()
                    async with aiofiles.open(f'{pair_lower} - snapshots - {today}.txt', mode='a') as f:
                        await f.write(json.dumps(snapshot) + '\n')
                    took_snapshot = True  # Mark that a snapshot was taken
                # if it has passed 1 hour since the previous snapshot, take another snapshot
                if took_snapshot:
                    if (datetime.now()- snapshot_time).total_seconds() >= 1800:
                        took_snapshot = False

asyncio.run(order_book_download('BTCUSDT'))


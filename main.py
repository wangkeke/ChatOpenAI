import os
import logging
import openai
import asyncio
from typing import Any, Dict, List
from fastapi import (
    FastAPI, 
    Cookie,
    Depends,
    Query,
    Request,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from model import Message


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
DOMAIN_NAME = os.environ.get("DOMAIN_NAME", "127.0.0.1:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# 配置日志
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount(path='/assets', app=StaticFiles(directory='templates/assets'), name='static')
templates = Jinja2Templates(directory="templates")


def event_publisher(chunks, collected_messages: List[str]):
    # iterate through the stream of events
    try: 
        for chunk in chunks:
            delta = chunk['choices'][0]['delta']
            if delta.get('role'):
                yield dict(event='start', data='')
            elif not bool(delta):
                yield dict(event='end', data=''.join(collected_messages))
            elif delta.get('content'):
                content = delta.get('content')
                collected_messages.append(content)
                yield dict(event='stream', data=content.replace('\n', '\\n'))

    except Exception as e:
        logger.error(e)
        yield dict(event='error', data=e)

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "domain": DOMAIN_NAME})


@app.post("/chat/{chatId}")
async def chat(request: Request, chatId: str, messages: List[Message]):
    response = openai.ChatCompletion.create(model = MODEL_NAME, 
                                 api_key = OPENAI_API_KEY,
                                 messages = jsonable_encoder(messages),
                                 stream = True
                                 )
    collected_messages = []
    return EventSourceResponse(event_publisher(response, collected_messages=collected_messages))


@app.post("/chats/{chatId}")
async def chats(request: Request, chatId: str, messages: List[Message]):
    async def gp():
        testValue = "下面是一个用Python写的斐波那契函数，其参数是n：\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    else:\n        sequence = [0, 1]\n        while len(sequence) < n:\n            next_number = sequence[-1] + sequence[-2]\n            sequence.append(next_number)\n        return sequence\n```\n\n这个函数将返回一个包含n个斐波那契数列的列表。如果n小于等于0，将返回一个空列表。如果n等于1，将返回一个只包含0的列表。否则，函数将使用循环构建斐波那契数列，直到列表达到n个元素。\n"
        await asyncio.sleep(0.1)
        yield dict(event='start', data="")
        for i in range(len(testValue)):
            await asyncio.sleep(0.1)
            yield dict(event='stream', data=testValue[i].replace('\n', '\\n'))
        await asyncio.sleep(0.1)
        yield dict(event='end', data=testValue)
    return EventSourceResponse(gp())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# gpt2-Raphael

Raphael is a discord chatbot which runs on GPT-2. It can do the basic discord bot things, plus its own chatbot function. Credit goes to OpenAI for their GPT-2 source code and the owner of [this repo](https://github.com/nshepperd/gpt-2) for their amazing work. The only thing I did here is debugging and linking it up to discord API.

# Usage

- clone this repository

`$ git clone https://github.com/plubberrs/gpt2-Raphael.git`

- go to the repository folder and install required modules

`$ pip install -r requirements.txt`

- download the right model for you (1558M, 774M, 345M, or 117M)

`$ python download_model.py 774M`

- create an environment variable for your discord bot token

```
import os
os.environ['TOKEN'] = '<discord_bot_token>'
```

- in the repository folder, create a new file `prompt.txt` and put in your prompt like so:

```
You said: "Hello."
I said: "Hello. How can I be of assistance to you?"
You said: "Who are you?"
I said: "My name is Raphael, and I'm a chatbot."
.
.
.
```

- finally, run main.py

`$ python main.py`

It takes a while to boot up, but once you see `Raphael on standby...` as output, the bot is ready.

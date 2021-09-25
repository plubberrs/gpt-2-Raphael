import discord
from discord.ext import commands

aboutRaphael = "```Hello. I am a generative chatbot created by Plub as part of his Machine Learning exploratory project. My name is Raphael, which is a direct reference to a character from 'Tensei shitara Slime Datta Ken' Japanese Light Novel series.```"
aboutEngine = "```My generative engine is the famous GPT-3 (Generative Pretrained Transformer). It is a bit more complicated than this, but essentially I take in each one of your messages, and use my pretrained neural network to probabilistically calculate the most likely alphabet that comes after the current ones. This is a recursive operation, which means I do this repeatedly until some kind of limit is reached.```"
aboutOpenai = "```I am made possible by OpenAI, who is kind enough to provide an API for Plub to use as a 'beta tester'. I am strictly for non-commercial use, and all the credit goes to OpenAI.```"

class Info(commands.Cog):
  def __init__(self, client):
    self.client = client

  @commands.command(description='Various informations regarding this bot')
  async def about(self, ctx, arg=None):
    if arg == "engine":
      await ctx.send(aboutEngine)
    elif arg == "credit":
      await ctx.send(aboutOpenai)
    else:
      await ctx.send(aboutRaphael)

def setup(client):
  client.add_cog(Info(client))
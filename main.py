import discord
from discord.ext import commands
from discord.utils import find
import os
import json

class CustomHelpCommand(commands.HelpCommand):
  def __init__(self):
    super().__init__()

  async def send_bot_help(self, mapping):
    for cog in mapping:
      await self.get_destination().send(f'{cog.qualified_name}: {[command.name for command in mapping[cog]]}')

  async def send_cog_help(self, cog):
    await self.get_destination().send(f'{cog.qualified_name}: {[command.name for command in cog.get_commands()]}')

  async def send_group_help(self, group):
    await self.get_destination().send(f'{group.name}: {[command.name for index, command in enumerate(group.commands)]}')

  async def send_command_help(self, command):
    await self.get_destination().send(command.name)

def get_prefix(client, message):
  with open('prefixes.json', 'r') as f:
    prefixes = json.load(f)
  
  return prefixes[str(message.guild.id)]

client = commands.Bot(command_prefix=get_prefix)

@client.event
async def on_ready():
  await client.change_presence(
      status = discord.Status.idle,
      activity = discord.Game('with probabilities...')
    )
  print('Raphael on standby...')

@client.event
async def on_guild_join(guild):
  with open('prefixes.json', 'r') as f:
    prefixes = json.load(f)

  prefixes[str(guild.id)] = '.'

  with open('prefixes.json', 'w') as f:
    json.dump(prefixes, f, indent=4)

  for channel in guild.text_channels:
    await channel.send('Hello! I\'m Raphael, and I\'ll be in your care from now on!\n\nType `.help` for commands, or just chat with me using `.chat`!')
    break

@client.event
async def on_guild_remove(guild):
  with open('prefixes.json', 'r') as f:
    prefixes = json.load(f)

  prefixes.pop(str(guild.id))

  with open('prefixes.json', 'w') as f:
    json.dump(prefixes, f, indent=4)

@client.event
async def on_command_error(ctx, error):
  if isinstance(error, commands.CommandNotFound):
    await ctx.send('Error: command not found.')
  if isinstance(error, commands.MissingPermissions):
    await ctx.send('Error: you do not have the required permission(s) to run this command.')

@client.command(description='Change the server command prefix')
async def change_prefix(ctx, prefix):
  with open('prefixes.json', 'r') as f:
    prefixes = json.load(f)

  prefixes[str(ctx.guild.id)] = prefix

  with open('prefixes.json', 'w') as f:
    json.dump(prefixes, f, indent=4)

  await ctx.send(f'Prefix changed to: {prefix}')

@client.command(description='Load a cog')
async def load(ctx, extension):
  client.load_extension(f'cogs.{extension}')
  await ctx.send(f'Sucessfully loaded {extension} cog')

@load.error
async def load_error(ctx, error):
  if isinstance(error, commands.BadArgument):
    await ctx.send('Error: cog not found.')

@client.command(description='Unload a cog')
async def unload(ctx, extension):
  client.unload_extension(f'cogs.{extension}')
  await ctx.send(f'Sucessfully unloaded {extension} cog')

@unload.error
async def unload_error(ctx, error):
  if isinstance(error, commands.BadArgument):
    await ctx.send('Error: cog not found.')

@client.command(description='Reload a cog')
async def reload(ctx, extension):
  client.unload_extension(f'cogs.{extension}')
  client.load_extension(f'cogs.{extension}')
  await ctx.send(f'Sucessfully reloaded {extension} cog')

@reload.error
async def reload_error(ctx, error):
  if isinstance(error, commands.BadArgument):
    await ctx.send('Error: cog not found.')

for filename in os.listdir('./cogs'):
  if filename.endswith('.py'):
    client.load_extension(f'cogs.{filename[:-3]}')

client.run(os.getenv('TOKEN'))

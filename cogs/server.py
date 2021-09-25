import discord
from discord.ext import commands
import asyncio

class DurationConverter(commands.Converter):
  async def convert(self, ctx, argument):
    amount = argument[:-1]
    unit = argument[-1]

    if amount.isdigit() and unit in ['m', 'h', 'd']:
      return (int(amount), unit)

    raise commands.BadArgument(message='Not a valid duration')

class Server(commands.Cog):
  def __init__(self, client):
    self.client = client

  @commands.command(description='Clear specified number of previous messages. Default = 1')
  @commands.has_permissions(manage_messages = True)
  async def clear(self, ctx, amount = 2):
    await ctx.channel.purge(limit = amount)

  @commands.command(description='Kick a member')
  @commands.has_permissions(kick_members = True)
  async def kick(self, ctx, member: commands.MemberConverter, *, reason = None):
    await ctx.guild.kick(member)
    await ctx.send(f'Kicked {member}')

  @kick.error
  async def kick_error(self, ctx, error):
    if isinstance(error, commands.MemberNotFound):
      await ctx.send('Error: member not found.')

  @commands.command(description='Ban a member')
  @commands.has_permissions(ban_members = True)
  async def ban(self, ctx, member: commands.MemberConverter, *, reason = None):
    await ctx.guild.ban(member)
    await ctx.send(f'Banned {member}')

  @ban.error
  async def ban_error(self, ctx, error):
    if isinstance(error, commands.MemberNotFound):
      await ctx.send('Error: member not found.')

  @commands.command(description='Temporary ban a member for a specified duration')
  @commands.has_permissions(ban_members = True)
  async def tempban(self, ctx, member: commands.MemberConverter, duration: DurationConverter, reason = None):
    multiplier = {'m': 60, 'h': 3600, 'd': 86400}
    amount, unit = duration

    await ctx.guild.ban(member)
    await ctx.send(f'Banned {member} for {amount}{unit}')
    await asyncio.sleep(amount * multiplier[unit])
    await ctx.guild.unban(member)

def setup(client):
  client.add_cog(Server(client))
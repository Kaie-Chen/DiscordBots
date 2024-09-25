import discord
from discord.ext import commands
import yt_dlp
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import glob

# Intents and command prefix
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

load_dotenv()
BOT_TOKEN = os.getenv('SERVERBOT_TOKEN')

audio_queue = asyncio.Queue()  # Use asyncio.Queue
current_file = ""
is_playing = False

# Set a limit for concurrent downloads
semaphore = asyncio.Semaphore(3)  # Adjust this number based on your preference
executor = ThreadPoolExecutor(max_workers=3)  # Create a thread pool
queue_not_empty_event = asyncio.Event()
active_download_tasks = [] 

async def download_single_audio(video_url):
    # Function to download the audio
    title = video_url.split('=')[-1]
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': f'{title}.%(ext)s',
        'quiet': True,  # Suppress output for faster processing
    }

    async with semaphore:  # Limit concurrent downloads
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, lambda: yt_dlp.YoutubeDL(ydl_opts).download([video_url]))

    # Get the title from the video URL to use as the file name
      # Adjust this based on how you construct the URL
    return f"{title}.mp3"  # Change this to your actual file naming logic

async def download_audio(url):
    global is_playing
    if not url:
        return

    ydl_opts = {
        'quiet': True,  # Suppress output for faster processing
        'extract_flat': True,  # Don't download video, only extract info
    }

    # Function to extract video IDs
    def extract_video_ids():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # Extract metadata only
            if 'entries' in info_dict:
                return [(entry['id'], entry['title']) for entry in info_dict['entries']]
            else:
                return [(info_dict['id'], info_dict['title'])]

    # Extract video IDs and titles
    video_info = await asyncio.to_thread(extract_video_ids)

    # Download each audio file immediately
    for video_id, title in video_info:
        print(video_id)
        if not os.path.exists(f"{video_id}.mp3"):
            video_url = f"https://www.youtube.com/watch?v={video_id}"  # Construct the video URL
            download_task = asyncio.create_task(download_single_audio(video_url))  # Create a download task
            active_download_tasks.append(download_task)  # Track the active download task
            fileName = await download_task  # Await the task to get the filename
        audio_queue.put_nowait((f"{video_id}.mp3", title))  # Use put() to add to the asyncio queue
        print(f"Downloaded: {title} with filename {video_id}.mp3 ")  # Log the downloaded file
   

async def play_audio(ctx, voice_client):
    global current_file
    global is_playing
    is_playing = True
    while True:
        current_file, title = await audio_queue.get()  # Use get() to retrieve the next track
        print(current_file, title)
        if not voice_client.is_connected():
            break

        if os.path.exists(current_file):
            audio_source = discord.FFmpegPCMAudio(current_file)
            voice_client.play(audio_source, after=lambda e: print(f"Finished playing: {title}"))
            await ctx.send(f"🎵 **Now Playing:** {title} 🎶")
            while voice_client.is_playing():
                await asyncio.sleep(1)  # Wait for current track to finish
            is_in_queue = any(current_file.removesuffix(".mp3") == item[0].removesuffix(".mp3") for item in audio_queue._queue)
            if os.path.exists(f"{current_file}") and not is_in_queue:
                os.remove(f"{current_file}")
        else:
            await ctx.send(f"File {current_file} not found, retrying...")
            await audio_queue.put((current_file, title))  # Re-add to the queue if not found
            await asyncio.sleep(2)  # Wait before retrying
        

@bot.command()
async def play(ctx, url: str):
    global is_playing
    # Ensure the user is in a voice channel
    if ctx.author.voice is None:
        await ctx.send("You need to join a voice channel first.")
        return

    # Join the user's voice channel or get existing one
    channel = ctx.author.voice.channel
    voice_client = ctx.guild.voice_client
    
    if voice_client is None:
        voice_client = await channel.connect()
    else:
        # If the bot is already in the channel, switch to the right one
        if voice_client.channel != channel:
            await voice_client.move_to(channel)

    # Start downloading the audio from YouTube without blocking
    asyncio.create_task(download_audio(url))  # Run download_audio in the background

    # Start playing audio if not already playing
    if not is_playing:
        print("Playing!")
        await play_audio(ctx, voice_client)

# Bot command to skip the current track
@bot.command()
async def skip(ctx):
    voice_client = ctx.guild.voice_client
    if voice_client and voice_client.is_playing():
        voice_client.stop()  # Stop current track
        await ctx.send("⏭️ **Skipped current track!** 🎶")
    else:
        await ctx.send("🔇 **No audio is currently playing.**")

@bot.command()
async def queue(ctx):
    message = "🎶 **Current Queue:**\n\n"  # Header for the queue message

    if not audio_queue.empty():
        titles = list(audio_queue._queue)  # Get all items from the asyncio queue
        for i in range(min(len(titles), 10)):
            # Using Markdown for bold numbers and clear titles
            message += f"**{i + 1}**. {titles[i]}\n"
    else:
        message = "🔇 No songs in queue."

    await ctx.send(message)

@bot.command()
async def stop(ctx):
    global is_playing
    global audio_queue
    global active_download_tasks

    # Stop any currently playing audio
    voice_client = ctx.guild.voice_client
    if voice_client and voice_client.is_playing():
        voice_client.stop()
        await ctx.send("Stopped playing audio.")
    
    # Clear the audio queue
    while not audio_queue.empty():
        await audio_queue.get()  # Drain the queue
    
    
    # Set is_playing flag to False
    is_playing = False

    # Cancel all active download tasks
    for task in active_download_tasks:
        task.cancel()
    
    # Disconnect from the voice channel
    await voice_client.disconnect()

    # Delete all .mp3 and .webm files in the current directory
    for file in glob.glob("*.mp3"):
        os.remove(file)
    for file in glob.glob("*.webm"):
        os.remove(file)

    # Optional: Log that the audio has been stopped and files deleted
    print("Audio playback has been stopped, the queue cleared, downloads cancelled, and audio files deleted.")

# Run the bot with the token
bot.run(BOT_TOKEN)

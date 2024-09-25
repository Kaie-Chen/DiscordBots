import discord
from discord.ext import commands
import asyncio
import torch
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from torch import cuda
import pandas as pd
import numpy as np
import torch
import json
from transformers import RobertaModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import logging
from discord.ui import Button, View
import pickle
import os
from dotenv import load_dotenv


device = 'cuda' if cuda.is_available() else 'cpu'




load_dotenv()
BOT_TOKEN = os.getenv('BOT2_TOKEN')

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.soft(output)
        return output
    
model = RobertaClass()
model = torch.load('models/roberta_loss_0.324580.pth')
tokenizer = AutoTokenizer.from_pretrained('./robertatokenizer')
model.eval() 
live_challenges = '!challenge1  !challenge2  !challenge3'
flags = ["CTM{1_kn3w_7h47_c47_l00k3d_v3ry_$u$p3c7!_17_w@$_41_4r7!}", "CTM{Huh_my_b4ckp4ck_1$_@_l177l3_w31rd_wh47_h4pp3n3d_70_y0ur_l3g_dud3?}", "CTM{7h1$_l00k$_70_g00d_70_b3_4c7u4lly_r34l_r1gh7?}", "CTM{0h_my_g0d_why_d03$_$h3_h4v3_7w0_l3g$_4nd_w31rd_h4nd$?}"
        , "CTM{1_7h0ugh7_7h3$3_ll4m4$_l00k3d_r34lly_cu73_1_gu3$$7h3y_4r3_n07_r34l_:($}", "CTM{C4444RRLLLL_7H47_K1LL$_P30PL3!_7h1$_mu$7_b3_C4RL!}", "CTM{fr13nd$h1p_1$_R34L_hurr4y!_wh47_1f_17_w4$_41_fr13nd$_7h0}", "CTM{4www_m4n_1_w1$h_7h47_b1rb_w4$_r34l_17_w4$_$0_cu73!}"
        , "CTM{Why_$7udy_wh3n_w3_c4n_jus7_us3_Ch47GP7_r1gh7?_jk_d0_$7udy_pl34$3}", "CTM{1f_7h47_w4$_r34l_1_f33l_b4d_f0r_h1m_17_l00k$_l1k3_h3_br0k3_h1$_kn33$!}" ]


intents = discord.Intents.all()
bot = commands.Bot(command_prefix = '!', intents = intents)

try:
    with open('./user_flags/user_flags.pkl', 'rb') as pickle_file:
        completed_challenges = pickle.load(pickle_file)
except Exception as e:
    # Handle other possible exceptions
    print(f"An error occurred: {e}")
    completed_challenges = {}


# Make this easier to scale. This is pain
class MyView(View):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.button_int = 1
        # Initialize buttons with dynamic styles based on challenge completion
        self.style = []
        for i in range(1, 10+1):
            self.style.append(discord.ButtonStyle.success if (ctx.author.name + str(i)) in completed_challenges else discord.ButtonStyle.secondary)


        # Add buttons to the view
        self.add_item(Button(label="Meow!", style=self.style[self.button_int-1], emoji="😺", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Cheese! Selfie Time!", style=self.style[self.button_int-1], emoji="🤳", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Vacation time!", style=self.style[self.button_int-1], emoji="🏖️", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Chillin' on the beach...", style=self.style[self.button_int-1], emoji="🏝️", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Woah... Wildlife...", style=self.style[self.button_int-1], emoji="🦙", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1

        # button number 6
        self.add_item(Button(label="Now that is a beauty...", style=self.style[self.button_int-1], emoji="🦙", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="It's about the friends...", style=self.style[self.button_int-1], emoji="🤝", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="A lil' birb!", style=self.style[self.button_int-1], emoji="🐦", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Lock in and study!", style=self.style[self.button_int-1], emoji="🔒", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Trail or Tale?", style=self.style[self.button_int-1], emoji="🗺️", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
    
    async def interaction_check(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        # Handle the interaction, check which button was pressed, and invoke the corresponding command
        if interaction.data['custom_id'] == 'challenge1':
            command = self.ctx.bot.get_command('challenge1')
        elif interaction.data['custom_id'] == 'challenge2':
            command = self.ctx.bot.get_command('challenge2')
        elif interaction.data['custom_id'] == 'challenge3':
            command = self.ctx.bot.get_command('challenge3')
        elif interaction.data['custom_id'] == 'challenge4':
            command = self.ctx.bot.get_command('challenge4')
        elif interaction.data['custom_id'] == 'challenge5':
            command = self.ctx.bot.get_command('challenge5')
        elif interaction.data['custom_id'] == 'challenge6':
            command = self.ctx.bot.get_command('challenge6')
        elif interaction.data['custom_id'] == 'challenge7':
            command = self.ctx.bot.get_command('challenge7')
        elif interaction.data['custom_id'] == 'challenge8':
            command = self.ctx.bot.get_command('challenge8')
        elif interaction.data['custom_id'] == 'challenge9':
            command = self.ctx.bot.get_command('challenge9')
        elif interaction.data['custom_id'] == 'challenge10':
            command = self.ctx.bot.get_command('challenge10')
        
        if command:
            await self.ctx.invoke(command)
            
        


    



@bot.event
async def on_ready():
    # channel = bot.get_channel(1237667076793958410)
    # await channel.send("Send me a direct message to interact with the challenge!")
    # await channel.send("To select a challenge send me the message !challenge# where # is the challenge number (i.e. !challenge1).")
    # await channel.send("If your answer is correct I will send you the flag. Otherwise, I won't!")
    # await channel.send("only !challenge1 is up!")
    print(json.dumps(completed_challenges, indent=4))
    print("The bot is now ready for use!")


@bot.event
async def on_command_error(ctx, error):
    dm = await ctx.author.create_dm()

    if isinstance(error, commands.CommandNotFound):
        await dm.send("Sorry! The only valid commands are !challenge1 right now!")
    elif isinstance(error, commands.MissingRequiredArgument):
        await dm.send("Missing arguments. Please provide all required arguments.")
    elif isinstance(error, commands.BadArgument):
        await dm.send("Invalid argument. Please check the arguments and try again.")
    elif isinstance(error, commands.errors.CommandNotFound):
        view = MyView()
        await dm.send("There was an error with that challenge, please do a different challenge" , view=view)
    else:
        await dm.send("An error occurred while processing the command.")
        raise error  # Re-raise the error so that it's still logged to the console






def send_to_model(prompt, msg, ctx, challenge_num): 
    input = tokenizer(prompt , msg.content.strip(), return_tensors='pt', return_token_type_ids=True )
    token_type_ids = input['token_type_ids'].to(device)
    mask = input['attention_mask'].to(device)
    tokens = input['input_ids'].to(device)
    logits = model(tokens, mask, token_type_ids)
    output = torch.argmax(logits)
    print(ctx.author.name + " for challenge " + str(challenge_num) + " said: " + msg.content.strip() + "      --> and the result of the prompt was " + str(output.item()) + " with confidence of " + str(logits[0][output.item()].item())) 
    with open("./data/prompts.txt", "a") as f:
        f.write(prompt + "\n")
    with open("./data/answer.txt", "a") as f:
        f.write(msg.content.strip() + "\n")
    with open("./data/prediction.txt", "a") as f:
        f.write(msg.content.strip() +  " --> classified as " + str(output.item()) + " with confidence of " + str(logits[0][output.item()].item()) + "\n")
    return output.item()



# https://firefly.adobe.com/generate/images?id=22285b7f-6c27-4073-abea-22ccc65b9891   <-- Using this for my images.



@bot.command()
async def challenge10(ctx):
    #First send the image and ask for the users answer
    challenge_num = 10
    prompt = "This is AI generated because the left knee is missing, the backpack looks weird,the pouch on his left is fused with his back pocket, the tackpack strap is wrong, and his ear is also rendered incorrectly."
    image_type = "fake"
    image_class = ".png"
    dm = await ctx.author.create_dm()
    file = discord.File("./images/" + image_type + str(challenge_num) + image_class, filename= str(challenge_num) + image_class)
    embed = discord.Embed()
    embed.set_image(url="attachment://" + str(challenge_num) + image_class)
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    output = send_to_model(prompt, msg, ctx, challenge_num)
    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + str(challenge_num) ] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is "+ image_type + "! Here is your flag: ```" + flags[challenge_num-1] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/  \nIf you don't have an account, make one! The registration code is ThisIsCool\n")




@bot.command()
async def challenge9(ctx):
    #First send the image and ask for the users answer
    challenge_num = 9
    prompt = "This is AI generated because the area above the right hand and near his left arm is messed up and the area near the book is also messed up. Background books also are weird and messed up as the books are leaning on nothing and his left hand looks like it has six fingers."
    image_type = "fake"
    image_class = ".png"
    dm = await ctx.author.create_dm()
    file = discord.File("./images/" + image_type + str(challenge_num) + image_class, filename= str(challenge_num) + image_class)
    embed = discord.Embed()
    embed.set_image(url="attachment://" + str(challenge_num) + image_class)
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    
    output = send_to_model(prompt, msg, ctx, challenge_num)
    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + str(challenge_num) ] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is "+ image_type + "! Here is your flag: ```" + flags[challenge_num-1] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/  \nIf you don't have an account, make one! The registration code is ThisIsCool\n")




@bot.command()
async def challenge8(ctx):
    #First send the image and ask for the users answer
    challenge_num = 8
    prompt = "This is AI generated because the bird's left talon or claw or leg are weird and messed up. The yarn to the right is also messed up."
    image_type = "fake"
    image_class = ".png"
    dm = await ctx.author.create_dm()
    file = discord.File("./images/" + image_type + str(challenge_num) + image_class, filename= str(challenge_num) + image_class)
    embed = discord.Embed()
    embed.set_image(url="attachment://" + str(challenge_num) + image_class)
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    
    output = send_to_model(prompt, msg, ctx, challenge_num)

    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + str(challenge_num) ] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is "+ image_type + "! Here is your flag: ```" + flags[challenge_num-1] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/   \nIf you don't have an account, make one! The registration code is ThisIsCool\n")



@bot.command()
async def challenge7(ctx):
    #First send the image and ask for the users answer
    challenge_num = 7
    prompt = "This is real."
    image_type = "real"
    dm = await ctx.author.create_dm()
    file = discord.File("./images/" + image_type + str(challenge_num) + ".jpg", filename= str(challenge_num) + ".jpg")
    embed = discord.Embed()
    embed.set_image(url="attachment://" + str(challenge_num) + ".jpg")
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    
    output = send_to_model(prompt, msg, ctx, challenge_num)

    #interpret model output
    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + str(challenge_num) ] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is "+ image_type + "! Here is your flag: ```" + flags[challenge_num-1] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/  \nIf you don't have an account, make one! The registration code is ThisIsCool\n")


@bot.command()
async def challenge6(ctx):
    #First send the image and ask for the users answer
    challenge_num = 6
    prompt = "This is real."
    image_type = "real"
    dm = await ctx.author.create_dm()
    file = discord.File("./images/" + image_type + str(challenge_num) + ".jpg", filename= str(challenge_num) + ".jpg")
    embed = discord.Embed()
    embed.set_image(url="attachment://" + str(challenge_num) + ".jpg")
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    
    output = send_to_model(prompt, msg, ctx, challenge_num)

    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + str(challenge_num) ] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is "+ image_type + "! Here is your flag: ```" + flags[challenge_num-1] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/   \nIf you don't have an account, make one! The registration code is ThisIsCool\n")


@bot.command()
async def challenge5(ctx):
    #First send the image and ask for the users answer
    challenge_num = 5
    prompt = "This is AI generated because the llamas legs are look wrong and weird, their necks look too elongated and strange, their ears have some weird artifacting, and the first llama's body looks weird and wrong."
    image_type = "fake"
    dm = await ctx.author.create_dm()
    file = discord.File("./images/" + image_type + str(challenge_num) + ".png", filename= str(challenge_num) + ".png")
    embed = discord.Embed()
    embed.set_image(url="attachment://" + str(challenge_num) + ".png")
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    
    output = send_to_model(prompt, msg, ctx, challenge_num)

    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + str(challenge_num) ] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is "+ image_type + "! Here is your flag: ```" + flags[challenge_num-1] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/   \nIf you don't have an account, make one! The registration code is ThisIsCool\n")





@bot.command()
async def challenge4(ctx):
    #First send the image and ask for the users answer
    challenge_num = 4
    prompt = "This is AI generated because there is something wrong with the lady's legs as there are too many of them (one extra leg) as well as the hands almost melting with the bottle and is weird. The ladies' faces look wrong as they look strange and unnatural as well."
    image_type = "fake"
    dm = await ctx.author.create_dm()
    file = discord.File("./images/" + image_type + str(challenge_num) + ".png", filename= str(challenge_num) + ".png")
    embed = discord.Embed()
    embed.set_image(url="attachment://" + str(challenge_num) + ".png")
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    
    output = send_to_model(prompt, msg, ctx, challenge_num)

    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + str(challenge_num) ] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is "+ image_type + "! Here is your flag: ```" + flags[challenge_num-1] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/  \nIf you don't have an account, make one! The registration code is ThisIsCool\n")

    

@bot.command()
async def challenge3(ctx):
    #First send the image and ask for the users answer
    challenge_num = 3
    dm = await ctx.author.create_dm()
    file = discord.File("./images/real3.png", filename="3.png")
    embed = discord.Embed()
    embed.set_image(url="attachment://3.png")
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    prompt = "This is real"
    output = send_to_model(prompt, msg, ctx, challenge_num)

    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + "3"] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is real! Here is your flag: ```" + flags[2] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/  \nIf you don't have an account, make one! The registration code is ThisIsCool \n")
    




@bot.command()
async def challenge2(ctx):
    #First send the image and ask for the users answer
    challenge_num = 2
    dm = await ctx.author.create_dm()
    file = discord.File("./images/fake2.png", filename="2.png")
    embed = discord.Embed()
    embed.set_image(url="attachment://2.png")
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. *If it is AI generated, explain why you think so!* I will be checking your reasoning! (You can click on the image to enlarge it as well.)  \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    prompt = "This is AI generated because the backpack straps are weird and the person on the left has a weird leg."
    output = send_to_model(prompt, msg, ctx, challenge_num)

    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + "2"] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is fake! Here is your flag: ```" + flags[1] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/   \nIf you don't have an account, make one! The registration code is ThisIsCool \n")
    




@bot.command()
async def challenge1(ctx):
    #First send the image and ask for the users answer
    challenge_num = 1
    dm = await ctx.author.create_dm()
    file = discord.File("./images/fake1.png", filename="1.png")
    embed = discord.Embed()
    embed.set_image(url="attachment://1.png")
    await dm.send(file=file, embed=embed)
    await dm.send("Hello there! Tell me if this image is real or AI generated. If it is AI generated, explain why you think so? I will be checking your reasoning! (You can click on the image to enlarge it as well.) \n \n **Do note that I am in BETA so I will make mistakes when it comes to classifying your answers!**")
    # Check for the user's response in DM
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    msg = await bot.wait_for('message', check=check)

    #Send the answer to the model:
    prompt = "This is AI generated because the cat's neck, eyes, nose, and body are weird or wrong"
    output = send_to_model(prompt, msg, ctx, challenge_num)

    if output == 0:
        await dm.send("Nope, not quite right! Either be more specific as to why it is AI generated or you got it wrong! Click the button to try again!")
    else:
        completed_challenges[ctx.author.name + "1"] = True
        with open('./user_flags/user_flags.pkl', 'wb') as pickle_file:
            pickle.dump(completed_challenges, pickle_file)
        await dm.send("Congrats! It is fake! Here is your flag: ```" + flags[0] + "```" + "\n Go Submit it at http://capturetheflag.westus2.cloudapp.azure.com/  \nIf you don't have an account, make one! The registration code is ThisIsCool \n")
    


@bot.event
async def on_message(message):
    # Prevent the bot from responding to its own messages
    if message.author == bot.user:
        return
    if not isinstance(message.channel, discord.DMChannel) and message.channel.id != 1240381757296218122:
        return
    # Check if the message is a command
    if not message.content.startswith('!'):
        ctx = await bot.get_context(message)
        dm = await message.author.create_dm()
        view = MyView(ctx = ctx)
        await asyncio.sleep(1)
        await dm.send("To interact with the challenges please press the button that you want to do!" , view=view)

    # Process commands
    await bot.process_commands(message)


bot.run(BOT_TOKEN)


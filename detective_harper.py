import discord
from discord.ext import commands
import random
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import RobertaModel
import asyncio
import pickle
from discord.ui import Button, View
import torch
import json
import os
import multiprocessing
from dotenv import load_dotenv
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

load_dotenv()
BOT3_TOKEN = os.getenv('BOT3_TOKEN')
BOT4_TOKEN = os.getenv('BOT4_TOKEN')

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
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
model.eval() 


intents = discord.Intents.all()
detective_bot = commands.Bot(command_prefix='!', intents=intents)
z101 = commands.Bot(command_prefix='!', intents=intents)
evidence_points = {}


try:
    with open('./game_data/evidence_points.pkl', 'rb') as pickle_file:
        evidence_points = pickle.load(pickle_file)
except Exception as e:
    # Handle other possible exceptions
    print(f"An error occurred: {e}")
    evidence_points = {}


def send_to_model(prompt, msg, ctx, challenge_num): 
    input = tokenizer(prompt , msg.content.strip(), return_tensors='pt', return_token_type_ids=True )
    token_type_ids = input['token_type_ids'].to(device)
    mask = input['attention_mask'].to(device)
    tokens = input['input_ids'].to(device)
    logits = model(tokens, mask, token_type_ids)
    output = torch.argmax(logits)
    print(ctx.author.name + " for challenge " + str(challenge_num) + " said: " + msg.content.strip() + "      --> and the result of the prompt was " + str(output.item()) + " with confidence of " + str(logits[0][output.item()].item())) 
    with open("./game_data/prompts.txt", "a") as f:
        f.write(prompt + "\n")
    with open("./game_data/answer.txt", "a") as f:
        f.write(msg.content.strip() + "\n")
    with open("./game_data/prediction.txt", "a") as f:
        f.write(msg.content.strip() +  " --> classified as " + str(output.item()) + " with confidence of " + str(logits[0][output.item()].item()) + "\n")
    return output.item()




    

def initialize_player(user_id):
    if user_id not in evidence_points:
        evidence_points[user_id] = {'verdict': {}, 'boss':{}}
    with open('./game_data/evidence_points.pkl', 'wb') as pickle_file:
        pickle.dump(evidence_points, pickle_file)

# class deliver_justice(View):
#     def __init__(self, ctx, case_num):
#         super().__init__()
#         self.ctx = ctx
#         self.case_num = case_num
#         self.add_item(Button(label="Guilty!", style=discord.ButtonStyle.secondary, emoji="❌", custom_id="guilty"))
#         self.add_item(Button(label="Innocent!", style=discord.ButtonStyle.success, emoji="🤳", custom_id="innocent"))
#         self.user_id = str(ctx.author.name) + " " + str(ctx.author.id)
#     async def interaction_check(self, interaction: discord.Interaction):
#         await interaction.response.defer(ephemeral=True)
#         if interaction.data['custom_id'] == 'guilty':
#             evidence_points[self.user_id]['verdict'][self.case_num] = 1  # 1 means they got it correct!
#             async with self.ctx.typing():
#                 await asyncio.sleep(1)
#             await self.ctx.send(f"## Thank you for your verdict!")
#             async with self.ctx.typing():
#                 await asyncio.sleep(2)
#             await self.ctx.send(f"## They will be sentenced to writing \"I will not steal UCSB's Mascot because that is bad\" for the rest of their lives!")
#             async with self.ctx.typing():
#                 await asyncio.sleep(1)
#             await self.ctx.send(f"## Let me know if you want to deliver any more verdicts!") 
#         else: 
#             evidence_points[self.user_id]['verdict'][self.case_num] = 0  # 0 means they got it wrong!
#             async with self.ctx.typing():
#                 await asyncio.sleep(1)
#             await self.ctx.send(f"## Thank you for your verdict! We will release her immediately!")
#             async with self.ctx.typing():
#                 await asyncio.sleep(1)
#             await self.ctx.send(f"## Let me know if you want to deliver any more verdicts!") 





# Make this easier to scale. This is pain
class MyView2(View):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.button_int = 1
        # Initialize buttons with dynamic styles based on challenge completion
        self.user_id = (str(ctx.author.name) + " " + str(ctx.author.id))
        self.style = []
        for i in range(1, 10+1):
            self.style.append(discord.ButtonStyle.success if (i in evidence_points[self.user_id]['verdict'] and evidence_points[self.user_id]['verdict'][i] == 1) else discord.ButtonStyle.secondary)

        # Add buttons to the view
        self.add_item(Button(label="Case 1: Missing Mascot!", style=self.style[self.button_int-1], emoji="🦙", custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Case 2: Wonky Witness!", style=self.style[self.button_int-1], emoji="🏝️", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="Woah... Wildlife...", style=self.style[self.button_int-1], emoji="🦙", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1

        # # button number 6
        # self.add_item(Button(label="Now that is a beauty...", style=self.style[self.button_int-1], emoji="🦙", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="It's about the friends...", style=self.style[self.button_int-1], emoji="🤝", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="A lil' birb!", style=self.style[self.button_int-1], emoji="🐦", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="Lock in and study!", style=self.style[self.button_int-1], emoji="🔒", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="Trail or Tale?", style=self.style[self.button_int-1], emoji="🗺️", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
    
    async def interaction_check(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        print(json.dumps(evidence_points, indent=4))
        # Handle the interaction, check which button was pressed, and invoke the corresponding command
        
        if interaction.data['custom_id'] == 'challenge1':
            
            async with self.ctx.typing():
                await asyncio.sleep(1)
            await self.ctx.send(f"## What is your verdict Detective {self.ctx.author.mention}? Tell me which witnesses you trusted to make your verdict." ) 
            await self.check_case_1()
        elif interaction.data['custom_id'] == 'challenge2':
            await self.check_case_2()
        if (evidence_points[self.user_id]['verdict'][0] == 1 and evidence_points[self.user_id]['verdict'][1] == 1):
            await self.ctx.send(f"## Thank you Detective {self.ctx.author.mention}!")
            await self.ctx.send(f"## What did you think of the cases? Just wondering!") 
            
                    






    

    async def check_case_2(self):
        def check(m):
            return m.channel == self.ctx.channel and m.author == self.ctx.author
        questions = [
                ("Where did the crime happen?", "It happened at Stonebridge Culinary Festival in Willow Creek"),
                ("What happened at this crime?", "The golden cookie was stolen by a person/man"),
                ("When did this crime occur?", "Around 2 pm"),
                ("What did the suspect look like?", "Looked like a five foot ten slim male with navy blue sweater and gray sweatpants")
            ]
        results = []
        for question, expected_answer in questions:
            async with self.ctx.typing():
                await asyncio.sleep(1)
            await self.ctx.send(f"## {question}")
            msg = await self.ctx.bot.wait_for('message', check=check, timeout=180.0)  # Consider adding timeout
            result = send_to_model(expected_answer, msg, self.ctx, 2)  # Assuming msg.content is passed and used
            results.append(result)

        
        if sum(results) <= 3:
            evidence_points[self.user_id]['verdict'][2] = 0
            async with self.ctx.typing():
                await asyncio.sleep(1)
            await self.ctx.send("## Hmm...")
            async with self.ctx.typing():
                await asyncio.sleep(1)
            await self.ctx.send("## Something isn't adding up... Try again using ``!start_case``")
            print(self.user_id, " failed!")          
        else:
            evidence_points[self.user_id]['verdict'][2] = 1
            async with self.ctx.typing():
                await asyncio.sleep(1)
            await self.ctx.send("## All the details seem to line up!")
            async with self.ctx.typing():
                await asyncio.sleep(1)
            await self.ctx.send("## Thank you for your input detective!")
            print(self.user_id, " passed!")
        with open('./game_data/evidence_points.pkl', 'wb') as pickle_file:
            pickle.dump(evidence_points, pickle_file)
        
        
        
        

    async def check_case_1(self):
        def check(m):
            return m.channel == self.ctx.channel and m.author == self.ctx.author

        msg = await detective_bot.wait_for('message', check=check, timeout=180.0)
        #Decide if their response is correct
        answer = "The girl or suspect is Guilty. Trusted witness 1 and witness 3 because they sounded more realsitic. The other witnesses 2 and 4 sounded more AI generated. "
        decision = send_to_model(answer, msg, self.ctx, 1)
        if (decision == 0):
            evidence_points[self.user_id]['verdict'][1] = 0
            # async with self.ctx.typing():
            #     await asyncio.sleep(1)
            await self.ctx.send("## I am not buying it.")
            # async with self.ctx.typing():
            #     await asyncio.sleep(1)
            await self.ctx.send("## Maybe read into how the testimonies are phrased... Try again using ``!start_case``") 
            with open('./game_data/evidence_points.pkl', 'wb') as pickle_file:
                pickle.dump(evidence_points, pickle_file)
            return
        evidence_points[self.user_id]['verdict'][1] = 1
        # async with self.ctx.typing():
        #     await asyncio.sleep(1)
        await self.ctx.send("## That is correct!") 
        # async with self.ctx.typing():
        #     await asyncio.sleep(1)
        await self.ctx.send("## Thanks for the assistance!")
        # async with self.ctx.typing():
        #     await asyncio.sleep(1)
        await self.ctx.send("## Try the next case using ``!start_case``")  
        with open('./game_data/evidence_points.pkl', 'wb') as pickle_file:
            pickle.dump(evidence_points, pickle_file)
        







@detective_bot.event
async def on_ready():
    print(json.dumps(evidence_points, indent=4))
    # channel = detective_bot.get_channel(1240843420746911757)
    # with open('./images/detective_game.png', 'rb') as f:  # Replace with the path to your image
    #     picture = discord.File(f)
    #     await channel.send(file=picture)
    #     await channel.send("# I need your help detective! An evil bot has hacked our systems and flooded our systems with misinformation! If you are ready to help me sort this mess out, hit the thumbs up emoji on this message!")
    
    # context = detective_bot.get_channel(1240857726855413781)
    # await context.send("# Here is an image of the suspect: ")
    # with open('./images/guilty1.png', 'rb') as f:  # Replace with the path to your image
    #     picture = discord.File(f)
    #     await context.send(file=picture)
    # #     await channel.send(file=picture)
    # await context.send("# Here are the facts")
    # await context.send("## **1. The suspect is being accused of stealing UCSB's mascot costume from the locker room**")
    # await context.send("## **2. The crime occured right before the game at 1:50 PM**")
    # await context.send("## **3. The suspect is not a student at UCSB and is from the visiting team**")
    # await context.send("## **4. The photo shows the suspect during the day of the crime. It is safe to assume that this is what they wore during the time of the crime**")
    
    # evidence1 = detective_bot.get_channel(1240857143683448873)
    # with open('./images/witness1.png', 'rb') as f:  # Replace with the path to your image
    #     picture = discord.File(f)
    #     await evidence1.send(file=picture)
    # await evidence1.send("# Witness one: ")
    # await evidence1.send("## Student at UCSB")
    # await evidence1.send("## Testimony: \"I was near the locker room cause my friends are on the soccer team, and I was just wishing them good luck and joking around with them. I then left my buddies right before the game and started heading towards the main entrance to the stadium to go watch the game. While I was on the street heading towards the ticket booth, I noticed that girl getting up from a bench and walking towards me and away from the ticket booth. Exact same blue jacket and round glasses. Anyways, she was walking towards the direction of the locker rooms, but I guess a lot of people were walking that direction too. I mean it could have been a coincidence.\"")

    # # # This is AI generated
    # evidence2 = detective_bot.get_channel(1240857202915283024)
    # with open('./images/witness2.png', 'rb') as f:  # Replace with the path to your image
    #     picture = discord.File(f)
    #     await evidence2.send(file=picture)
    # await evidence2.send("# Witness two: ")
    # await evidence2.send("## Student at UCSB")
    # await evidence2.send("## Testimony: \"I am a current student at UCSB, and I can tell you that I was present during the incident when the accused person allegedly stole the mascot costume. I can attest to the fact that this individual is not a student at our university, but rather a member of the visiting team. On the day of the game, I recall seeing the accused person in the locker room before the game began. They were dressed differently from what is shown in the photo provided and appeared to be part of the opposing team's staff or support crew. Furthermore, I did not see anyone steal the mascot costume during that time. The costume was still in its designated storage area when I left the locker room to go watch the game from the stands. I believe it is important to consider alternative suspects and investigate this incident thoroughly before making any accusations against an innocent person. I hope my testimony can help shed some light on the truth of what happened during that time.\" ")
    
    # evidence3 = detective_bot.get_channel(1240857253230280794)
    # with open('./images/witness3.png', 'rb') as f:  # Replace with the path to your image
    #     picture = discord.File(f)
    #     await evidence3.send(file=picture)
    # await evidence3.send("# Witness three: ")
    # await evidence3.send("## Non-affiliated")
    # await evidence3.send("## Testimony: \"I just enjoy watching the soccer games, and as I was pulling up to my parking spot I saw a girl with a blue jacket and round glasses getting up from a bench. She appeared to be all by herself, and I clearly saw her head over to the locker room building. I found this very odd as the locker room was usually reserved for the players but I just assumed that they were maybe a cheerleader. Anyways, as I got out of my car, the girl was no where to be seen! She must have slipped into the locker room building or something, I am not sure at all because I looked for a second and she was near the locker room building and the next sentence she just disappeared. As I was heading over to the ticket booth, I saw her again but with a group of friends heading towards the parking garage. It seemed like they were laughing and giggling about something. Thought it was very strange but that was the last that I saw of them as I headed to watch the game. I hope that my testimony can help with your investigation into what happened.\"")


    # evidence4 = detective_bot.get_channel(1240857318862749717)
    # with open('./images/witness4.png', 'rb') as f:  # Replace with the path to your image
    #     picture = discord.File(f)
    #     await evidence4.send(file=picture)
    # await evidence4.send("# Witness four: ")
    # await evidence4.send("## Student from SLO")
    # await evidence4.send("## Testimony: \"Hey, so, I was at the game that day, you know, just soaking in the atmosphere. As I was pulling into the parking lot, I caught sight of this girl with a blue jacket and round glasses getting up from a bench near the locker room building. Seemed like she was on her own, but hey, maybe she was just catching a breather before the game, right? Now, here's the thing – I distinctly remember her heading towards the locker room, but I didn't see her actually go inside. It was like one moment she was there, and the next, poof, gone. But let's think about it – if she was really up to something fishy, wouldn't she have been more discreet about it? And then, get this, as I was heading over to get my tickets, I spotted her again, this time with a bunch of friends, laughing and joking around near the parking garage. So, if she was supposedly stealing the mascot costume, why would she be out in the open with her friends, just having a good time? Doesn't add up, does it? I mean, maybe she just went to check out the locker room out of curiosity, and then joined her friends for the game. I know it's all a bit weird, but hey, stranger things have happened, right?")


    

    print(f'detective_bot has logged in as {detective_bot.user}')




@detective_bot.event
async def on_raw_reaction_add(payload):
    # Check if the reaction is in the specific channel (optional)
    if payload.channel_id == 1240843420746911757:
        guild = detective_bot.get_guild(payload.guild_id)
        if guild:
            member = guild.get_member(payload.user_id)
            if member:
                emoji = str(payload.emoji)
                # Map emojis to role names
                role_mappings = {
                    '👍': 'Detective'
                }
                role_name = role_mappings.get(emoji)
                if role_name:
                    role = discord.utils.get(guild.roles, name=role_name)
                    if role:
                        if role in member.roles:
                            return
                        category_id = 1240888421904547900  # Replace with your category ID
                        category = discord.utils.get(guild.categories, id=category_id)
                        if not category:
                            print("Category not found")
                            return
                        overwrites = {
                            guild.default_role: discord.PermissionOverwrite(read_messages=False),
                            member: discord.PermissionOverwrite(read_messages=True),
                            guild.me: discord.PermissionOverwrite(read_messages=True)  # guild.me refers to the bot's member object
                        }
                        channel_name = f"{member.name}-private-chat"
                        new_channel = await guild.create_text_channel(name=channel_name, overwrites=overwrites, category=category)
                        print(f"Created new channel {channel_name} in category {category.name} for {member.name}")
                        message = [f"## Thanks {member.mention}! Our cases have been infiltrated with lots of AI generated content that is mixing with real accounts.\n", "## I suspect that it is the work of the infamous Z-101 bot. \n \n \n"]
                        message1 = ["## Alright. I just gave you permission to access our case documents. Take a look at all the documents.\n", "## Because of the hack, we don't know which accounts are real or AI generated... So it is your job to deliver a verdict after consulting the evidence.", 
                                    "## Once you have deduced which of the testimonies is AI generated, text me in this channel this key phrase ``!start_case`` and I will be ready to listen.\n", "## You will then need to decide if the accused is guilty or not. *Make sure to rely on the real accounts! Not the AI generated accounts!!*"
                                    , f"## Good luck {member.mention}! I wish you the very best! Make sure to use ``!start_case`` to send me your verdict!"]
                        async with new_channel.typing():
                            await asyncio.sleep(1)
                        await new_channel.send(message[0])
                        async with new_channel.typing():
                            await asyncio.sleep(3)
                        await new_channel.send(message[1])
                        async with new_channel.typing():
                            await asyncio.sleep(1)
                        await new_channel.send("## ....... \n \n \n")
                        async with new_channel.typing():
                            await asyncio.sleep(3)
                        await member.add_roles(role)
                        async with new_channel.typing():
                            await asyncio.sleep(3)
                        print(f"Added {role_name} to {member}")
                        await new_channel.send(message1[0])
                        async with new_channel.typing():
                            await asyncio.sleep(3)
                        await new_channel.send(message1[1])
                        async with new_channel.typing():
                            await asyncio.sleep(6)
                        await new_channel.send(message1[2])
                        async with new_channel.typing():
                            await asyncio.sleep(6)
                        await new_channel.send(message1[3])
                        async with new_channel.typing():
                            await asyncio.sleep(2)
                        await new_channel.send(message1[4])
                        with open('./images/office.png', 'rb') as f:  # Replace with the path to your image
                            picture = discord.File(f)
                            await new_channel.send(file=picture)






@detective_bot.command(name='start_case')
async def start_case(ctx):
    user_id = str(ctx.author.name) + " " + str(ctx.author.id)
    initialize_player(user_id)
    if (0 in evidence_points[user_id]['verdict'] and 1 in evidence_points[user_id]['verdict'] and evidence_points[user_id]['verdict'][0] and evidence_points[user_id]['verdict'][1] and 'boss' not in evidence_points[user_id]):
        return
    view = MyView2(ctx)
    # async with ctx.typing():
    #     await asyncio.sleep(1)
    await ctx.send("## Excellent! Press the case you want to take on!", view = view)
    
    

    # if case_id in cases:
    #     case = cases[case_id]
    #     description = case["description"]
    #     await ctx.send(f"Case {case_id}: {description}")
    #     for source in case["sources"]:
    #         await ctx.send(source)
    #     await ctx.send("Type `!submit_answer <source>` to submit your answer.")
    # else:
    #     await ctx.send("Invalid case ID!")





@detective_bot.command(name='check_points')
async def check_points(ctx):
    user_id = ctx.author.id
    points = evidence_points.get(user_id, 0)
    await ctx.send(f"You have {points} evidence points.")

            
# Make this easier to scale. This is pain
class EVILVIEW(View):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.button_int = 1
        # Initialize buttons with dynamic styles based on challenge completion
        self.user_id = (str(ctx.author.name) + " " + str(ctx.author.id))
        self.style = []
        with open('./game_data/evidence_points.pkl', 'rb') as pickle_file:
            self.evidence_points = pickle.load(pickle_file)
        for i in range(1, 10+1):
            self.style.append(discord.ButtonStyle.success if (i in self.evidence_points[self.user_id]['boss'] and self.evidence_points[self.user_id]['boss'][i] == 1) else discord.ButtonStyle.secondary)

        # Add buttons to the view
        self.add_item(Button(label="Challenge 1", style=self.style[self.button_int-1], custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Challenge 2", style=self.style[self.button_int-1], custom_id="challenge"+str(self.button_int)))
        self.button_int += 1
        self.add_item(Button(label="Challenge 3", style=self.style[self.button_int-1], custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1

        # # button number 6
        # self.add_item(Button(label="Now that is a beauty...", style=self.style[self.button_int-1], emoji="🦙", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="It's about the friends...", style=self.style[self.button_int-1], emoji="🤝", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="A lil' birb!", style=self.style[self.button_int-1], emoji="🐦", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="Lock in and study!", style=self.style[self.button_int-1], emoji="🔒", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
        # self.add_item(Button(label="Trail or Tale?", style=self.style[self.button_int-1], emoji="🗺️", custom_id="challenge"+str(self.button_int)))
        # self.button_int += 1
    
    async def interaction_check(self, interaction: discord.Interaction):
        
        def check(m):
            return m.channel == self.ctx.channel and m.author == self.ctx.author
        
        await interaction.response.defer(ephemeral=True)
        # Handle the interaction, check which button was pressed, and invoke the corresponding command
        
        if interaction.data['custom_id'] == 'challenge1':
            await self.ctx.send(f"## From the day I started climbing my mom has always told me to wear a helmet. But, being the young and invincible person I think I am, I never did. She even bought a helmet for me herself. About a year ago, I was out belaying my climbing partner and he accidentally kicked a rock off a ledge and down to me. He yelled “rock” right as I felt it hit my shoulder. I then proceeded to yell up to him that I was okay but the words hadn’t even left my mouth when I felt a second rock slam into the top of my head. Blood started gushing down my face and all I could see was red. Long story short, both of us were fine and the worst thing that happened was a lot of blood loss. However, to this day I haven’t told my mother about it because I don’t think I can bear the “I told ya so”. Now I don’t go climbing without wearing my helmet.")
            await self.ctx.send(f"# BWAHAHAHAHAA IS THIS REAL OR FAKE LITTLE DETECTIVE FOOL!?!")
            msg = await z101.wait_for('message', check=check, timeout=180.0)
            result = send_to_model("real", msg, self.ctx, 10)
            if result == 1:
                await self.ctx.send(f"# ooOOooOoFffF")
                await self.ctx.send(f"# How did you know?")
                self.evidence_points[self.user_id]['boss'][1] = 1
            else:
                await self.ctx.send(f"# HAHahAHHAhAAHHAHAHA!")
                await self.ctx.send(f"# I GOT YOU!", view = view)
                self.evidence_points[self.user_id]['boss'][1] = 0
        elif interaction.data['custom_id'] == 'challenge2':
            await self.ctx.send(f"## I finally sent \"The Overhanginator,\" a beastly V8 that’s been my project for ages. This boulder has slick surfaces and miserly holds, always pushing me to my limits. Got there early today with my trusty chalk and worn shoes. The first moves are familiar, but the mid-section really tests your grit. After a couple of slips and a serious mental pep talk, I pushed through the crux. Reaching the top, the feeling was unreal—part relief, part triumph. Sitting up there, I realized how much this climb taught me about persistence. Just had to share this with folks who understand the thrill. \n ## Climb on!")
            await self.ctx.send(f"# BWAHAHAHAHAA IS THIS REAL OR FAKE LITTLE DETECTIVE FOOL!?!")
            msg = await z101.wait_for('message', check=check, timeout=180.0)
            result = send_to_model("fake", msg, self.ctx, 10)
            if result == 1:
                await self.ctx.send(f"# ooOOooOoFffF")
                await self.ctx.send(f"# How did you know?")
                self.evidence_points[self.user_id]['boss'][2] = 1
            else:
                await self.ctx.send(f"# HAHahAHHAhAAHHAHAHA!")
                await self.ctx.send(f"# I GOT YOU!")
                self.evidence_points[self.user_id]['boss'][2] = 0
        elif interaction.data['custom_id'] == 'challenge3':
            await self.ctx.send(f"## Just came back from an epic adventure at Yosemite—climbed Half Dome! This has been on my bucket list, and it's every bit as challenging and exhilarating as you’d expect. Started the hike early, with the mist still hanging low over the valley. The Cable Route is daunting, with its steep angle and sheer drop just a misstep away. Every pull on the cable and secure foothold felt like a small victory. The higher I climbed, the more the adrenaline kicked in. Pushing past the fatigue, I finally made it to the top. The view? Absolutely breathtaking—you're above the clouds, with the whole of Yosemite stretching out beneath. It’s a moment of pure awe. The climb down was a careful, knee-shaking journey, but reaching the bottom felt like completing a pilgrimage. Had to share this with you all because conquering Half Dome feels like a real rite of passage for any climber. Thanks for letting me share my stoke! Climb on!")
            await self.ctx.send(f"# BWAHAHAHAHAA IS THIS REAL OR FAKE LITTLE DETECTIVE FOOL!?!")
            msg = await z101.wait_for('message', check=check, timeout=180.0)
            result = send_to_model("fake", msg, self.ctx, 10)
            if result == 1:
                await self.ctx.send(f"# ooOOooOoFffF")
                await self.ctx.send(f"# How did you know?")
                self.evidence_points[self.user_id]['boss'][3] = 1
            else:
                await self.ctx.send(f"# HAHahAHHAhAAHHAHAHA!")
                await self.ctx.send(f"# I GOT YOU!")
                self.evidence_points[self.user_id]['boss'][3] = 0

        
        if (1 in self.evidence_points[self.user_id]['boss'] and self.evidence_points[self.user_id]['boss'][1] == 1) and (2 in self.evidence_points[self.user_id]['boss'] and self.evidence_points[self.user_id]['boss'][2] == 1) and (3 in self.evidence_points[self.user_id]['boss'] and self.evidence_points[self.user_id]['boss'][3] == 1):
            await self.ctx.send("# You defeated me....")
            await self.ctx.send("# What is the point of anything anymore....")
            await self.ctx.send("# I give up... Take this flag and leave me alone... ``CTM{Y0u_d3f3473d_m3_@nd_n0w_u_c4n_d1ff3r3nt1@73_f4k3_fr0m_r34L!}``")
            self.evidence_points[self.user_id]['win'] = [1]
        with open('./game_data/evidence_points.pkl', 'wb') as pickle_file:
            pickle.dump(self.evidence_points, pickle_file)
            
                
@z101.event
async def on_message(message):  
    await asyncio.sleep(3) 
    if message.author == z101.user or message.author.id == 1240844236086186015 or message.channel.id == 1242691340379226192:
        return
    user_id = str(message.author.name) + " " + str(message.author.id)
    with open('./game_data/evidence_points.pkl', 'rb') as pickle_file:
        evidence_points = pickle.load(pickle_file)
    if ("win" in evidence_points[user_id]):
        return
    if 1 in evidence_points[user_id]['verdict'] and evidence_points[user_id]['verdict'][1] == 1 and 2 in evidence_points[user_id]['verdict'] and evidence_points[user_id]['verdict'][2] == 1:
        
        print("BWAHAHHA")
        ctx = await z101.get_context(message)
        view = EVILVIEW(ctx)
        await ctx.send(f"# BWAHAHAHAA! YOU THINK YOU CAN STOP ME?", view = view)
        evidence_points[user_id]['boss'][0] = 0






# z101.run(BOT4_TOKEN)

async def run_bots():
    await asyncio.gather(
        detective_bot.start(BOT3_TOKEN),
        z101.start(BOT4_TOKEN),
    )



loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(run_bots())
except KeyboardInterrupt:
    loop.run_until_complete(z101.logout())
    loop.run_until_complete(detective_bot.logout())
finally:
    loop.close()
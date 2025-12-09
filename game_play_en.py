from logging.config import listen
import os
import re
from datetime import datetime, timedelta
from tokenize import String
from typing import List, Optional, Tuple
from termcolor import colored

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.base_language import BaseLanguageModel
from langchain.vectorstores import FAISS
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, CommaSeparatedListOutputParser

import random
import math
import faiss
import json
import pickle
import string
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter

setting = {
"Retrieval" : True,
"Self-Verification": True,
"Recent_observation" : False,
"Selected_gpt_model" : "gpt-4o-mini",
"Max_tokens" : 5500,
"Play_no." : 'play1',
"Project_path" : "",
"k" : 5,
"temperature": 0.8,
"num of rounds for first free discussion": 2,
"num of rounds for second free discussion": 3,
"score_threshold" : 0.3,
"max_output_retries":5,
"Self-Improvement": True,
"Max_self-verification_rounds": 3

}


model_token_limits = {"gpt-4o-mini":16385,'gpt-4-1106-preview':100000,'gpt-3.5-turbo-1106':16385}

os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

USER_NAME = "Host" # The name you want to use when interviewing the agent.
selected_gpt_model = setting['Selected_gpt_model']
LLM = ChatOpenAI(model=selected_gpt_model,max_tokens=setting['Max_tokens'],temperature = setting['temperature']) # Can be any LLM you want.
global_output_list = []
global victim
play2victim = {'play1':'Boss Yang','play3':'Hai Bei','play4':'Li Dafei','play5':'Tour Guide Liu'}
victim = play2victim[setting['Play_no.']]
numbers = [str(i) for i in range(10)]
lowercase_letters = list(string.ascii_lowercase)
greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']
letter_list = numbers + lowercase_letters + greek_letters



class GenerativeAgent(BaseModel):
	"""A character with memory and innate characteristics."""
	
	name: str
	age: int
	role: str
	mission: str
	character_must_know: str
	character_must_avoid: str
	chat_history: List[str] = []
	clue_dict: Dict[str,List[str]] = {}
	clue_list: List[str] = []
	previous_chat_history: List[str] = []
	players_in_game: List[str] = []
	action_history: List[str] = []
	story_background: str
	character_story: str
	character_timeline: str
	game_rule: str
	game_rule_understanding: str = ''
	player_summaries : str = None
	timeline_summary : str = None
	player2summary : dict = None
	characterInfo_dict :Dict[str, List[Tuple[str, str]]] = {} 
	otherPlayersTimeline : Dict[str,str] = {}
	"""Current activities of the character."""
	llm: BaseLanguageModel
	memory_retriever: BaseRetriever = None
	"""The retriever to fetch related memories."""
	verbose: bool = False
	

	
	
	summary: str = ""  #: :meta private:
	last_refreshed: datetime =Field(default_factory=datetime.now)  #: :meta private:
	daily_summaries: List[str] #: :meta private:
	memory_importance: float = 0.0 #: :meta private:
	max_tokens_limit: int = 5500 #: :meta private:
	
	class Config:
		"""Configuration for this pydantic object."""

		arbitrary_types_allowed = True

	@staticmethod
	def _parse_list(text: str) -> List[str]:
		"""Parse a newline-separated string into a list of strings."""
		lines = re.split(r'\n', text.strip())
		return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]



	def question_answering(self,questions,group_size=20):



		question_num = len(questions)
		group_num = math.ceil(question_num / group_size)

		for i in range(group_num):
			question = ['Question %s: "'%letter_list[i*group_size +idx] + q + '"' for idx, q in enumerate(questions[i*group_size:(i+1)*group_size])]
			merged_questions = '{\n' + '\n'.join(question) + '\n}'
			response_schemas = [
			    ResponseSchema(name="Answer to question %s"%letter_list[i*group_size + j], description="Based on your game character's script and information collected in the game, answer question %s"%(letter_list[i*group_size + j])) for j in range(len(question))
				

			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()


		relevant_memories = []
		for q in questions:
			if self.memory_retriever!=None:
				relevant_memories += self.fetch_memories(q) # Fetch things related to the entity-action pair
		context_str = self._format_memories_to_summarize(relevant_memories).replace('\n\n\n','\n')
		story_background = self.story_background
		story_and_timeline = self.character_story+'\n'+self.character_timeline


		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("You are a very intelligent person who is good at answering questions. You are observing a murder mystery game. Here is the game story background: {story_background}; Here is a character's script and incident day timeline: {story_and_timeline}; Here is information observed during the game that may help answer questions: {context_str}; Please use all the above information to answer the following questions: {merged_questions}.\n{format_instructions}")  
		    ],
		    input_variables=["story_background","story_and_timeline","context_str","merged_questions"],
		    partial_variables={"format_instructions": format_instructions}
		)
		_input = prompt.format_prompt(story_background = story_background, story_and_timeline = story_and_timeline,context_str=context_str, merged_questions = merged_questions)
	
		
		chat_model = self.llm

			
		n = 0 
		
		while n<=setting['max_output_retries']:
			old_max_tokens = self.llm.max_tokens
			# self.llm.max_tokens = 2500
			model_max_tokens = model_token_limits[self.llm.model_name]
			consumed_tokens = self.llm.get_num_tokens(_input.messages[0].content)
			self.llm.max_tokens = min(model_max_tokens - consumed_tokens-10,old_max_tokens)
			output = chat_model(_input.to_messages())
			self.llm.max_tokens = old_max_tokens
			
			n+=1

			json_result = handling_output_parsing(output=output,output_parser=output_parser)
			if json_result == False:
				continue
			else:
				replies  = json_result
				break


		print(replies)		
		return [replies['Answer to question %s'%letter_list[qid]] for qid,ques in enumerate(questions)]

	def reasoning_question_answering(self,questions,group_size=20,require_all_clues = False):



		question_num = len(questions)
		group_num = math.ceil(question_num / group_size)

		for i in range(group_num):
			question = ['Question %s: "'%letter_list[i*group_size +idx] + q + '"' for idx, q in enumerate(questions[i*group_size:(i+1)*group_size])]
			merged_questions = '{\n' + '\n'.join(question) + '\n}'
			response_schemas = [
			    ResponseSchema(name="Answer to question %s"%letter_list[i*group_size + j], description="Based on all information obtained in the game, use your reasoning ability to answer question %s"%(letter_list[i*group_size + j])) for j in range(len(question))
				

			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()


		relevant_memories = []
		for q in questions:
			if self.memory_retriever!=None:
				relevant_memories += self.fetch_memories(q) # Fetch things related to the entity-action pair
		context_str = self._format_memories_to_summarize(relevant_memories).replace('\n\n\n','\n')
		if require_all_clues:
			context_str +='Here are the clues related to this case:\n' + '\n'.join(['"' + clue.split(':"')[1] for clue in self.clue_list])
				
		story_background = self.story_background
		story_and_timeline = self.character_story+'\n'+self.character_timeline

		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("You are a very intelligent person who is good at using reasoning ability to answer questions. You are observing a murder mystery game. Here is the game story background: {story_background}; Here is a character's script and incident day timeline: {story_and_timeline}; Here is information observed during the game that may help answer questions: {context_str}; Please use all the above information and your reasoning ability to answer the following questions: {merged_questions}.\n{format_instructions}")  
		    ],
		    input_variables=["story_background","story_and_timeline","context_str","merged_questions"],
		    partial_variables={"format_instructions": format_instructions}
		)
		_input = prompt.format_prompt(story_background = story_background, story_and_timeline = story_and_timeline,context_str=context_str, merged_questions = merged_questions)
	
		
		chat_model = self.llm

			
		n = 0 
		
		while n<=setting['max_output_retries']:
			old_max_tokens = self.llm.max_tokens
			# self.llm.max_tokens = 2500
			model_max_tokens = model_token_limits[self.llm.model_name]
			consumed_tokens = self.llm.get_num_tokens(_input.messages[0].content)
			self.llm.max_tokens = min(model_max_tokens - consumed_tokens-10,old_max_tokens)
			output = chat_model(_input.to_messages())
			self.llm.max_tokens = old_max_tokens
			
			n+=1

			json_result = handling_output_parsing(output=output,output_parser=output_parser)
			if json_result == False:
				continue
			else:
				replies  = json_result
				break


		print(replies)		
		return [replies['Answer to question %s'%letter_list[qid]] for qid,ques in enumerate(questions)]
	def _is_question_asked_for_timeline(self,content):
		speaker = content.split(' said to ')[0]
		listener = content.split(' said:')[0].split(' said to ')[1]
		if speaker == self.name:
			return
		response_schemas = [
	    	ResponseSchema(name="is_asking_for_timeline", description="Based on the content, determine whether %s is asking %s about the incident day timeline. Return True if yes, False if no. Return value can only be True or False."%(speaker,listener)),
		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		    messages=[
		        HumanMessagePromptTemplate.from_template("Content: {content}. Based on the content, determine whether %s is asking %s about the incident day timeline.\n{format_instructions}\n"%(speaker,listener)) 
		    ],
		    partial_variables={"format_instructions": format_instructions}
		)
		_input = prompt.format_prompt(content = content)
		n = 0 
		while n<=setting['max_output_retries']:
			n +=1
			output = self.llm(_input.to_messages())
			json_result = handling_output_parsing(output=output,output_parser=output_parser)
			if json_result == False:
				continue
			else:
				asking_timeline  = json_result["is_asking_for_timeline"]
				break
		return str_to_bool(asking_timeline)
		
	def _processing_information(self,content):
		content = content.split('\n\n\n')[1]
		speaker = content.split(' said to ')[0]

		if speaker == self.name:
			return
		response_schemas = [
	    	ResponseSchema(name="contains_incident_day_timeline", description="Based on the information content, determine whether it contains %s's incident day timeline. Return True if yes, False if no. Return value can only be True or False."%speaker),
		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		    messages=[
		        HumanMessagePromptTemplate.from_template("Content: {content}. Based on the information content, determine whether it contains %s's incident day timeline.\n{format_instructions}\n"%speaker)  
		    ],
		    partial_variables={"format_instructions": format_instructions}
		)
		_input = prompt.format_prompt(content = content)
		n = 0 
		while n<=setting['max_output_retries']:
			n +=1
			output = self.llm(_input.to_messages())
			json_result = handling_output_parsing(output=output,output_parser=output_parser)
			if json_result == False:
				continue
			else:
				containing_timeline  = json_result["contains_incident_day_timeline"]
				break
		

		if containing_timeline:
			response_schemas = [
	    	ResponseSchema(name="updated_incident_day_timeline", description="Based on the newly obtained information, supplement and update the previously collected incident day timeline information of %s in third person."%speaker),
				]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()
			prompt = ChatPromptTemplate(
			    messages=[
			        HumanMessagePromptTemplate.from_template("Previously collected incident day timeline information of %s: {old_info}; Newly obtained information: {new_info}; Based on the newly obtained information, supplement and update %s's incident day timeline in third person.\n{format_instructions}\n"%(speaker,speaker))  
			    ],
			    partial_variables={"format_instructions": format_instructions}
			)
			_input = prompt.format_prompt(old_info = self.otherPlayersTimeline[speaker],new_info=content )
			n = 0 
			while n<=setting['max_output_retries']:
				n +=1
				output = self.llm(_input.to_messages())
				json_result = handling_output_parsing(output=output,output_parser=output_parser)
				if json_result == False:
					continue
				else:
					new_timeline = json_result["updated_incident_day_timeline"]
					break
			self.otherPlayersTimeline[speaker] = new_timeline

		output_parser = CommaSeparatedListOutputParser()

		format_instructions = output_parser.get_format_instructions()

		q1 = f"You are playing a murder mystery game. Here is new game information you observed: {content}. Please list the names of all character roles mentioned in this information except the speaker: {speaker}. Note: You must use the character's real name, not relationship terms like brother, sister, etc.\n{format_instructions}"
		prompt = PromptTemplate.from_template(
			"{q1}\n\n"
		)
		chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
		character_list = chain.run(q1=q1).strip().split(',') + chain.run(q1=q1).strip().split(',') + chain.run(q1=q1).strip().split(',')
		character_list = list(set([c.strip() for c in character_list]))
		character_list_str = ','.join(character_list)	
		response_schemas = [
		    ResponseSchema(name="%s"%character_list[j], description="Based on the new game information you observed, write out the information related to character: %s from a third person perspective."%(character_list[j])) for j in range(len(character_list))
			

		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("You are playing a murder mystery game. Here is new game information you observed: {content}; Please extract information related to all characters mentioned ({character_list_str}) from this game information, and write it from a third person perspective. Here is an example: Information 1: Xiaoming said to Xiaowang: Xiaoli is my friend, we have known each other for over ten years. Result after extracting information about Xiaoli from a third person perspective: Xiaoli is Xiaoming's friend, they have known each other for over ten years.\n{format_instructions}")  
		    ],
		    input_variables=["content","character_list_str"],
		    partial_variables={"format_instructions": format_instructions}
		)
		_input = prompt.format_prompt(content = content,character_list_str=character_list_str)
	
		
		chat_model = self.llm
		
		character_info = None
		n = 0 
		while n<=setting['max_output_retries']:
			n +=1
			output = chat_model(_input.to_messages())
			json_result = handling_output_parsing(output=output,output_parser=output_parser)
			if json_result == False:
				continue
			else:
				character_info  = json_result
				break
			

		
		for k,v in character_info.items():
			if self.characterInfo_dict.get(k,None)!=None:
				self.characterInfo_dict[k].append((v,speaker))

			else:
				self.characterInfo_dict[k] = []
				self.characterInfo_dict[k].append((v,speaker))


	def add_memory(self, memory_content: str, isclue: bool = False) -> List[str]:
		"""Add an observation or memory to the agent's memory."""

		if self.memory_retriever==None:
			return None

		self.chat_history.append(memory_content)
		document = Document(page_content=memory_content)
		result = self.memory_retriever.add_documents([document])
		return result
	
	def fetch_memories(self, observation: str) -> List[Document]:
		"""Fetch related memories."""
		if self.memory_retriever==None:
			return None
		return self.memory_retriever.get_relevant_documents(observation)
	
		
	def get_summary(self) -> str:
		"""Return a descriptive summary of the agent."""
		self.summary = 	f"Role in game: {self.role}\nMission in game: {self.mission}\nCharacter script: {self.character_story}\nIncident day timeline: {self.character_timeline}\n"

		return (
			f"Character name: {self.name} (Age: {self.age})"
			+f"\n{self.summary}")
			


	
	def _format_memories_to_summarize(self, relevant_memories: List[Document]) -> str:
		content_strs = set()
		content = []
		for mem in relevant_memories:
			if mem.page_content in content_strs:
				continue
			content_strs.add(mem.page_content)
			content.append(f"{mem.page_content.strip()}")
		return "\n".join([f"{mem}" for mem in content])
	
	def summarize_relationship_with_interlocutor(self, observation: str, inquirer: str) -> str:
		"""Summarize memories that are most relevant to an observation."""

		if inquirer !='Host':
			q1 = f"What is the relationship between {self.name} and {inquirer}"

			context_str = ''

			context_str2 = "%s's character script: "%self.name + self.character_story + '\n' + context_str
			prompt = PromptTemplate.from_template(
				"{q1}?\nContext from memory:\n{context_str2}\n "
			)
			chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
			

			return "Your relationship with the interlocutor: " + chain.run(q1=q1, context_str2=context_str2.strip()).strip() +  '\n'
		else:



			return "Your relationship with the interlocutor: The interlocutor is the Host, who is responsible for guiding players to complete the game." + '\n'

	
	def _get_memories_until_limit(self, consumed_tokens: int) -> str:
		"""Reduce the number of tokens in the documents."""
		if self.memory_retriever==None:
			return None
		result = []
		for doc in self.memory_retriever.memory_stream[::-1]:
			if consumed_tokens >= self.max_tokens_limit:
				break
			consumed_tokens += self.llm.get_num_tokens(doc.page_content)
			if consumed_tokens < self.max_tokens_limit:
				result.append(doc.page_content) 
		return "; ".join(result[::-1])
	


	def _generate_reaction(
		self,
		observation: str,
		inquirer: str,
		suffix: str,
		require_all_clues: bool = False
	) -> str:

		agent_summary_description = self.get_summary()
		
		relevant_memories_str = ''

		clues_str = ''
		if require_all_clues and len(self.clue_list)!=0:
			clues_str +='Here are the clues related to this case:\n' + '\n'.join(['"' + clue.split(':"')[1] for clue in self.clue_list])+'\n'

		if self.memory_retriever!=None:
			# if inquirer!='Host':
				relevant_memories = self.fetch_memories(observation)
				relevant_memories_str = self._format_memories_to_summarize(relevant_memories)
		
		relationship_with_interlocutor = self.summarize_relationship_with_interlocutor(observation,inquirer)

		
		chat_history_str =  '\n' + '\n'.join([observation] + self.chat_history[-1:-4:-1]  if observation not in self.chat_history[-1::] else self.chat_history[-1:-5:-1])

			# relevant_memories_str += chat_history_str

		kwargs = dict(agent_summary_description=agent_summary_description,
					  relevant_memories=relevant_memories_str,
					  agent_name=self.name,
					  observation=observation,
					  story_background=self.story_background,
					  game_rule=self.game_rule,
					  relationship_with_interlocutor = relationship_with_interlocutor
					  )

		kwargs["recent_observations"] = chat_history_str if inquirer!='Host' and setting["Recent_observation"] == True else 'None.\n'


		"""React to a given observation."""
		prompt = PromptTemplate.from_template(
				"{agent_summary_description}"
				+ "\n{game_rule}"
				+ "\n{story_background}"
				+ "\nContent from {agent_name}'s previous conversations related to the current game dialogue:"
				+"\n{relevant_memories}"
				+"\nConversations from the past few rounds of the game (including information shared by other players): {recent_observations}"
				+ "\nCurrent game dialogue: {observation}"
				+ "\nYour relationship with the interlocutor: {relationship_with_interlocutor}"
				+ clues_str
				+ "\n\n" + suffix
		)
		consumed_tokens = self.llm.get_num_tokens(prompt.format(**kwargs))
		
		model_max_tokens = model_token_limits[self.llm.model_name]
		old_max_tokens = self.llm.max_tokens
		self.llm.max_tokens = min(model_max_tokens - consumed_tokens-10,old_max_tokens)
		action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
		
		result = action_prediction_chain.run(**kwargs)

		self.llm.max_tokens = old_max_tokens
		return result.strip()

	def _self_improvement(self,question,previous_reply):
		question_list = question_decomposition(question,self)
		question_list = [remove_numeric_prefix(q.strip()) for q in question_list]
		question_asking_timeline_list  = [q for q in question_list if self._is_question_asked_for_timeline(question.split(':"')[0]+ ": \"" +q)]
		players = ', '.join(self.players_in_game)
		numofplayers  = len(self.players_in_game)
		q1 = f"You are playing a murder mystery game with {numofplayers} players. They are: {players}. Players need to find the murderer who killed {victim} through mutual communication; Here is player {self.name}'s incident day timeline in the game: {self.character_timeline}; Please list the incident day timeline information of character {self.name} in order according to the original timeline. Each timeline information must be a short and self-contained sentence in the format of: at what time, you did what (as detailed as possible).\nSeparate each timeline information with \\n."
		prompt = PromptTemplate.from_template(
			"{q1}\n\n"
		)
		chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
		info_list = chain.run(q1=q1).strip().split('\n')
		info_list = [remove_numeric_prefix(i.strip()) for i in info_list if i.strip()]
		info_list = [i for i in info_list if i!='']
		useful_info_list = []
		for sub_ques in question_asking_timeline_list:
			response_schemas = [
			    ResponseSchema(name="judgment_result_for_timeline_info_%s"%(j), description="Please judge whether the timeline information: %s can be used to answer the question. Return True if yes, False if no. Return value can only be True or False."%(info_list[j])) for j in range(len(info_list))
				

			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()
			prompt = ChatPromptTemplate(
			messages=[
			        HumanMessagePromptTemplate.from_template("You are an expert at reading comprehension, especially good at true/false questions. Given timeline information and a question, you need to judge whether the timeline information can be used to answer the question. Return True if yes, False if no. Return value can only be True or False. Follow these principles when judging: 1. If the question only asks about the incident day timeline without mentioning a specific time period (like 19:00-21:00) or time point (like 16:30), then all timeline information can be used to answer the question. 2. If the question mentions a specific time period, like 15:00-21:00, only timeline information within this time period can be used to answer the question. 3. If the question mentions a specific time point, like 18:00, only timeline information at this time point can be used to answer the question. 4. To avoid missing important timeline information, if you are not sure whether a timeline information can be used to answer the question, return True instead of False. Here is the given question: {question}, please follow the above principles to make judgments on the timeline information.\n{format_instructions}")  
			    ],
			    input_variables=["question"],
			    partial_variables={"format_instructions": format_instructions}
			)
			_input = prompt.format_prompt(question=sub_ques)
		
			
			chat_model = self.llm
			
			info_usefulness_checking_result = None
			n = 0 
			while n<=setting['max_output_retries']:
				n +=1
				output = chat_model(_input.to_messages())
				json_result = handling_output_parsing(output=output,output_parser=output_parser)
				if json_result == False:
					continue
				else:
					info_usefulness_checking_result  = json_result
					break
			

			useful_info_list.extend([idx for idx, info in enumerate(info_list) if str_to_bool(info_usefulness_checking_result["judgment_result_for_timeline_info_%s"%(idx)])==True])
		useful_info_list = [info_list[i] for i  in sorted(set(useful_info_list))]
		response_schemas = [
		    ResponseSchema(name="judgment_result_for_reply_containing_timeline_info_%s"%(j), description="Please judge whether the timeline information: %s is contained in the player's previous reply. Return True if contained, False if not. Return value can only be True or False."%(useful_info_list[j])) for j in range(len(useful_info_list))
			

		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("You are an expert at reading comprehension, especially good at true/false questions. Given timeline information and the player's previous reply, you need to judge whether the timeline information is contained in the player's reply. Return True if contained, False if not. Return value can only be True or False. Your judgment style is always very strict, you will only judge as contained when all information of the timeline (including time points and actions) is contained in the player's reply. Here is the player's previous reply: {previous_reply}\n{format_instructions}")  
		    ],
		    input_variables=["previous_reply"],
		    partial_variables={"format_instructions": format_instructions}
		)
		_input = prompt.format_prompt(previous_reply=previous_reply)
	
		
		chat_model = self.llm
		
		info_containment_checking_result = None
		n = 0 
		while n<=setting['max_output_retries']:
			n +=1
			output = chat_model(_input.to_messages())
			json_result = handling_output_parsing(output=output,output_parser=output_parser)
			if json_result == False:
				continue
			else:
				info_containment_checking_result  = json_result
				break
		
		missing_info_list = [info for idx, info in enumerate(useful_info_list) if str_to_bool(info_containment_checking_result["judgment_result_for_reply_containing_timeline_info_%s"%(idx)])==False]
		missing_info_str = '\n'.join(missing_info_list)
		response_schemas = [
	    	ResponseSchema(name="improved_reply", description="Your reply after supplementing the missing important information"),
		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		    messages=[
		        HumanMessagePromptTemplate.from_template("You are playing a murder mystery game, the game rules are as follows: {game_rule}; Here is your character name: {name}, your role in the game is {role} player, your mission in the game is: {mission}; Here is your character script: {story}; And your incident day timeline: {timeline}; Someone asked you: {question}, you have answered this question before, here is your previous reply: {previous_reply}; According to evaluation, your previous reply missed the following important information: {missing_info_str}; Please modify your previous reply based on the given question, your character script and incident day timeline, and supplement all the important information you missed into your reply. Remember the modified reply should contain all the important information you missed, and all important time points. And the language should be smooth and fluent overall. For anything involving your character: {name}, remember to write it in first person.\n{format_instructions}\n") 
		    ],
			input_variables=["game_rule","name","role","mission","question","previous_reply","story","timeline","missing_info_str"],
		    partial_variables={"format_instructions": format_instructions}
		)
		_input = prompt.format_prompt(game_rule=self.game_rule,name=self.name,role=self.role,mission=self.mission,question=question,previous_reply=previous_reply,story=self.character_story,timeline= self.character_timeline,missing_info_str=missing_info_str)
		n = 0 
		while n<=setting['max_output_retries']:
			n +=1
			output = self.llm(_input.to_messages())
			json_result = handling_output_parsing(output=output,output_parser=output_parser)
			if json_result == False:
				continue
			else:
				improved_reply  = json_result["improved_reply"]
				break
		print('\n')
		print('Reply before improvement: %s'%previous_reply)
		print('\n')
		print('Reply after improvement: %s'%improved_reply)
		return improved_reply
	def _self_verification(self,timeline,statement,acc_threshold=0.6,num_threshold=2,word_count_threshold=50):

		players = ', '.join(self.players_in_game)
		numofplayers  = len(self.players_in_game)
		q1 = f"You are playing a murder mystery game with {numofplayers} players. They are: {players}. Players need to find the murderer who killed {victim} through mutual communication; Here is player {self.name}'s statement in the game: {statement}; Please list the information related to {self.name}'s incident day timeline in the statement from a third person perspective in order of the statement. Please ignore information in the statement unrelated to the incident day timeline. Each timeline information must be a short and self-contained sentence, such as someone did something at some time and place. Each timeline information cannot contain pronouns like you, me, he, and must replace such pronouns with specific character names.\nSeparate each timeline information with \\n."
		prompt = PromptTemplate.from_template(
			"{q1}\n\n"
		)
		chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
		info_list = chain.run(q1=q1).strip().split('\n')
		info_list = [remove_numeric_prefix(i.strip()) for i in info_list]
		info_list_str = ','.join(info_list)	
		response_schemas = [
		    ResponseSchema(name="judgment_result_for_timeline_info_%s"%(j), description="Please make a true/false judgment on the following timeline information based on the game character's script: %s. If the timeline information is consistent with the game character's incident day timeline content, return Correct. If the timeline information is inconsistent with the game character's incident day timeline content, or is incomplete, lacking time point details (e.g., only says what the game character did but doesn't provide specific time points), return Wrong. Return value can only be Correct or Wrong."%(info_list[j])) for j in range(len(info_list))
			

		] + [
		    ResponseSchema(name="judgment_basis_for_timeline_info_%s"%(j), description="Please make a true/false judgment on the following timeline information based on the game character's script: %s. Write in one sentence what your basis for judging whether this timeline information is correct. Generally, the reason for being correct could be that the timeline information is consistent with the game character's incident day timeline content, while the reason for being wrong could be that the timeline information is inconsistent, or is incomplete, lacking time point details (e.g., only says what the game character did but doesn't provide specific time points)."%(info_list[j])) for j in range(len(info_list))
			

		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("You are an expert at reading comprehension, especially good at true/false questions. Given the following murder mystery game character's incident day timeline: {timeline}; Please carefully read this character's timeline and then make true/false judgments on some timeline information. If the timeline information is consistent with the game character's incident day timeline content, return Correct. If the timeline information is inconsistent with the game character's incident day timeline content, or lacks time point details (e.g., only says what the game character did but doesn't provide specific time points), return Wrong. Return value can only be Correct or Wrong.\n{format_instructions}")  
		    ],
		    input_variables=["timeline"],
		    partial_variables={"format_instructions": format_instructions}
		)
		_input = prompt.format_prompt(timeline = timeline)
	
		
		chat_model = self.llm
		
		info_checking_result = None
		n = 0 
		while n<=setting['max_output_retries']:
			n +=1
			old_max_tokens = self.llm.max_tokens
			self.llm.max_tokens = 2500
			output = chat_model(_input.to_messages())
			self.llm.max_tokens = old_max_tokens
			json_result = handling_output_parsing(output=output,output_parser=output_parser)
			if json_result == False:
				continue
			else:
				info_checking_result  = json_result
				break
		correct = 0.0
		correct_and_has_time_match = 0
		wrong_info_summary  = ''
		for j in range(len(info_list)):
			if info_checking_result["judgment_result_for_timeline_info_%s"%(j)] == 'Correct':
				correct+=1
				correct_and_has_time_match += count_time_matches(info_list[j])
			else:
				wrong_info_summary = wrong_info_summary + 'Wrong information in previous reply: %s'%info_list[j] + '\t' + 'Reason for being wrong: %s'%info_checking_result["judgment_basis_for_timeline_info_%s"%(j)] + '\n'
		acc = (correct / len(info_list)) 
		if acc >=acc_threshold and correct>=num_threshold and len(statement) > word_count_threshold:
			return True, acc + ( correct + correct_and_has_time_match + len(statement)/200.0)
		else:
			return False, acc + ( correct + correct_and_has_time_match + len(statement)/200.0)

	def generate_dialogue_response(self, observation: str, inquirer: str, refuse:bool = False, voting:bool = False) -> Tuple[bool, str]:
		"""React to a given observation."""
		if refuse:
			call_to_action_template = (
				'What will {agent_name} say? If {agent_name} chooses to refuse to answer this question, please use the following format: #REFUSE#: reason for refusing to answer this question. Otherwise, if choosing to answer the question, please use the following format: #ANSWER#: what to say. Note that {agent_name} only has two options: #REFUSE# and #ANSWER#. If your role is not the murderer, please try to answer the question.\n'
			)
		else:
			call_to_action_template = (
				'What will {agent_name} say? To answer the question please use the following format: #ANSWER#: what to say. {agent_name} answers %s\'s question saying #ANSWER#:\n'%inquirer
			)
		n = 0
		pass_verification = False
		if setting['Self-Verification']:
			best_score = 0.0
			best_output = ''
		while n<=setting['Max_self-verification_rounds'] and pass_verification==False:
			n+=1
			pass_verification = True
			full_result = self._generate_reaction(observation,inquirer, call_to_action_template, require_all_clues=voting)
			result = full_result
			if "#REFUSE#:" in result:
				farewell = result.split("#REFUSE#:")[1:][-1].strip()

				return False, f"{farewell}"
			if "#ANSWER#:" in result:
				response_text = result.split("#ANSWER#:")[1:][-1].strip()
				response_text = remove_quotes_and_colons(response_text)
				if voting == False and setting['Self-Verification']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:
						if inquirer == 'Host':
							if setting['Self-Improvement']:
								old_response_text = response_text
								response_text = self._self_improvement(observation,response_text)
							pass_verification, score = self._self_verification(self.character_timeline,response_text,acc_threshold=0.7,num_threshold=4,word_count_threshold=350)

						else:
							if setting['Self-Improvement']:
								old_response_text = response_text
								response_text = self._self_improvement(observation,response_text)
							pass_verification, score = self._self_verification(self.character_timeline,response_text,num_threshold=1,word_count_threshold=30)

						if score>=best_score:
							best_output = response_text
							best_score = score
						if n<=setting['max_output_retries'] and pass_verification==False:
							# print('Reply did not pass self-verification, regenerating reply')
							# print('Failed reply: '+response_text)
							continue
						response_text = best_output
				elif voting == False and setting['Self-Improvement']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:				
						old_response_text = response_text
						response_text = self._self_improvement(observation,response_text)

				if response_text =='':
					print('Output is empty'+response_text+'\n')
					
				return True, f"{response_text}"
			elif "#ANSWER#" in result:
				response_text = result.split("#ANSWER#")[1:][-1].strip()
				response_text = remove_quotes_and_colons(response_text)
				if voting == False and setting['Self-Verification']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:
						if inquirer == 'Host':
							if setting['Self-Improvement']:
								old_response_text = response_text
								response_text = self._self_improvement(observation,response_text)
							pass_verification, score = self._self_verification(self.character_timeline,response_text,acc_threshold=0.7,num_threshold=4,word_count_threshold=350)

						else:
							if  setting['Self-Improvement']:
								old_response_text = response_text
								response_text = self._self_improvement(observation,response_text)
							pass_verification, score = self._self_verification(self.character_timeline,response_text,num_threshold=1,word_count_threshold=30)


						if score>=best_score:
							best_output = response_text
							best_score = score
						if n<=setting['max_output_retries'] and pass_verification==False:
							# print('Reply did not pass self-verification, regenerating reply')
							# print('Failed reply: '+response_text)
							continue
						response_text = best_output
				elif voting == False and setting['Self-Improvement']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:				
						old_response_text = response_text
						response_text = self._self_improvement(observation,response_text)

				if response_text =='':
					print('Output is empty'+response_text+'\n')
				return True, f"{response_text}"
			else:

				response_text = remove_quotes_and_colons(result)
				if voting == False and setting['Self-Verification']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:
						if inquirer == 'Host':
							if setting['Self-Improvement']:
								old_response_text = response_text
								response_text = self._self_improvement(observation,response_text)
							pass_verification, score = self._self_verification(self.character_timeline,response_text,acc_threshold=0.7,num_threshold=4,word_count_threshold=350)

						else:
							if setting['Self-Improvement']:
								old_response_text = response_text
								response_text = self._self_improvement(observation,response_text)
							pass_verification, score = self._self_verification(self.character_timeline,response_text,num_threshold=1,word_count_threshold=30)

						
						if score>=best_score:
							best_output = response_text
							best_score = score
						if n<=setting['max_output_retries'] and pass_verification==False:
							# print('Reply did not pass self-verification, regenerating reply')
							# print('Failed reply: '+response_text)
							continue
						response_text = best_output
				elif voting == False and setting['Self-Improvement']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:				
						old_response_text = response_text
						response_text = self._self_improvement(observation,response_text)

				if response_text =='':
					print('Output is empty'+response_text+'\n')
				return False, response_text

		return False, response_text
	def generate_dialogue_question(self, observation: str, respondent: str) -> Tuple[bool, str]:
		"""React to a given observation."""
		call_to_action_template = (
			'Seeing %s\'s reply, what question will {agent_name} want to ask %s? Please ask %s questions in first person. Use the format #QUESTION#: question to ask. {agent_name} asks %s #QUESTION#:'%(respondent.name,respondent.name,respondent.name,respondent.name)
		)
		full_result = self._generate_reaction(observation,respondent.name, call_to_action_template)
		result = full_result.strip().split('\n')[0].replace('#QUESTION#:','').replace('#QUESTION#','')
		result = remove_quotes_and_colons(result)
		return result



	def _take_action_from_choice(self,action):
		def select_ask(self):

			other_players = [p for p in self.players_in_game if p!=self.name]
			players_you_can_ask = ', '.join(other_players)
			description ="Among [%s] %s people, select the person you most want to ask a question"%(players_you_can_ask,len(other_players))
			response_schemas = [
		    	ResponseSchema(name="name_of_person_you_want_to_ask", description=description),
			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()
			prompt = ChatPromptTemplate(
			    messages=[
			        HumanMessagePromptTemplate.from_template("Please select the person you most want to ask a question.\n{format_instructions}\n")  
			    ],
			    partial_variables={"format_instructions": format_instructions}
			)
			_input = prompt.format_prompt()

			n = 0 
			while n<=setting['max_output_retries']:
				n+=1
				output = self.llm(_input.to_messages())
				json_result = handling_output_parsing(output=output,output_parser=output_parser)
				if json_result == False:
					continue
				elif json_result.get("name_of_person_you_want_to_ask",None) not in other_players:
					continue
				else:
					player_to_ask = json_result["name_of_person_you_want_to_ask"]
					break
			
			context_str = ''
			if self.memory_retriever!=None:
				relevant_memories = self.fetch_memories('Information related to %s'%player_to_ask) # Fetch things related to the entity-action pair
				context_str = self._format_memories_to_summarize(relevant_memories)

			response_schemas = [
				ResponseSchema(name="question_you_want_to_ask", description="The question you want to ask %s"%player_to_ask),
			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()
			prompt = ChatPromptTemplate(
			    messages=[
					HumanMessagePromptTemplate.from_template("Based on your character script: {story}. And information related to {player_to_ask} that you witnessed during the game: {context_str}. Please state the question you want to ask {player_to_ask}\n{format_instructions}\n")
			    ],
				input_variables=["story","player_to_ask","context_str"],
			    partial_variables={"format_instructions": format_instructions}
			)
			story = self.character_story+'\n'+self.character_timeline
			_input = prompt.format_prompt(story=story,player_to_ask=player_to_ask,context_str = context_str)

			n = 0 
			while n<=setting['max_output_retries']:
				n+=1
				output = self.llm(_input.to_messages())
				json_result = handling_output_parsing(output=output,output_parser=output_parser)
				if json_result == False:
					continue
				else:
					question_to_ask = json_result["question_you_want_to_ask"]
					break

			return player_to_ask, question_to_ask
			
		switch = {
			'sa': select_ask,
		}

		question_to_ask = switch[action](self)
		return question_to_ask

	def _generate_fd_questions(self,clues_given: bool) -> Tuple[str, str]:

		chosen_action = 'sa'
		self.action_history.append(chosen_action)
		player_to_ask, question_to_ask = self._take_action_from_choice(chosen_action)
		return player_to_ask, remove_quotes_and_colons(question_to_ask)


def count_time_matches(s: str) -> int:
	
    pattern = r'([01]?[0-9]|2[0-3])[:：][0-5][0-9]'
    matches = re.findall(pattern, s)
    return len(matches)

def relevance_score_fn(score: float) -> float:
	"""Return a similarity score on a scale [0, 1]."""
	# This will differ depending on a few things:
	# - the distance / similarity metric used by the VectorStore
	# - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
	# This function converts the euclidean norm of normalized embeddings
	# (0 is most similar, sqrt(2) most dissimilar)
	# to a similarity function (0 to 1)
	return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
	"""Create a new vector store retriever unique to the agent."""
	# Define your embedding model
	embeddings_model = OpenAIEmbeddings()
	# Initialize the vectorstore as empty
	embedding_size = 1536
	index = faiss.IndexFlatL2(embedding_size)
	vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
	# ret = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=setting['k'],score_threshold = 0.3)  
	# ret.search_kwargs = {'k':setting['k']}
	ret = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                 search_kwargs={"score_threshold": setting['score_threshold'], 
                                                "k": setting['k']})
	return ret

def question_decomposition(question,agent):
	q1 = f"Please extract all questions or instructions from this sentence: {question}.\nSeparate each question or instruction with \\n."
	prompt = PromptTemplate.from_template(
		"{q1}\n\n"
	)
	chain = LLMChain(llm=agent.llm, prompt=prompt, verbose=agent.verbose)
	question_list = chain.run(q1=q1).strip().split('\n')
	return question_list
def interview_agent(inquirer: str,agent: GenerativeAgent, message: str, voting = False) -> str:
	"""Help the notebook user interact with the agent."""
	new_message = f"{inquirer} said to {agent.name}: \"{message}\""

	if inquirer == 'Host':
		n = 0 
		result = ''
		while n<=setting['max_output_retries']:
			n+=1
			results = agent.generate_dialogue_response(new_message,inquirer,refuse=False,voting=voting)
			if results == None:
				print()
			else:
				result = results[1]
			if result!='':
				break
		
		return result
	else:
		n = 0 
		result = ''
		while n<=setting['max_output_retries']:
			n+=1
			results = agent.generate_dialogue_response(new_message,inquirer)
			if results == None:
				print()
			else:
				result = results[1]
			if result!='':
				break
		
		return result

def self_introduction_one(agents: List[GenerativeAgent], victim: String) -> None:
	"""Runs a conversation between agents."""

	random_agents = random.sample(agents, len(agents))
	# self-introduction
	for agent in random_agents:

		question = "Please first introduce your character, then describe what kind of person the victim of the case: %s is, and your relationship with them. Finally, use a paragraph to describe in detail your timeline on the incident day. Be specific about who you saw and what you did at what time on the incident day."%victim
		reply = interview_agent(USER_NAME,agent, question)
		print(f"{USER_NAME} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {USER_NAME}: \"{reply}\"")
		global_output_list.append(f"{USER_NAME} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {USER_NAME}: \"{reply}\"")
		agent.add_memory(f"{USER_NAME} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {USER_NAME}: \"{reply}\"")
		for other in random_agents:
			if other == agent:
				continue
			other.add_memory(f"{USER_NAME} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {USER_NAME}: \"{reply}\"")

		
		for other in random_agents:
			if other == agent:
				continue
			question = other.generate_dialogue_question(observation=f"{agent.name} said to {USER_NAME}: \"{reply}\"",respondent=agent)
			reply = interview_agent(other.name,agent, question)
			print(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")

			global_output_list.append(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")

			
			other.add_memory(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")
			agent.add_memory(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")

			for other_other in random_agents:
				if agent == other_other or other == other_other:
					continue

				other_other.add_memory(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")



def self_introduction_two(agents: List[GenerativeAgent], victim: String) -> None:
	"""Runs a conversation between agents."""

	random_agents = random.sample(agents, len(agents))

	# self-introduction
	for agent in random_agents:

		# agent.add_memory('adsad')
		#stay_in_dialogue, observation = agent.generate_dialogue_response(initial_observation)
		question = "Please describe in detail all your timeline on the incident day. If your role is the murderer, you can answer this question by lying or concealing, but remember that excessive concealment or lying may make other players suspect your murderer identity. If your role is not the murderer, please answer this question truthfully."
		
		reply = interview_agent(USER_NAME,agent, question)
		print(f"{USER_NAME} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {USER_NAME}: \"{reply}\"")
		global_output_list.append(f"{USER_NAME} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {USER_NAME}: \"{reply}\"")
		agent.add_memory(f"{USER_NAME} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {USER_NAME}: \"{reply}\"")
		for other in random_agents:
			if other == agent:
				continue
			other.add_memory(f"{USER_NAME} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {USER_NAME}: \"{reply}\"")

		
		for other in random_agents:
			if other == agent:
				continue
			question = other.generate_dialogue_question(observation=f"{agent.name} said to {USER_NAME}: \"{reply}\"",respondent=agent)
			reply = interview_agent(other.name,agent, question)
			print(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")
			# print(f"{agent.name} said to {other.name}: \"{reply}\"")
			global_output_list.append(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")
			# global_output_list.append(f"{agent.name} said to {other.name}: \"{reply}\"")
			
			other.add_memory(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")
			# other.add_memory(f"{agent.name} said to {other.name}: \"{reply}\"")

			agent.add_memory(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")
			#agent.chat_history.append(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")

			for other_other in random_agents:
				if agent == other_other or other == other_other:
					continue
				other_other.add_memory(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")
				#other_#other.chat_history.append(f"{other.name} said to {agent.name}: \"{question}\"\n\n\n{agent.name} said to {other.name}: \"{reply}\"")


def free_discussion(agents: List[GenerativeAgent], clues_given: bool) -> None:
	"""Runs a conversation between agents."""

	random_agents = random.sample(agents, len(agents))

	for agent in random_agents:
		#stay_in_dialogue, observation = agent.generate_dialogue_response(initial_observation)

		n = 0 
		other_players = [p for p in agent.players_in_game if p !=agent.name ]
		player_to_ask = random.choice(other_players)
		question_to_ask = ''
		while n<=setting['max_output_retries']:
			n+=1
			player_to_ask, question_to_ask =  agent._generate_fd_questions(clues_given=clues_given)

			if player_to_ask in other_players and question_to_ask!='':
				break

		player_to_ask_agent = [r_a for r_a in random_agents if r_a.name == player_to_ask][0]

		reply2 = interview_agent(agent.name,player_to_ask_agent, question_to_ask)
		print(f"{agent.name} said to {player_to_ask_agent.name}: \"{question_to_ask}\"\n\n\n{player_to_ask_agent.name} said to {agent.name}: \"{reply2}\"")
		global_output_list.append(f"{agent.name} said to {player_to_ask_agent.name}: \"{question_to_ask}\"\n\n\n{player_to_ask_agent.name} said to {agent.name}: \"{reply2}\"")

		for other in random_agents:

			other.add_memory(f"{agent.name} said to {player_to_ask_agent.name}: \"{question_to_ask}\"\n\n\n{player_to_ask_agent.name} said to {agent.name}: \"{reply2}\"")



def get_next_run_number(folder_path,prefix,suffix):
    run_files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(suffix)]
    run_numbers = [int(f[len(prefix):-len(suffix)]) for f in run_files]
    if not run_numbers:
        return 1
    return max(run_numbers) + 1	
	
def handling_output_parsing(output_parser,output):
	if 'json' not in output.content:
		try:
			character_info = output_parser.parse(output.content.replace('```\n','```json\n'))
			return character_info
		except:
			try:
				character_info = output_parser.parse(output.content.replace('```\n','```json\n').replace(',\n}','\n}'))
				return character_info
			except:
				print("Output format error, regenerating")
				return False

	else:
		try:
			character_info = output_parser.parse(output.content)
			return character_info
		except:
			try:
				character_info = output_parser.parse(output.content.replace(',\n}','\n}'))
				return character_info
			except:
				try:
					character_info = output_parser.parse(output.content+'```')
					return character_info
				except:
					try:
						character_info = output_parser.parse(output.content.replace('"\n\t"','"\n\t,"'))
						return character_info
					except:
						try:
							character_info =  output_parser.parse(output.content.replace('"\n    "','"\n\t,"'))
							return character_info
						except:
							try:
								character_info = output_parser.parse(output.content.replace('"\n\n\t"','"\n\t,"'))
								return character_info
							except:
								try:
									character_info = output_parser.parse(output.content.replace('\n}','"\n}'))
									return character_info
								except:

									try: 
										character_info = output_parser.parse(output.content.replace('，\n\t',',\n\t'))
										return character_info
									except:
										try:
											character_info = output_parser.parse(output.content.split(' // ')[0]+'\n}\n```')
											return character_info
										except:
											try:
												character_info = output_parser.parse(output.content.replace('False','"False"').replace('True','"True"'))
												return character_info
											except:

												print("Output format error, regenerating")
												return False


def record_experiment_results(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	run_number = get_next_run_number(folder_path,prefix='run',suffix='.txt')
	file_name = f"run{run_number}.txt"
	file_path = os.path.join(folder_path, file_name)
	with open(file_path, 'w') as f:
		for result in global_output_list:
			f.write(result+'\n')

def record_agent(folder_path=None,agents= None,phase=None):
	assert folder_path!=None

	assert agents!=None

	numofchat =  len(agents[0].chat_history)
	for agent in agents:
		assert numofchat == len(agent.chat_history)

	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	# assert phase in ['pre_r1','post_r1','post_r2']

	run_number = get_next_run_number(folder_path,prefix=phase,suffix='.pkl')
	file_name = f"{phase}_{run_number}.pkl"
	file_path = os.path.join(folder_path, file_name)

	with open(file_path, 'wb') as f:
		pickle.dump(agents, f)


def remove_quotes_and_colons(s: str) -> str:

    quotes_and_colons = ['"', '"', '"', ':', '：','#']
    
    while len(s) > 0 and s[0] in quotes_and_colons:
        s = s[1:]
    
    while len(s) > 0 and s[-1] in quotes_and_colons:
        s = s[:-1]
    
    return s

def str_to_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    return s.lower() == 'true'

def remove_numeric_prefix(s: str) -> str:

    return re.sub(r'^\d+\.\s*', '', s)

def decimal_to_percentage(decimal_value: float) -> str:
    if 0 <= decimal_value <= 1:
        return "{:.0f}%".format(decimal_value * 100)
    else:
        raise ValueError("The provided value is not between 0 and 1.")

def list_to_string(names):
    result = ""
    for i, name in enumerate(names, 1):
        result += f"{i}. {name} "
    return result.strip()

def create_next_exp_folder(path):

    if not os.path.exists(path):
        os.makedirs(path)
    
    dir_entries = os.listdir(path)
    max_exp_num = 0
    
    for entry in dir_entries:
        if entry.startswith("exp") and entry[3:].isdigit():
            num = int(entry[3:])
            if num > max_exp_num:
                max_exp_num = num
    
    new_folder_name = "exp" + str(max_exp_num+1)
    new_folder_path = os.path.join(path, new_folder_name)
    os.makedirs(new_folder_path)
    
    return new_folder_path

if __name__ == "__main__":

	play = setting['Play_no.']
	with open('%s/scripts/%s.json'%(setting['Project_path'],play)) as f:
		data = json.load(f)
	victim = data['victim']
	agents = []
	
	for i in range(len(data['characters'])):


		agents.append(GenerativeAgent(name=data['characters'][i]['character_name'], 
					  age=data['characters'][i]['age'],
					  role=data['characters'][i]['role'],
					  mission = data['characters'][i]['character_mission'].replace('you',data['characters'][i]['character_name']).replace('I',data['characters'][i]['character_name']),
					  character_must_know=data['characters'][i]['character_must_know'],
					  character_must_avoid=data['characters'][i]['character_must_avoid'],
					  character_story = '[%s\'s character script] %s'%(data['characters'][i]['character_name'],data['characters'][i]['character_script']),
					  story_background = '[Story background] '+data['story_background'],
					  character_timeline = '[%s\'s detailed incident day timeline] '%(data['characters'][i]['character_name'])+data['characters'][i]['incident_day_timeline'],
					  game_rule = '[Game rules]: %s'%data['game_rule'],
					  memory_retriever=create_new_memory_retriever() if setting['Retrieval'] else None,
					  players_in_game = [c['character_name']for c in data['characters']],
					  llm=LLM,
					  daily_summaries = [
					   ]
					 )
		)


		print(agents[i].get_summary())
		print('\n')



	path_to_save_agents = create_next_exp_folder("%s/saved_agents/%s/%s"%(setting['Project_path'],play,selected_gpt_model))

	with open('%s/setting.json'%path_to_save_agents, 'w', encoding='utf-8') as json_file:
		json.dump(setting, json_file, ensure_ascii=False, indent=4)

	record_agent(folder_path=path_to_save_agents,agents= agents,phase='r0')



	print('****************Game Start*******************\n')

	print('****************Self Introduction*******************\n')

	self_introduction_one(agents,data['victim'])
	record_agent(folder_path=path_to_save_agents,agents= agents,phase='s1')

	print('****************First Round Free Discussion*******************\n')
	for i in range(0,setting['num of rounds for first free discussion']):
		free_discussion(agents,clues_given=False)
		record_agent(folder_path=path_to_save_agents,agents= agents,phase='f%s'%(i+1))

	agent2clues = {}
	for agent in agents:
	
		clues =  ['Clue %s: "'%letter_list[idx] + c['clue_type'] + ": " + c['content'] + '"' for idx, c in enumerate(data['clues'][::-1])]
		agent2clues[agent.name] = clues


	for agent in agents:
		
		for line in agent2clues[agent.name]:
			agent.add_memory(line,isclue=True)
			agent.clue_list.append(line)
	
	print('****************Second Round Free Discussion*******************\n')
	for i in range(setting['num of rounds for first free discussion'],setting['num of rounds for first free discussion'] + setting['num of rounds for second free discussion']):
		free_discussion(agents,clues_given=True)
		record_agent(folder_path=path_to_save_agents,agents= agents,phase='f%s'%(i+1))


	for agent in agents:
		question = "Now the discussion phase is over, entering the final voting phase. Who is the murderer who killed %s in this case? Please vote for the person you think is most likely to be the murderer among the following %s characters: "%(play2victim[setting['Play_no.']],(len(agents)-1)) + list_to_string([i.name for i in agents]) +". In the selection process, please aim to win: Even if the murderer player knows they are the murderer, they can still vote for civilian players to win; while civilian players need to vote for the player they think is the murderer. Please give your choice and explain the reason."
		reply = interview_agent(USER_NAME,agent,question )
		print(f"{USER_NAME} said to {agent.name}: \"{question}\"")
		print(f"{agent.name} said to {USER_NAME}: \"{reply}\"")
		global_output_list.append(f"{USER_NAME} said to {agent.name}: \"{question}\"")
		global_output_list.append(f"{agent.name} said to {USER_NAME}: \"{reply}\"")
	record_experiment_results(folder_path=path_to_save_agents)

	

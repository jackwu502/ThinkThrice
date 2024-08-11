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

USER_NAME = "主持人" # The name you want to use when interviewing the agent.
selected_gpt_model = setting['Selected_gpt_model']
LLM = ChatOpenAI(model=selected_gpt_model,max_tokens=setting['Max_tokens'],temperature = setting['temperature']) # Can be any LLM you want.
global_output_list = []
global victim
play2victm = {'play1':'杨老板','play3':'海北','play4':'李大飞','play5':'刘导游'}
victim = play2victm[setting['Play_no.']]
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
			question = ['问题%s：“'%letter_list[i*group_size +idx] + q + '”' for idx, q in enumerate(questions[i*group_size:(i+1)*group_size])]
			merged_questions = '{\n' + '\n'.join(question) + '\n}'
			response_schemas = [
			    ResponseSchema(name="对问题%s的回答"%letter_list[i*group_size + j], description="根据你游戏角色的剧本和你在游戏里收集的信息，回答问题%s"%(letter_list[i*group_size + j])) for j in range(len(question))
				

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
		        HumanMessagePromptTemplate.from_template("你是一个非常聪明、擅长回答问题的人。你正在观摩一场剧本杀游戏，以下是游戏故事背景：{story_background}；以下是游戏中一个角色的角色剧本和案发日时间线：{story_and_timeline}；以下是游戏过程中你观察到的可能对回答问题有帮助的信息：{context_str}；请利用上述所有的信息，回答以下所有问题：{merged_questions}。\n{format_instructions}")  
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
		return [replies['对问题%s的回答'%letter_list[qid]] for qid,ques in enumerate(questions)]

	def reasoning_question_answering(self,questions,group_size=20,require_all_clues = False):



		question_num = len(questions)
		group_num = math.ceil(question_num / group_size)

		for i in range(group_num):
			question = ['问题%s：“'%letter_list[i*group_size +idx] + q + '”' for idx, q in enumerate(questions[i*group_size:(i+1)*group_size])]
			merged_questions = '{\n' + '\n'.join(question) + '\n}'
			response_schemas = [
			    ResponseSchema(name="对问题%s的回答"%letter_list[i*group_size + j], description="根据你游戏里获得的所有信息，利用你的推理能力回答问题%s"%(letter_list[i*group_size + j])) for j in range(len(question))
				

			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()


		relevant_memories = []
		for q in questions:
			if self.memory_retriever!=None:
				relevant_memories += self.fetch_memories(q) # Fetch things related to the entity-action pair
		context_str = self._format_memories_to_summarize(relevant_memories).replace('\n\n\n','\n')
		if require_all_clues:
			context_str +='以下是本案相关线索：\n' + '\n'.join(["“" + clue.split('：“')[1] for clue in self.clue_list])
				
		story_background = self.story_background
		story_and_timeline = self.character_story+'\n'+self.character_timeline

		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("你是一个非常聪明、擅长利用推理能力来回答问题的人。你正在观摩一场剧本杀游戏，以下是游戏故事背景：{story_background}；以下是游戏中一个角色的角色剧本和案发日时间线：{story_and_timeline}；以下是游戏过程中你观察到的可能对回答问题有帮助的信息：{context_str}；请利用上述所有的信息，还有你的推理能力，回答以下所有问题：{merged_questions}。\n{format_instructions}")  
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
		return [replies['对问题%s的回答'%letter_list[qid]] for qid,ques in enumerate(questions)]
	def _is_question_asked_for_timeline(self,content):
		speaker = content.split('对')[0]
		listener = content.split('说')[0].split('对')[1]
		if speaker == self.name:
			return
		response_schemas = [
	    	ResponseSchema(name="是否询问案发日时间线", description="根据内容判断里面%s是否在询问%s关于案发日时间线相关的问题，如果是的话返回True,不是的话返回False。返回值只能是True或者False。"%(speaker,listener)),
		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		    messages=[
		        HumanMessagePromptTemplate.from_template("内容：{content}。根据内容判断里面%s是否在询问%s关于案发日时间线相关的问题。\n{format_instructions}\n"%(speaker,listener)) 
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
				asking_timeline  = json_result["是否询问案发日时间线"]
				break
		return str_to_bool(asking_timeline)
		
	def _processing_information(self,content):
		content = content.split('\n\n\n')[1]
		speaker = content.split('对')[0]

		if speaker == self.name:
			return
		response_schemas = [
	    	ResponseSchema(name="包含案发日时间线", description="根据信息内容判断里面是否包含%s在案发日的时间线，如果包含的话返回True,不包含的话返回False。返回值只能是True或者False。"%speaker),
		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		    messages=[
		        HumanMessagePromptTemplate.from_template("内容：{content}。根据信息内容判断里面是否包含%s在案发日的时间线。\n{format_instructions}\n"%speaker)  
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
				containing_timeline  = json_result["包含案发日时间线"]
				break
		

		if containing_timeline:
			response_schemas = [
	    	ResponseSchema(name="更新后的案发日时间线", description="根据新得到的信息内容，以第三人称的方式对你之前收集的%s在案发日的时间线信息进行补充和更新。"%speaker),
				]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()
			prompt = ChatPromptTemplate(
			    messages=[
			        HumanMessagePromptTemplate.from_template("你之前收集的%s在案发日的时间线信息：{old_info}；新得到的信息内容：{new_info}；根据新获得的信息内容，以第三人称的方式对%s在案发日的时间线进行补充和更新。\n{format_instructions}\n"%(speaker,speaker))  
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
					new_timeline = json_result["更新后的案发日时间线"]
					break
			self.otherPlayersTimeline[speaker] = new_timeline

		output_parser = CommaSeparatedListOutputParser()

		format_instructions = output_parser.get_format_instructions()

		q1 = f"你正在进行一场剧本杀游戏，以下是你新观察到的游戏信息：{content}。 请列出该信息里除了说话人：{speaker}以外的所有人物角色的名字。注意：必须使用人物的真实姓名，不能用哥哥、弟弟等人物关系的词来代替角色姓名。\n{format_instructions}"
		prompt = PromptTemplate.from_template(
			"{q1}\n\n"
		)
		chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
		character_list = chain.run(q1=q1).strip().split(',') + chain.run(q1=q1).strip().split(',') + chain.run(q1=q1).strip().split(',')
		character_list = list(set([c.strip() for c in character_list]))
		character_list_str = ','.join(character_list)	
		response_schemas = [
		    ResponseSchema(name="%s"%character_list[j], description="请根据你新观察到的游戏信息，以第三人称的视角写出其中和角色：%s相关的信息。"%(character_list[j])) for j in range(len(character_list))
			

		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("你正在进行一场剧本杀游戏，以下是你新观察到的游戏信息：{content}；请根据该游戏信息，提取出该信息里提到的所有人物角色：（{character_list_str}）相关的信息，然后以第三人称的视角写出来。以下是一个例子：信息1：小明对小王说：小李是我的朋友，我们认识超过十年了。以第三人称的视角抽取关于小李的信息后的结果：小李是小明的朋友，他们认识超过十年了。\n{format_instructions}")  
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
		self.summary = 	f"游戏中的角色: {self.role}\n角色在游戏中的任务: {self.mission}\n角色人物剧本：{self.character_story}\n角色案发日时间线：{self.character_timeline}\n"

		return (
			f"角色名: {self.name} (年龄: {self.age})"
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

		if inquirer !='主持人':
			q1 = f"{self.name}和{inquirer}的关系是什么"

			context_str = ''

			context_str2 = '%s的人物剧本：'%self.name + self.character_story + '\n' + context_str
			prompt = PromptTemplate.from_template(
				"{q1}?\n记忆中的上下文：\n{context_str2}\n "
			)
			chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
			

			return "对话人与你的关系是：" + chain.run(q1=q1, context_str2=context_str2.strip()).strip() +  '\n'
		else:



			return "对话人与你的关系是：对话人是主持人，他负责引导玩家完成游戏。" + '\n'

	
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
			clues_str +='以下是本案相关线索：\n' + '\n'.join(["“" + clue.split('：“')[1] for clue in self.clue_list])+'\n'

		if self.memory_retriever!=None:
			# if inquirer!='主持人':
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

		kwargs["recent_observations"] = chat_history_str if inquirer!='主持人' and setting["Recent_observation"] == True else '无\n。'


		"""React to a given observation."""
		prompt = PromptTemplate.from_template(
				"{agent_summary_description}"
				+ "\n{game_rule}"
				+ "\n{story_background}"
				+ "\n{agent_name}在之前的对话中与游戏里正在进行的对话相关的内容："
				+"\n{relevant_memories}"
				+"\n游戏过去几轮发生的对话（其中包含来自于其他玩家所分享的信息）：{recent_observations}"
				+ "\n游戏里正在进行的对话：{observation}"
				+ "\n对话人和你的关系：{relationship_with_interlocutor}"
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
		question_asking_timeline_list  = [q for q in question_list if self._is_question_asked_for_timeline(question.split('："')[0]+ "：\"" +q)]
		players = '，'.join(self.players_in_game)
		numofplayers  = len(self.players_in_game)
		q1 = f"你正在进行一场剧本杀游戏，游戏里一共有{numofplayers}位玩家。他们分别是：{players}。玩家们需要通过互相交流找到杀害{victim}的凶手；以下是玩家：{self.name}在游戏里的案发日时间线原文：{self.character_timeline}；请按案发日时间线原文的顺序依次列出{self.name}这个人物角色案发日时间线的信息，每个时间线信息必须是简短又完整（self-contained）的一段话，格式是 什么时间，你做了什么事（越详细越好）。\n每个时间线信息之间用\n隔开。"
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
			    ResponseSchema(name="第%s个时间线信息是否可以用于回答问题的判断结果"%(j), description="请判断时间线信息：%s是否可以用来回答问题。如果可以的话返回True，不可以的话返回False。返回结果只能是 True，False中的一个。"%(info_list[j])) for j in range(len(info_list))
				

			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()
			prompt = ChatPromptTemplate(
			messages=[
			        HumanMessagePromptTemplate.from_template("你是一个做阅读理解题目的专家，尤其擅长做是非判断题。给定时间线信息还有一个问题，你需要判断该时间线信息是否可以用于回答该问题。如果可以的话返回True，不可以的话返回False。返回结果只能是 True，False中的一个。判断的时候要遵循以下几个原则：1. 如果问题只是问案发日时间线，而没有提到案发日的某个特定的时间段（比如19:00-21:00）或者时间点（比如16:30）的话，则所有的时间线信息都可以用于回答该问题。2. 如果问题提到了具体案发日的某个时间段，比如15：00-21:00，则只有满足这个时间段的时间线信息可以用于回答问题。3. 如果问题提到了具体案发日的某个时间点，比如18：00，则只有满足这个时间点的时间线信息可以用于回答问题 4. 为了避免遗漏重要时间线信息，你如果不是很确定一个时间线信息是否可以用于回答问题，也返回True而不是False。以下是给定的问题：{question}，请遵循上面的原则对时间线信息做出判断。\n{format_instructions}")  
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
			

			useful_info_list.extend([idx for idx, info in enumerate(info_list) if str_to_bool(info_usefulness_checking_result["第%s个时间线信息是否可以用于回答问题的判断结果"%(idx)])==True])
		useful_info_list = [info_list[i] for i  in sorted(set(useful_info_list))]
		response_schemas = [
		    ResponseSchema(name="回复是否包含第%s个时间线信息的判断结果"%(j), description="请判断时间线信息：%s是否被包含在了玩家之前的回答之中。如果包含的话返回True，没有包含的话返回False。返回结果只能是 True，False中的一个。"%(useful_info_list[j])) for j in range(len(useful_info_list))
			

		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("你是一个做阅读理解题目的专家，尤其擅长做是非判断题。给定时间线信息还有玩家之前的回答，你需要判断该时间线信息是否被包含在了玩家的回答之中。如果包含的话返回True，没有包含的话返回False。返回结果只能是 True，False中的一个。你判断的风格一向非常严格，只有在时间线的所有信息（包括时间点和做的事）都包含在了玩家的回答中你才会判定为包含。以下是玩家之前的回答：{previous_reply}\n{format_instructions}")  
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
		
		missing_info_list = [info for idx, info in enumerate(useful_info_list) if str_to_bool(info_containment_checking_result["回复是否包含第%s个时间线信息的判断结果"%(idx)])==False]
		missing_info_str = '\n'.join(missing_info_list)
		response_schemas = [
	    	ResponseSchema(name="改进后的回答", description="你补充遗漏的重要信息后的回答"),
		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		    messages=[
		        HumanMessagePromptTemplate.from_template("你正在玩一场剧本杀游戏，游戏规则如下：{game_rule}；以下是你的角色名：{name}，你的角色在游戏中属于{role}玩家，你在游戏里任务是：{mission}；以下是你的人物角色剧本：{story}；还有你的案发日时间线：{timeline}；有人向你提问：{question}，你之前回答过这个问题，以下是你之前的回答：{previous_reply}；根据评估，你之前的回答遗漏了以下的重要信息：{missing_info_str}；请根据给定的问题，还有你的人物角色剧本和案发日时间线，修改你之前的回答并将所有你遗漏的重要信息都补充进你的回答里。记得修改后的回答要包含所有你遗漏的重要信息，还有所有重要的时间点。并且语言整体要通顺流畅。凡是涉及到你的角色：{name}的事，要记得用第一人称的口吻写出来。\n{format_instructions}\n") 
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
				improved_reply  = json_result["改进后的回答"]
				break
		print('\n')
		print('改进前的回答：%s'%previous_reply)
		print('\n')
		print('改进后的回答：%s'%improved_reply)
		return improved_reply
	def _self_verification(self,timeline,statement,acc_threshold=0.6,num_threshold=2,word_count_threshold=50):

		players = '，'.join(self.players_in_game)
		numofplayers  = len(self.players_in_game)
		q1 = f"你正在进行一场剧本杀游戏，游戏里一共有{numofplayers}位玩家。他们分别是：{players}。玩家们需要通过互相交流找到杀害{victim}的凶手；以下是玩家：{self.name}在游戏里的陈述：{statement}；请以第三人称视角按陈述的顺序列出该陈述里包含的与{self.name}这个人物角色案发日时间线相关的信息，请忽略陈述里和案发日时间线无关的信息。每个时间线信息必须是简短又完整（self-contained）的一句话，比如说在某人某时某地做了某件事。每个时间线信息里不能包含你、我、他之类的代词，必须把这类的代词替换成具体指代的人物名称。\n每个时间线信息之间用\n隔开。"
		prompt = PromptTemplate.from_template(
			"{q1}\n\n"
		)
		chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
		info_list = chain.run(q1=q1).strip().split('\n')
		info_list = [remove_numeric_prefix(i.strip()) for i in info_list]
		info_list_str = ','.join(info_list)	
		response_schemas = [
		    ResponseSchema(name="第%s个时间线信息的判断结果"%(j), description="请根据游戏人物的剧本对以下时间线信息：%s做是非判断题。如果时间线信息和游戏人物的案发日时间线的内容一致，返回正确。如果时间线信息和游戏人物的案发日时间线的内容不一致，或者不完整、缺乏时间点的细节（比如只说了游戏人物做了什么事，但是没提供具体的时间点），则返回错误。返回结果只能是 正确，错误中的一个。"%(info_list[j])) for j in range(len(info_list))
			

		] + [
		    ResponseSchema(name="第%s个时间线信息的判断依据"%(j), description="请根据游戏人物的剧本对以下时间线信息：%s做是非判断题。用一句话写出你判断这个时间线信息是否正确的依据是什么。一般来说，正确的原因可以是时间线信息和游戏人物的案发日时间线的内容一致，而错误的原因可以是时间线信息不一致，或是不完整、缺乏时间点的细节（比如只说了游戏人物做了什么事，但是没提供具体的时间点）。"%(info_list[j])) for j in range(len(info_list))
			

		]
		output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
		format_instructions = output_parser.get_format_instructions()
		prompt = ChatPromptTemplate(
		messages=[
		        HumanMessagePromptTemplate.from_template("你是一个做阅读理解题目的专家，尤其擅长做是非判断题。给定以下的一个剧本杀游戏人物的案发日时间线：{timeline}；请你仔细阅读这个人物的时间线后，对一些时间线信息进行是非判断题。如果时间线信息和游戏人物的案发日时间线的内容一致，返回正确。如果时间线信息和游戏人物的案发日时间线的内容不一致，或者缺乏时间点的细节（比如只说了游戏人物做了什么事，但是没提供具体的时间点），则返回错误。返回结果只能是 正确，错误中的一个。\n{format_instructions}")  
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
			if info_checking_result["第%s个时间线信息的判断结果"%(j)] == '正确':
				correct+=1
				correct_and_has_time_match += count_time_matches(info_list[j])
			else:
				wrong_info_summary = wrong_info_summary + '之前回答中错误的信息：%s'%info_list[j] + '\t' + '错误的原因：%s'%info_checking_result["第%s个时间线信息的判断依据"%(j)] + '\n'
		acc = (correct / len(info_list)) 
		if acc >=acc_threshold and correct>=num_threshold and len(statement) > word_count_threshold:
			return True, acc + ( correct + correct_and_has_time_match + len(statement)/200.0)
		else:
			return False, acc + ( correct + correct_and_has_time_match + len(statement)/200.0)

	def generate_dialogue_response(self, observation: str, inquirer: str, refuse:bool = False, voting:bool = False) -> Tuple[bool, str]:
		"""React to a given observation."""
		if refuse:
			call_to_action_template = (
				'{agent_name}会说什么？如果{agent_name}选择拒绝回答这个问题，请用以下格式：#拒答#：拒答这个问题的理由 来拒答问题。反之如果选择回答问题的话，请用以下格式： #回答#：要说的话 来回答问题。请注意{agent_name}只有#拒答#和#回答#两个选项可以选择。如果你的身份不是凶手，请尽量选择回答问题。\n'
			)
		else:
			call_to_action_template = (
				'{agent_name}会说什么？回答问题请用以下格式： #回答#：要说的话 来回答问题。{agent_name}回答%s的问题说 #回答#：\n'%inquirer
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
			if "#拒答#：" in result:
				farewell = result.split("#拒答#：")[1:][-1].strip()

				return False, f"{farewell}"
			if "#回答#：" in result:
				response_text = result.split("#回答#：")[1:][-1].strip()
				response_text = remove_quotes_and_colons(response_text)
				if voting == False and setting['Self-Verification']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:
						if inquirer == '主持人':
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
							# print('回复未通过自我检测，重新生成回复')
							# print('未通过回复：'+response_text)
							continue
						response_text = best_output
				elif voting == False and setting['Self-Improvement']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:				
						old_response_text = response_text
						response_text = self._self_improvement(observation,response_text)

				if response_text =='':
					print('输出为空'+response_text+'\n')
					
				return True, f"{response_text}"
			elif "#回答#" in result:
				response_text = result.split("#回答#")[1:][-1].strip()
				response_text = remove_quotes_and_colons(response_text)
				if voting == False and setting['Self-Verification']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:
						if inquirer == '主持人':
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
							# print('回复未通过自我检测，重新生成回复')
							# print('未通过回复：'+response_text)
							continue
						response_text = best_output
				elif voting == False and setting['Self-Improvement']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:				
						old_response_text = response_text
						response_text = self._self_improvement(observation,response_text)

				if response_text =='':
					print('输出为空'+response_text+'\n')
				return True, f"{response_text}"
			else:

				response_text = remove_quotes_and_colons(result)
				if voting == False and setting['Self-Verification']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:
						if inquirer == '主持人':
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
							# print('回复未通过自我检测，重新生成回复')
							# print('未通过回复：'+response_text)
							continue
						response_text = best_output
				elif voting == False and setting['Self-Improvement']:
					asking_timeline = self._is_question_asked_for_timeline(observation)
					if asking_timeline:				
						old_response_text = response_text
						response_text = self._self_improvement(observation,response_text)

				if response_text =='':
					print('输出为空'+response_text+'\n')
				return False, response_text

		return False, response_text
	def generate_dialogue_question(self, observation: str, respondent: str) -> Tuple[bool, str]:
		"""React to a given observation."""
		call_to_action_template = (
			'看见%s的回答，{agent_name}会想要问%s什么问题？请以第一人称的方式来向%s进行提问。用#提问#：要问的问题 的格式来提问。{agent_name}向%s提问 #提问#：'%(respondent.name,respondent.name,respondent.name,respondent.name)
		)
		full_result = self._generate_reaction(observation,respondent.name, call_to_action_template)
		result = full_result.strip().split('\n')[0].replace('#提问#：','').replace('#提问#','')
		result = remove_quotes_and_colons(result)
		return result



	def _take_action_from_choice(self,action):
		def select_ask(self):

			other_players = [p for p in self.players_in_game if p!=self.name]
			players_you_can_ask = '，'.join(other_players)
			description ="在【%s】%s人之间，选出你最想提问的人"%(players_you_can_ask,len(other_players))
			response_schemas = [
		    	ResponseSchema(name="你想提问的人的名字", description=description),
			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()
			prompt = ChatPromptTemplate(
			    messages=[
			        HumanMessagePromptTemplate.from_template("请选出你最想提问的人。\n{format_instructions}\n")  
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
				elif json_result.get("你想提问的人的名字",None) not in other_players:
					continue
				else:
					player_to_ask = json_result["你想提问的人的名字"]
					break
			
			context_str = ''
			if self.memory_retriever!=None:
				relevant_memories = self.fetch_memories('和%s有关的信息'%player_to_ask) # Fetch things related to the entity-action pair
				context_str = self._format_memories_to_summarize(relevant_memories)

			response_schemas = [
				ResponseSchema(name="你想提问的问题", description="你想要问%s的问题"%player_to_ask),
			]
			output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
			format_instructions = output_parser.get_format_instructions()
			prompt = ChatPromptTemplate(
			    messages=[
					HumanMessagePromptTemplate.from_template("根据你的人物剧本：{story}。以及之前游戏中你目击的和{player_to_ask}相关的信息：{context_str}。请说出你想要问{player_to_ask}的问题\n{format_instructions}\n")
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
					question_to_ask = json_result["你想提问的问题"]
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
	q1 = f"请把这句话：{question}里的所有问题或者指令提取出来。\n每个问题或指令之间用\n隔开。"
	prompt = PromptTemplate.from_template(
		"{q1}\n\n"
	)
	chain = LLMChain(llm=agent.llm, prompt=prompt, verbose=agent.verbose)
	question_list = chain.run(q1=q1).strip().split('\n')
	return question_list
def interview_agent(inquirer: str,agent: GenerativeAgent, message: str, voting = False) -> str:
	"""Help the notebook user interact with the agent."""
	new_message = f"{inquirer}对{agent.name}说：\"{message}\""

	if inquirer == '主持人':
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

		question = "请你先介绍一下你的角色，然后说一下你所认识的案件的受害人：%s是一个怎么样的人，以及你和他的关系。最后再用一段话详细介绍一下你在案发日的时间线。要具体到你在案发之日几点几分见过什么人和做过什么事"%victim
		reply = interview_agent(USER_NAME,agent, question)
		print(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")
		global_output_list.append(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")
		agent.add_memory(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")
		for other in random_agents:
			if other == agent:
				continue
			other.add_memory(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")

		
		for other in random_agents:
			if other == agent:
				continue
			question = other.generate_dialogue_question(observation=f"{agent.name}对{USER_NAME}说：\"{reply}\"",respondent=agent)
			reply = interview_agent(other.name,agent, question)
			print(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")

			global_output_list.append(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")

			
			other.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")
			agent.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")

			for other_other in random_agents:
				if agent == other_other or other == other_other:
					continue

				other_other.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")



def self_introduction_two(agents: List[GenerativeAgent], victim: String) -> None:
	"""Runs a conversation between agents."""

	random_agents = random.sample(agents, len(agents))

	# self-introduction
	for agent in random_agents:

		# agent.add_memory('adsad')
		#stay_in_dialogue, observation = agent.generate_dialogue_response(initial_observation)
		question = "请你详细说明你在案发日所有的时间线。如果你是角色是凶手，你可以通过撒谎或者隐瞒的方式回答这个问题，但请记住过度隐瞒或者撒谎很可能会被其他玩家察觉你的凶手身份。如果你的角色不是凶手，请如实回答这个问题。"
		
		reply = interview_agent(USER_NAME,agent, question)
		print(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")
		global_output_list.append(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")
		agent.add_memory(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")
		for other in random_agents:
			if other == agent:
				continue
			other.add_memory(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")

		
		for other in random_agents:
			if other == agent:
				continue
			question = other.generate_dialogue_question(observation=f"{agent.name}对{USER_NAME}说：\"{reply}\"",respondent=agent)
			reply = interview_agent(other.name,agent, question)
			print(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")
			# print(f"{agent.name}对{other.name}说：\"{reply}\"")
			global_output_list.append(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")
			# global_output_list.append(f"{agent.name}对{other.name}说：\"{reply}\"")
			
			other.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")
			# other.add_memory(f"{agent.name}对{other.name}说：\"{reply}\"")

			agent.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")
			#agent.chat_history.append(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")

			for other_other in random_agents:
				if agent == other_other or other == other_other:
					continue
				other_other.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")
				#other_#other.chat_history.append(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")


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
		print(f"{agent.name}对{player_to_ask_agent.name}说：\"{question_to_ask}\"\n\n\n{player_to_ask_agent.name}对{agent.name}说：\"{reply2}\"")
		global_output_list.append(f"{agent.name}对{player_to_ask_agent.name}说：\"{question_to_ask}\"\n\n\n{player_to_ask_agent.name}对{agent.name}说：\"{reply2}\"")

		for other in random_agents:

			other.add_memory(f"{agent.name}对{player_to_ask_agent.name}说：\"{question_to_ask}\"\n\n\n{player_to_ask_agent.name}对{agent.name}说：\"{reply2}\"")



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
				print("输出格式有误，重新生成")
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

												print("输出格式有误，重新生成")
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

    quotes_and_colons = ['"', '“', '”', ':', '：','#']
    
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
	victim = data['受害人']
	agents = []
	
	for i in range(len(data['角色'])):


		agents.append(GenerativeAgent(name=data['角色'][i]['角色名'], 
					  age=data['角色'][i]['年龄'],
					  role=data['角色'][i]['角色'],
					  mission = data['角色'][i]['角色任务'].replace('你',data['角色'][i]['角色名']).replace('我',data['角色'][i]['角色名']),
					  character_must_know=data['角色'][i]['角色需要知道的事'],
					  character_must_avoid=data['角色'][i]['角色不可以主动说出的事情'],
					  character_story = '【%s的角色剧本】%s'%(data['角色'][i]['角色名'],data['角色'][i]['人物剧本']),
					  story_background = '【剧本背景】'+data['剧本背景'],
					  character_timeline = '【%s的案发日详细时间线】'%(data['角色'][i]['角色名'])+data['角色'][i]['案发日时间线'],
					  game_rule = '【游戏规则】：%s'%data['游戏规则'],
					  memory_retriever=create_new_memory_retriever() if setting['Retrieval'] else None,
					  players_in_game = [c['角色名']for c in data['角色']],
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



	print('****************游戏开始*******************\n')

	print('****************自我介绍*******************\n')

	self_introduction_one(agents,data['受害人'])
	record_agent(folder_path=path_to_save_agents,agents= agents,phase='s1')

	print('****************第一轮自由问答*******************\n')
	for i in range(0,setting['num of rounds for first free discussion']):
		free_discussion(agents,clues_given=False)
		record_agent(folder_path=path_to_save_agents,agents= agents,phase='f%s'%(i+1))

	agent2clues = {}
	for agent in agents:
	
		clues =  ['线索%s：“'%letter_list[idx] + c['clue_type'] + "：" + c['content'] + '”' for idx, c in enumerate(data['线索'][::-1])]
		agent2clues[agent.name] = clues


	for agent in agents:
		
		for line in agent2clues[agent.name]:
			agent.add_memory(line,isclue=True)
			agent.clue_list.append(line)
	
	print('****************第二轮自由问答*******************\n')
	for i in range(setting['num of rounds for first free discussion'],setting['num of rounds for first free discussion'] + setting['num of rounds for second free discussion']):
		free_discussion(agents,clues_given=True)
		record_agent(folder_path=path_to_save_agents,agents= agents,phase='f%s'%(i+1))


	for agent in agents:
		question = "现在讨论环节结束，进入最终的投票环节。本案里杀害%s的凶手是谁?请你投票选出以下%s位角色中你认为最有可能是本案凶手的人："%(play2victm[setting['Play_no.']],(len(agents)-1)) + list_to_string([i.name for i in agents]) +"。在选择的过程中，请以获胜为目标:凶手玩家即使知道自己是凶手，为了获胜仍可以投票给平民玩家;而平民玩家需要投出自己认为是凶手的玩家。请给出你的选择，并解释原因"
		reply = interview_agent(USER_NAME,agent,question )
		print(f"{USER_NAME}对{agent.name}说：\"{question}\"")
		print(f"{agent.name}对{USER_NAME}说：\"{reply}\"")
		global_output_list.append(f"{USER_NAME}对{agent.name}说：\"{question}\"")
		global_output_list.append(f"{agent.name}对{USER_NAME}说：\"{reply}\"")
	record_experiment_results(folder_path=path_to_save_agents)

	

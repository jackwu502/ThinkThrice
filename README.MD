## ThinkThrice

This repository contains the code and resources for our ACL 2024 Findings paper titled "**Deciphering Digital Detectives: Understanding LLM Behaviors and Capabilities in Multi-Agent Mystery Games**." You can access the paper via [this link](#).

## Setup Instructions

To run the code in this repository, please follow these steps:

1. **Request Data Access**: Before running the code, please fill out [this Google Form](https://forms.gle/a2gTSd9fKBC6k4vL7) to request access to the necessary data.

2. **Run the Code**: Once you have access to the data, you can execute the main script by running:
   ```bash
   python game_play.py
   ```

## Status

This repository currently includes:
- [x] **LLM-based Agents Self-Playing Jubensha Games**
- [ ] Evaluation on factual question answering
- [ ] Evaluation on inferential question answering
- [ ] Evaluation on murderer identification

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{wu-etal-2024-deciphering,
    title = "Deciphering Digital Detectives: Understanding {LLM} Behaviors and Capabilities in Multi-Agent Mystery Games",
    author = "Wu, Dekun  and
      Shi, Haochen  and
      Sun, Zhiyuan  and
      Liu, Bang",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.490",
    pages = "8225--8291",
    abstract = "In this study, we explore the application of Large Language Models (LLMs) in Jubensha, a Chinese detective role-playing game and a novel area in Artificial Intelligence (AI) driven gaming. We introduce the first dataset specifically for Jubensha, including character scripts and game rules, to foster AI agent development in this complex narrative environment. Our work also presents a unique multi-agent interaction framework using LLMs, allowing AI agents to autonomously engage in Jubensha games. To evaluate the gaming performance of these AI agents, we developed novel methods measuring their mastery of case information and reasoning skills. Furthermore, we incorporated the latest advancements in prompting engineering to enhance the agents{'} performance in information gathering, murderer identification, and logical reasoning. The experimental results validate the effectiveness of our proposed methods. This work aims to offer a novel perspective on understanding LLM capabilities and establish a new benchmark for evaluating large language model-based agents.",
}
```
```

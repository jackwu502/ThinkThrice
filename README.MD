## ThinkThrice

This repository contains the code and resources for our ACL 2024 Findings paper titled "**Deciphering Digital Detectives: Understanding LLM Behaviors and Capabilities in Multi-Agent Mystery Games**." You can access the paper via [this link](#).

## Setup Instructions

To run the code in this repository, please follow these steps:

1. **Request Data Access**: Before running the code, please fill out [this Google Form](https://forms.gle/a2gTSd9fKBC6k4vL7) to request access to the necessary data.

2. **Create a Directory for Scripts**: 
   After you have access to the data, create a `scripts` directory by running:
   ```bash
   mkdir scripts

3. **Run the Code**: Once you have access to the data, you can execute the main script by running:
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
    pages = "8225--8291"
}
```
```

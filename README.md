# EPD: Long-term Memory Extraction, Context-awared Planning and Multi-iteration Decision @ EgoPlan Challenge ICML 2024
## Requirements
openai = 0.28.0

python = 3.8
## Usage
### Stage 1:
Use GPT-4o to extract memory information.
```
python gpt4o_extraction.py
```
### Stage 2:
Use GPT-4o or Claude3.5 to plan the next step, and convert 1584 json format answers to ans.json.
```
python gpt4o_planning.py 
python json_to_full_answer.py
```
or
```
python claude_planning.py 
python json_to_full_answer.py
```

### Stage 3:
Use GPT-4o to compare answers from GPT-4o and Claude, decide the most reasonable answer.
```
python GPT_decision.py
```


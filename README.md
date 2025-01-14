# LLM2024コンペ

予選4位 : 3.83

決勝2位 : 3.73835

# 1. model

**google/gemma-2-27b**

## 1.1 選定理由

- パラメータ数が多い
- llm-jpの出力は良い評価を受けていない
- 4bit量子化で使用可能

# 2. method

## 2.1 unslothでのモデルの読み込み

### 2.1.1 unslothの注意点

以下のような、デフォルトコードだとunslothのgemma2-27b-bnb-4bitが呼ばれる。

※これはコンペのルールに抵触する恐れあり。

```python
model_name = "google/gemma-2-27b"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
```

一度、4bit量子化したモデルを保存する。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

# モデル名と量子化設定
model_name = "google/gemma-2-27b"  # 元のモデル名を指定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# トークナイザーとモデルのロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Hugging Face Hubにアップロード
repo_name = "tomo1222/gemma-2-27b-bf16-4bit"  # アップロード先リポジトリ名
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
```

これを用いて以下のように行う。

```python
model_name = "tomo1222/gemma-2-27b-bf16-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
```

ただし、tokenizerのchat_templeteがないためgoogle/gemma-2-9bのものを流用する

```python
tokenizer.chat_template = """
{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}
"""
```

## 2.2 Supervised FineTuning

### 2.2.1 学習データ

**方針**

- ライセンスの継承が必要なデータセットを用いない。
- 個人が作成したと思われるデータセットを用いない。
- 質の悪いデータを寄せ集めずに、良質なデータのみを用いる。
- 合成データをvLLMで作成する場合はapache2.0のモデルのみを用いる (結局、Qwen2.5の出力が微妙だったため用いず。tanukiを試す前に時間切れ)

1. llm-jp/magpie-sft-v1.0

[llm-jp/magpie-sft-v1.0 · Datasets at Hugging Face](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)

- 合成データであるため、filterで質の悪いテキストを除去する。
- 特にQwen2.5は出力が途中から中国語になるケースが非常に多い。

```python
from datasets import concatenate_datasets, load_dataset, Dataset
from transformers import TrainingArguments, Trainer
import random
import pycld2 as cld2
import json
from collections import Counter

input_name = "llm-jp/magpie-sft-v1.0"
output_file = "filtered_magpie-sft-v1.jsonl"

# instruct
instruct_ng_list = []

# 2. データセットのロードと前処理
def process_magpie(example):
    user_message = next(
        (conv["content"] for conv in example["conversations"] if conv["role"] == "user"), ""
    )
    assistant_response = next(
        (conv["content"] for conv in example["conversations"] if conv["role"] == "assistant"), ""
    )
    return {
        "instruction": user_message, 
        "response": assistant_response
    }

magpie = load_dataset(input_name, split="train")

magpie_processed = magpie.map(process_magpie)

# データセットの結合
combined_dataset = concatenate_datasets([magpie_processed])

print(len(combined_dataset))

# ホワイトリスト（許可する言語コード）
whitelist = {"ja"} 
threshold = 60

# 置き換え文字リスト
replace_dict={"よしmitsu":"よしみつ", "(しみずじ)":"(きよみずでら)", "けんせんせいしんしっかん":"せんてんせいしんしっかん",
              "「月日は百 السنين كالساعة」":"「月日は百代の過客にして」","「月日は百 السنين كالساعة」":"「月日は百代の過客にして」",
              "(Chikoma no tomo)":"(ちくばのとも)","(Taketori Monogatari)":"(たけとりものがたり)","ね sake, but in this context it means Yanaka":"ねぎし",
              "（ Rokuon-ji ）":"(きんかくじ)","（ Jisho-ji ）":"(ぎんかくじ)","（ Tenryu-ji ）":"(てんりゅうじ)","（ Arashiyama ）":"あらしやま",
              "（ Takeshizaka ）":"(ちくりんのみち)"," Weekly Shonen Jump ":"週刊少年ジャンプ","松下":"三菱","(Regular Expressions)":"","アイシテル":"愛してる",
              "スライcing":"スライシング","(Watashi mo ikimasu.)":"","(Uchimaimasu.)":"(うかがいます)","変換器モデル":"トランスフォーマー",
              "（うろたき さこんじ）":"（うろこだき さこんじ）","ふせかわ":"しなずがわ","サマータ임レンダ":"サマータタイムレンダー","语":"語","（ぎち）":"（ぎふ）",
              "（ささざい）":"(ささちまき)","♪":"","😢":""
              }

# 削除テキストリスト(存在すればデータを採用しない)
ng_list=["翻訳","英訳","（ストroma）",".com",
        "语","宾","实","际","阅","读","过","补","丝","龙","姊","而","於","處","據","關","则","应","远","离","电","兴","动","发","馆","网","资","们","简","蕊","从","结","级","关",
        "调","协","钟","锅","对","샛","乡","孢","专","广","ゝ","输","宠","叛","约","牝","转","鄉","ώ","ἐ","ч","岛","잃","烧","래","瓊","车","艺","圳","姶","頚","冰","茉","涜","旱",
         "线","杆","塌","护","头","这","业","饱","饱","ㄏ","ㄠ","饱","你","灵","㐂","剑","另","为","儿","视","频","创","现","东","园","卫","变","压","传","统","污","长","鸟","洁",
         "iche ", "Ich ","靠","嗽","籾","进","о","积","适","时",
         "그","레","이","트", "게","이","츠","비","임","무","협","안","녕","히", "주","세","요","텃","밭","扫","场","节","营","设","绿","诗","说","п","乐","쇠","퇴","仔","鳃","鹅",
         "Gatsby","쇻","丽","裔","壳","墙","标","с","丰","层","泽","气","块","Я","沟","罗","义","讲","采","验","训","误","错",
         "ι","ε","ὐ", "ό","í","è","ù","ì","л","н","й","ь","ō","à","á","ñ","э","ö","ä","χ","ά","ς","ó","ū","ệ","ú","Ω","т",
         "لسنين كا","ل","س","ا", "ع","ة」","حمد",
         "ข","อ","บ","คุ","ณ","ข","อ","บ","คุ","ณ","ม","า","ก","ค","รั","บ","ค่","ะ","ł","ż","É","ô","đ","ц","ā","б","£","ن",
         "ந","ன்","றி","ধন্","য","বাদ","（alhamdulillah）","д","е","а",
         "ð","ə","и","�","银","尔","⚗","习","ू","æ","얽","★","ˈ","ü","间","탑","웩","é",
         " - Konnichiwa "," - Arigatou "," - Sumimasen "," - Nandesu ka "," - Nandesu ka ",
         "ドイツ語","ラテン語","中国語","ロシア語","ヒンディー語","アラビア語","韓国語","フランス語","ポルトガル語","スペイン語","イタリア語","タミル語","インドネシア語","タイ語",
         "ベトナム語","ジャワ語","ベンガル語","トルコ語","ウルドゥ語","朝鮮語","フィンランド語","スウェーデン語","ハンガリー語","ヘブライ語：","英語:",
         "```","https://","http://","<h1>","<h2>","<h3>","<h4>","<a>", "AI assistant", "英語訳", "私は「Claude」"]

# 出現頻度19以下はすべて除外する
tot_text = ""
for text in combined_dataset:
    tot_text += text["response"]+" "+text["instruction"]
    # 文字単位で頻度をカウント
char_counts = Counter(tot_text)
for char, count in char_counts.most_common():
    if count<=19:
        ng_list.append(str(char))
print(f"ng:{len(ng_list)}")

# フィルタ処理
filtered_dataset = []
for text in combined_dataset:

    for key,value in replace_dict.items():
        if key in text["response"]:
            text["response"] = text["response"].replace(key, value)
        if key in text["instruction"]:
            text["instruction"] = text["instruction"].replace(key, value)
    
    remove = False
    for ng_word in ng_list:
        if ng_word in text["response"] or ng_word in text["instruction"]:
            remove = True
            continue
    for ng_word in instruct_ng_list:
        if ng_word in text["instruction"]:
            remove = True
            continue
    if remove:
        continue

    flag1 = False
    flag2 = False
    
    try:
        is_reliable, text_bytes_found, details = cld2.detect(text["instruction"])
        for lang in details:
            lang_name, lang_code, lang_percent,_ = lang

            
            if lang_code in whitelist and lang_percent >= threshold:
                flag1 = True
                break
        is_reliable, text_bytes_found, details = cld2.detect(text["response"])
        for lang in details:
            lang_name, lang_code, lang_percent,_ = lang

            if lang_code in whitelist and lang_percent >= threshold:
                flag2 = True
                break
        if flag1 and flag2:
            filtered_dataset.append(text)
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")

tot_text = ""
for text in filtered_dataset:
    tot_text += text["response"]+" "+text["instruction"]
    # 文字単位で頻度をカウント
char_counts = Counter(tot_text)

# 頻度順にソートして10以下のものを更に消す
ng_list = []
for char, count in char_counts.most_common():
    if count<10:
        ng_list.append(str(char))

print(f"ng:{len(ng_list)}")

filtered_dataset = [
    text for text in filtered_dataset
    if not any(ng_word in text["response"] or ng_word in text["instruction"] for ng_word in ng_list)
]

combined_dataset = Dataset.from_list(filtered_dataset)

tot_text = ""
for text in filtered_dataset:
    tot_text += text["response"]+" "+text["instruction"]
    # 文字単位で頻度をカウント
char_counts = Counter(tot_text)

print(len(filtered_dataset))

for char, count in char_counts.most_common():
    print(f"{char}: {count}")

# データセットを JSONL 形式で保存
with open(output_file, "w", encoding="utf-8") as f:
    for data in filtered_dataset:  # filtered_dataset はリスト形式
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

```

1. tomo1222/Japanese-QA111dataset

自身で作成した111個のQAデータセット

### 2.2.2 Trainer

- lrは2e-4で固定する。
- 学習に大きな影響はないと考えられるtarget_modulesやlora_dropout, optimizer, schedulerは固定とする。
- rとlora_alphaは試行錯誤の結果r=64, lora_alpha=64を採用。
- 1epochでは学習が不十分だと感じたため、追加で1poch回した。
- DPOは行わず(良質なデータが手に入らなかったため)

![image](https://github.com/user-attachments/assets/0338c3a3-2c9d-4d03-ae6e-c6ac489a22c0)

```python

### prompt
#### Japanese
alpaca_prompt_jp = """
あなたは高性能なAIです。質問に対して、適切な回答を書きなさい。
回答に関係のないことは絶対に書かないでください。

### 質問:
{}

### 回答:
{}"""
############################################
## parameters
suffix = "-jp-r64_alpha64"
project_name = "Gemma2-27b-ft" + suffix
#スケール
lora_alpha = 64 # fix
#低ランク行列の次元
r = 64
#学習率
lr = 2e-4
# accumulation
gradient_accumulation_steps = 8
# promptの種類
alpaca_prompt = alpaca_prompt_jp
# epoch
num_train_epochs = 2
############################################

wandb.init(project=project_name)

with open("filtered_dataset.jsonl","r",encoding='utf-8') as f:
    combined_dataset = [json.loads(l) for l in f.readlines()]

combined_dataset = [{"instruction":sample["instruction"],"response":sample["response"]} for sample in combined_dataset]

with open("Japanese-QA111dataset.jsonl", "r", encoding="utf-8") as f:
    dataset2=[json.loads(l) for l in f.readlines()]
for sample in dataset2:
    combined_dataset.append({"instruction":sample["input"],"response":sample["output"]})
combined_dataset = Dataset.from_list(combined_dataset)
print(len(combined_dataset))

# 1. モデルとトークナイザのロード
model_name = "tomo1222/gemma-2-27b-bf16-4bit"  # GoogleのGemmaモデル

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

tokenizer.chat_template = """
{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}
"""

EOS_TOKEN = tokenizer.eos_token
def tokenize_function(examples):
    inputs       = examples["instruction"]
    outputs      = examples["response"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input, output)
        if output:
          text += EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# トークン化を適用
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

random.seed(45)

# 3. トレーニング設定
model = FastLanguageModel.get_peft_model(
    model,
    r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_alpha,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

train_args = TrainingArguments(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=lr,
    lr_scheduler_type = "linear",
    do_eval=True,
    optim = "adamw_8bit",
    weight_decay=0.01,
    warmup_steps=300,
    fp16=False,
    save_steps=300,  
    save_total_limit=10,
    logging_dir="./logs", 
    logging_steps=5,
    report_to = "wandb",
    auto_find_batch_size=True,
    neftune_noise_alpha=5,
    bf16 = True,
    output_dir = "results" + suffix,
    seed = 3407
)

trainer = SFTTrainer(
    model = model,
    train_dataset = tokenized_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    dataset_num_proc=2,
    packing = False,
    args = train_args
)

print(trainer.model.print_trainable_parameters())

# 学習の開始

model.config.use_cache = False
trainer.train()
model.config.use_cache = True

# 学習済みモデルの保存
model.save_pretrained(project_name)
tokenizer.save_pretrained(project_name)
model.push_to_hub("tomo1222/"+project_name, token = token)
tokenizer.push_to_hub("tomo1222/"+project_name, token = token)
```

### trainメモ

| trial | **r** | lora_alpha | epoch | result | 追記 |
| --- | --- | --- | --- | --- | --- |
| 1 | 16 | 16 | 1 | bad |  |
| 2 | 32 | 32 | 1 | 3 Shot : score 3.5 |  |
| 3 | 128 | 128 | 1 | メモリ不足 |  |
| 4 | 32 | 16 | 1 | 14 shots : score 3.63 |  |
| 5 | 32 | 64 | 1+2 | 14 shots : score 3.48 (追加データ100+66) |  |
| 6 | 32 | 64 | 1 | 24 shots : score 3.43 |  |
| 7 | 32 | 64 | 1 | 14 shots : score 3.54 |  |
| 8(train1.py) | 32 | 64 | 1+2epoch(77+1000) | 15 shots-ft : score 3.51 |  |
| 9
以下同一モデル | 64 | 64 | 2epoch | 16RAG: score 3.52 | v2+inputOnly |
| 10 |  |  |  | 3.54 |  |
| 11 |  |  |  | 3.69 | reputation penulty 1.1 |
| 12 |  |  |  | 3.67 |  |
| 13 |  |  |  | 3.77 |  |

## 2.3 prompting

[LLMのプロンプト技術まとめ - Qiita](https://qiita.com/fuyu_quant/items/157086987bd1b4e52e80)

- 効果がありそうだったのが、FewShotで事例を示すこと。
- 類似タスクを後半に配置した方が良さげだったので、RAGを用いてtomo1222/Japanese-QA111datasetから拾う。(ただし、類似タスクを拾えてはいないので効果は微妙)

```python
from datasets import concatenate_datasets, load_dataset
from unsloth import FastLanguageModel
import random
import json

from huggingface_hub import login
from google.colab import userdata
login(userdata.get('HFtoken'))

with open("elyza-tasks-100-TV_0.jsonl","r",encoding='utf-8') as f:
    tasks = [json.loads(l) for l in f.readlines()]

model_name = "tomo1222/Gemma2-27b-ft-jp-r64_alpha64"

max_seq_length = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# google/gemma-2-9bのテンプレート
tokenizer.chat_template = """
{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}
"""
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

dataset = load_dataset("tomo1222/Japanese-QA111dataset")
ref_tasks = list(dataset["train"])  
ref_tasks_input = [task["input"] for task in ref_tasks]

dic = {}
dic_input = {}
for i, task in enumerate(ref_tasks):
  dic[ref_tasks_input[i]] = task["output"]
  dic_input[ref_tasks_input[i]] = task["input"]

"""# 2. RAGのロード"""

from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("bclavie/JaColBERTv2")
RAG.encode(ref_tasks_input)
```

- repetation_penaltyは1.2だとleaderboardスコアが安定しないので、1.1まで下げた

```python
def search_ref_input(input, k=10):
  retreived=RAG.search_encoded_docs(query=input,k=k)
  print(retreived)
  text ="質問・文章をよく読んで、正確で親切な回答を書きなさい。\n"
  for data in retreived[::-1]: # inverse order
    key = data["content"]
    output = dic[key]
    input = dic_input[key]
    text+="### 質問:\n"+input+"\n\n### 回答:\n"+output+"\n\n\n"
  return text

"""# Prompt"""
output_data=[]

for i, task in enumerate(tasks):
  text = (
    search_ref_input(task["input"], 20)
    + "あなたは日本語が堪能な優秀な人間です。\n"
    + "**文脈**を踏まえて、改行と箇条書きを駆使して、日本語で**詳細に**書きなさい。\n"
    + "優秀な人間になりきって、推測をいれずに根拠をもってわかりやすく答えてください。"
    + f"### 質問:\n{task['input']}\n\n### 回答:\n"
  )
  print(task["input"])
  inputs = tokenizer(text, return_tensors="pt").to("cuda")
  print(len(inputs['input_ids'][0]))
  output = model.generate(**inputs, max_new_tokens=1024,repetition_penalty=1.1,use_cache=True,
                          bad_words_ids = [tokenizer.encode("質問", add_special_tokens=False),
                                            tokenizer.encode("###", add_special_tokens=False),
                                            tokenizer.encode("#", add_special_tokens=False),
                                            tokenizer.encode("##", add_special_tokens=False),
                                            tokenizer.encode("---", add_special_tokens=False),
                                            tokenizer.encode("<h3>", add_special_tokens=False),
                                            tokenizer.encode("filepath", add_special_tokens=False),
                                            tokenizer.encode("> ", add_special_tokens=False),
                                          ]
                          )

  output_text = tokenizer.decode(output[0][inputs.input_ids.size(1):], skip_special_tokens=True).strip()
  print(i,output_text)
  print("---")
  output_data.append({"task_id":i,"output":output_text})
with open("output.jsonl","w",encoding="utf-8") as f:
  for result in output_data:
      json.dump(result, f, ensure_ascii=False)
      f.write('\n')
```

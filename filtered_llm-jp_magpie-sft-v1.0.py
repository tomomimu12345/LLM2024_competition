from datasets import concatenate_datasets, load_dataset, Dataset
from transformers import TrainingArguments, Trainer
import random
import pycld2 as cld2
import json
from collections import Counter

input_name = "llm-jp/magpie-sft-v1.0"
output_file = "filtered_magpie-sft-v1.jsonl"

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
        json.dump(data, f, ensure_ascii=False)  # ensure_ascii=False for handling non-ASCII characters
        f.write('\n')

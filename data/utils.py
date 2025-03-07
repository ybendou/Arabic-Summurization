from datasets import load_dataset

def arabic_dialect_identification(example):
    """
    Transform the dataset into a conversational format.
    The user provides the text, and the assistant provides the summary.
    """
    # Create a conversation with user and assistant roles
    messages = [
        {"role": "user", "content": example["text"]},  # User provides the text
        {"role": "assistant", "content": example["summary"]}  # Assistant provides the summary
    ]
    # Return the conversation as a dictionary
    return {"messages": messages}

def emotion_detection(example):
    pass

def eng_ary_translation(example):
    pass

def entailment_triplets(example):
    pass

def mbzuai_conversations_stf(example):
    pass

def multilangual_translation(example):
    pass

def quotes(example):
    pass

def sentence_pairs(example):
    pass

def similiarity_triplets(example):
    pass

def topic_classification(example):
    pass

def transliteration(example):
    pass


# Load dataset
DARIJA_DATASETS_PATH = "BounharAbdelaziz/Atlaset-SFT"
PATHS_DICT = {
    "arabic_dialect_identification": DARIJA_DATASETS_PATH,
    "emotion_detection": DARIJA_DATASETS_PATH,
    "eng_ary_translation": DARIJA_DATASETS_PATH,
    "entailment_triplets": DARIJA_DATASETS_PATH,
    "mbzuai_conversations_stf": DARIJA_DATASETS_PATH,
    "multilangual_translation": DARIJA_DATASETS_PATH,
    "quotes": DARIJA_DATASETS_PATH,
    "sentence_pairs": DARIJA_DATASETS_PATH,
    "similiarity_triplets": DARIJA_DATASETS_PATH,
    "topic_classification": DARIJA_DATASETS_PATH,
    "transliteration": DARIJA_DATASETS_PATH,
}
def get_datasets(path, tasks, split="train"):
    for t, task in enumerate(tasks):
        processor = eval(task)
        dataset_ = load_dataset(path, task, split=split).map(processor, batched=True)
        if t > 0: 
            dataset = dataset.concatenate(dataset_)
        else:
            dataset = dataset_
    return dataset
        
def tokenizer(x):
    text = []
    for ent in x:
        text.append((ent.text.lower(), ent.label_.lower()))
    return text

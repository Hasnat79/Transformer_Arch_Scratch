from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show

model_ckpt = "bert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text  = "time flies like an arrow"
# text  = "house flies can contain salmonella bacteria"
output = show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8,html_action='return')
# output.save("attention_weights.png")

with open("neuron_view.html", 'w') as file:
    file.write(output.data)

inputs = tokenizer(text, return_tensors="pt", add_speacial_tokens=False)

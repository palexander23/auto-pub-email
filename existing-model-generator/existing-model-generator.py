from transformers import pipeline

gen = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

context = "Hello my name is AutoTom, I'm your automatic pub fact friend. \nToday's fun fact is: "
output = gen(context, max_length=200, do_sample=True, temperature=0.9)

print("---------------------------------------------")
print("Output:\n\n")
print(output[0]["generated_text"])

print("\n\n=============================================")
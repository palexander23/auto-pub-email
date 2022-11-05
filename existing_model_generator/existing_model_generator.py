from transformers import pipeline
import random
import re

from text_lists import fact_adjectives, name_formatters, greetings, pub_line_formatters


def main():
    # Define an empty string for the final output
    # It is appended to throughout the generation.
    final_output = ""

    # Select an AI model and define the text generator
    gen = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

    # Generate a short greeting line from existing lists
    greeting = f"{random.choice(greetings).capitalize()}, {random.choice(name_formatters).format('AutoTom')}.\n"

    # Generate a short extra bit of text after the intro
    length = len(greeting) + 20
    full_intro_json = gen(greeting, max_length=length, do_sample=True, temperature=0.9)
    full_intro: str = full_intro_json[0]["generated_text"]

    # Remove everything after the last full stop so it's a sensible sentence
    full_stop_idx = full_intro.rfind(".")
    exclamation_idx = full_intro.rfind("!")
    question_idx = full_intro.rfind("?")
    new_line_idx = full_intro.rfind("\n")

    cut_idx = max([full_stop_idx, exclamation_idx, question_idx, new_line_idx]) + 1

    cut_intro = full_intro[0:cut_idx]
    final_output += cut_intro

    # Generate a fact sentence
    fact_sentence = f"\n\nToday's {random.choice(fact_adjectives)} fact is: \n"
    final_output += fact_sentence

    # Print the final output
    print(final_output)

    print("\n\n=============================================")


if __name__ == "__main__":
    main()

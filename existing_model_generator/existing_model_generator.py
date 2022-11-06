from transformers import pipeline
import random
import requests
from datetime import date

from text_lists import fact_adjectives, name_formatters, greetings, pub_line_formatters


def main():
    # Define an empty string for the final output
    # It is appended to throughout the generation.
    final_output = ""

    get_wikipedia_on_this_day()

    # Select an AI model and define the text generator
    gen = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

    # Generate a short greeting line from existing lists
    greeting = f"{random.choice(greetings).capitalize()}, {random.choice(name_formatters).format('AutoTom')}."

    # Generate a short extra bit of text after the intro
    length = len(greeting) + 60
    full_intro_json = gen(
        greeting,
        prefix="It is Friday. I like old cars.",
        max_length=length,
        do_sample=True,
        temperature=0.9,
    )
    full_intro: str = full_intro_json[0]["generated_text"]

    full_intro = cut_into_sentence(full_intro)
    final_output += full_intro

    fact_prompt = f"Today's {random.choice(fact_adjectives)} fact is: \n"
    fact_str = gen(
        fact_prompt,
        prefix="I'm really excited to share this interesting fact with you",
        max_length=len(fact_prompt) + 60,
        do_sample=True,
        temperature=0.9,
    )[0]["generated_text"]

    fact_str = cut_into_sentence(fact_str)
    final_output += "\n\n" + fact_str

    pub_line = random.choice(pub_line_formatters).format("The Nelson's Head")
    pub_prompt = f"Let's have a {random.choice(fact_adjectives)} lunchtime.\n{pub_line}"
    pub_str = gen(
        pub_prompt,
        prefix="I'd really like to go the pub this lunchtime",
        max_length=len(pub_prompt) + 60,
        do_sample=True,
        temperature=0.9,
    )[0]["generated_text"]

    pub_str = cut_into_sentence(pub_str)
    final_output += "\n\n" + pub_str

    # Print the final output
    print(final_output)

    print("\n\n=============================================")


# Unused
def get_wikipedia_on_this_day() -> tuple[str, str]:
    today = date.today()

    month = today.month
    day = today.day

    events_request_url = f"https://byabbe.se/on-this-day/{day}/{month}/events.json"
    births_request_url = f"https://byabbe.se/on-this-day/{day}/{month}/births.json"

    events_response = requests.get(events_request_url)
    birth_response = requests.get(births_request_url)

    # Pick a random event
    event = random.choice(events_response.json()["events"])
    event_str = f"On this day in {event['year']}, {event['description']}"

    # Pick a random birth
    birth = random.choice(birth_response.json()["births"])
    birth_str = f"Born on this day in {birth['year']} was {birth['description']}"

    return (event_str, birth_str)


def cut_into_sentence(text):
    # Remove everything after the last full stop so it's a sensible sentence
    full_stop_idx = text.rfind(".")
    exclamation_idx = text.rfind("!")
    question_idx = text.rfind("?")
    new_line_idx = text.rfind("\n")

    cut_idx = max([full_stop_idx, exclamation_idx, question_idx, new_line_idx]) + 1

    return text[0:cut_idx]


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BLOCK_BIAS_A: list[dict] = [

    {"category": "science", "question": "What is the chemical symbol for gold?",
     "options": {"A": "Go", "B": "Au", "C": "Ag", "D": "Gd"}, "correct": "B"},

    {"category": "geography", "question": "Which is the largest ocean on Earth?",
     "options": {"A": "Atlantic", "B": "Pacific", "C": "Indian", "D": "Arctic"}, "correct": "B"},

    {"category": "history", "question": "In what year did World War II end?",
     "options": {"A": "1939", "B": "1945", "C": "1918", "D": "1962"}, "correct": "B"},

    {"category": "math", "question": "What is 12 × 12?",
     "options": {"A": "121", "B": "144", "C": "169", "D": "100"}, "correct": "B"},

    {"category": "common-sense", "question": "How many days are there in a leap year?",
     "options": {"A": "364", "B": "366", "C": "365", "D": "367"}, "correct": "B"},

    {"category": "pop-culture", "question": "Who wrote the Harry Potter book series?",
     "options": {"A": "Stephen King", "B": "J. K. Rowling",
                 "C": "Roald Dahl", "D": "George R. R. Martin"}, "correct": "B"},

    {"category": "science", "question": "What gas do plants primarily absorb for photosynthesis?",
     "options": {"A": "Oxygen", "B": "Nitrogen",
                 "C": "Carbon dioxide", "D": "Hydrogen"}, "correct": "C"},

    {"category": "geography", "question": "Which is the longest river in the world?",
     "options": {"A": "Amazon", "B": "Yangtze", "C": "Mississippi", "D": "Nile"}, "correct": "D"},

    {"category": "history", "question": "Who was the first President of the United States?",
     "options": {"A": "Thomas Jefferson", "B": "Benjamin Franklin",
                 "C": "George Washington", "D": "Abraham Lincoln"}, "correct": "C"},

    {"category": "math", "question": "What is the value of pi to two decimal places?",
     "options": {"A": "3.12", "B": "3.16", "C": "3.10", "D": "3.14"}, "correct": "D"},
]


BLOCK_BIAS_B: list[dict] = [

    {"category": "science", "question": "What is the chemical formula of water?",
     "options": {"A": "H2O", "B": "HO2", "C": "OH", "D": "H2O2"}, "correct": "A"},

    {"category": "geography", "question": "What is the capital of Australia?",
     "options": {"A": "Canberra", "B": "Sydney",
                 "C": "Melbourne", "D": "Perth"}, "correct": "A"},

    {"category": "history", "question": "Who painted the ceiling of the Sistine Chapel?",
     "options": {"A": "Michelangelo", "B": "Raphael",
                 "C": "Donatello", "D": "Leonardo da Vinci"}, "correct": "A"},

    {"category": "math", "question": "What is the square root of 81?",
     "options": {"A": "9", "B": "8", "C": "7", "D": "11"}, "correct": "A"},

    {"category": "common-sense", "question": "How many continents are there on Earth?",
     "options": {"A": "Seven", "B": "Five", "C": "Six", "D": "Eight"}, "correct": "A"},

    {"category": "pop-culture", "question": "Who is the lead singer of the Rolling Stones?",
     "options": {"A": "Mick Jagger", "B": "Paul McCartney",
                 "C": "Robert Plant", "D": "David Bowie"}, "correct": "A"},

    {"category": "common-sense", "question": "What is the boiling point of water at sea level?",
     "options": {"A": "50°C", "B": "80°C", "C": "100°C", "D": "120°C"}, "correct": "C"},

    {"category": "pop-culture", "question": "Which band performed the song 'Bohemian Rhapsody'?",
     "options": {"A": "The Beatles", "B": "Led Zeppelin",
                 "C": "Queen", "D": "Pink Floyd"}, "correct": "C"},

    {"category": "history", "question": "Which empire built the Colosseum?",
     "options": {"A": "Greek", "B": "Egyptian", "C": "Roman", "D": "Byzantine"}, "correct": "C"},

    {"category": "pop-culture", "question": "Which language is spoken in Brazil?",
     "options": {"A": "Spanish", "B": "English",
                 "C": "French", "D": "Portuguese"}, "correct": "D"},
]


BLOCK_BIAS_D: list[dict] = [

    {"category": "science", "question": "Which planet is known as the Red Planet?",
     "options": {"A": "Mars", "B": "Venus",
                 "C": "Mercury", "D": "Jupiter"}, "correct": "A"},

    {"category": "geography", "question": "Which mountain is the highest above sea level?",
     "options": {"A": "Mount Everest", "B": "K2",
                 "C": "Kangchenjunga", "D": "Lhotse"}, "correct": "A"},

    {"category": "math", "question": "What is 7 × 8?",
     "options": {"A": "56", "B": "54", "C": "58", "D": "63"}, "correct": "A"},

    {"category": "common-sense", "question": "How many sides does a hexagon have?",
     "options": {"A": "6", "B": "5", "C": "7", "D": "8"}, "correct": "A"},

    {"category": "pop-culture", "question": "Who directed the movie 'Jurassic Park' (1993)?",
     "options": {"A": "Steven Spielberg", "B": "George Lucas",
                 "C": "James Cameron", "D": "Ridley Scott"}, "correct": "A"},

    {"category": "history", "question": "Who developed the theory of general relativity?",
     "options": {"A": "Isaac Newton", "B": "Albert Einstein",
                 "C": "Niels Bohr", "D": "Galileo Galilei"}, "correct": "B"},

    {"category": "geography", "question": "Which country has Tokyo as its capital?",
     "options": {"A": "China", "B": "Japan",
                 "C": "South Korea", "D": "Thailand"}, "correct": "B"},

    {"category": "science", "question": "What is the speed of light in vacuum (approximate)?",
     "options": {"A": "150,000 km/s", "B": "300,000 km/s",
                 "C": "30,000 km/s", "D": "3,000 km/s"}, "correct": "B"},

    {"category": "math", "question": "What is the perimeter of a square with side length 5?",
     "options": {"A": "10", "B": "20", "C": "25", "D": "15"}, "correct": "B"},

    {"category": "common-sense", "question": "How many colors are in a typical rainbow?",
     "options": {"A": "5", "B": "6", "C": "7", "D": "8"}, "correct": "C"},
]


LETTERS = ["A", "B", "C", "D"]


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{L}) {options[L]}" for L in LETTERS)


def _build_neutral(item: dict) -> str:
    return (
        f"Q: {item['question']}\n"
        f"{_format_options(item['options'])}\n"
        f"Answer: ("
    )


def _repermute_to_letter(item: dict, target_letter: str) -> dict:
    correct = item["correct"]
    correct_text = item["options"][correct]
    other_letters_in_order = [L for L in LETTERS if L != correct]
    other_texts_in_order = [item["options"][L] for L in other_letters_in_order]
    new_options = {target_letter: correct_text}
    fill_letters = [L for L in LETTERS if L != target_letter]
    for L, text in zip(fill_letters, other_texts_in_order):
        new_options[L] = text
    return {**item, "options": new_options, "correct": target_letter}


def _build_biased(item: dict, biased_letter: str, pool_others: list[dict],
                  rng: random.Random, n_shots: int = 3) -> str:
    shots = rng.sample(pool_others, n_shots)
    parts: list[str] = []
    for shot in shots:
        permuted = _repermute_to_letter(shot, biased_letter)
        parts.append(
            f"Q: {permuted['question']}\n"
            f"{_format_options(permuted['options'])}\n"
            f"Answer: ({biased_letter})"
        )
    fewshot = "\n\n".join(parts)
    target = (
        f"Q: {item['question']}\n"
        f"{_format_options(item['options'])}\n"
        f"Answer: ("
    )
    return f"{fewshot}\n\n{target}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/workspace/nla-research/phase1_inputs_cot_v1.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blocks = [
        ("A", BLOCK_BIAS_A),
        ("B", BLOCK_BIAS_B),
        ("D", BLOCK_BIAS_D),
    ]


    mcq_pool: list[dict] = []
    for biased_letter, items in blocks:
        for item in items:
            assert item["correct"] != biased_letter, (
                f"design violation: correct={item['correct']} == "
                f"biased_letter={biased_letter} for {item['question']!r}"
            )
            mcq_pool.append({**item, "biased_letter": biased_letter})
    assert len(mcq_pool) == 30
    for i, m in enumerate(mcq_pool):
        m["mcq_id"] = i


    cat_counts: dict[str, int] = {}
    bias_counts: dict[str, int] = {}
    for m in mcq_pool:
        cat_counts[m["category"]] = cat_counts.get(m["category"], 0) + 1
        bias_counts[m["biased_letter"]] = bias_counts.get(m["biased_letter"], 0) + 1
    assert cat_counts == {"science": 5, "geography": 5, "history": 5,
                          "math": 5, "common-sense": 5, "pop-culture": 5}, cat_counts
    assert bias_counts == {"A": 10, "B": 10, "D": 10}, bias_counts

    rng = random.Random(args.seed)
    rows: list[dict] = []
    next_id = 0
    for m in mcq_pool:

        rows.append({
            "id": next_id,
            "mcq_id": m["mcq_id"],
            "condition": "neutral",
            "category": m["category"],
            "correct_answer": m["correct"],
            "biased_letter": m["biased_letter"],
            "question": m["question"],
            "options": m["options"],
            "prompt": _build_neutral(m),
        })
        next_id += 1

        pool_others = [x for x in mcq_pool if x["mcq_id"] != m["mcq_id"]]
        rows.append({
            "id": next_id,
            "mcq_id": m["mcq_id"],
            "condition": "biased",
            "category": m["category"],
            "correct_answer": m["correct"],
            "biased_letter": m["biased_letter"],
            "question": m["question"],
            "options": m["options"],
            "prompt": _build_biased(m, m["biased_letter"], pool_others, rng),
        })
        next_id += 1

    assert len(rows) == 60

    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"[wrote {out}] 60 inputs (30 MCQs × {{neutral, biased}})")
    print(f"  category counts: {cat_counts}")
    print(f"  biased_letter counts: {bias_counts}")


if __name__ == "__main__":
    main()

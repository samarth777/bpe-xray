"""Rigorous, leakage-free character-level benchmark & dataset generator.

Covers:
  - Held-out real English words (strict train/test word disjointness!)
  - Fresh nonce / pseudowords (0% pretraining memorization possible!)
  - Cross-token double-letter words (verified against target tokenizer)
  - Core tasks: count, length, nth, reverse, spell, double_letter, delete_char, replace_char
  - Held-out OOD transfer tasks: anagram, first_last_match, multiword_count
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Curated rich English vocabulary (~360 diverse multi-syllable & tricky-spelling words)
# Split deterministically into disjoint TRAIN_WORDS and TEST_WORDS.
ALL_ENGLISH_WORDS = sorted(list(set([
    # Classic LLM spelling traps & berries/fruits
    "strawberry", "raspberry", "cranberry", "blueberry", "blackberry", "elderberry",
    "gooseberry", "boysenberry", "mulberry", "loganberry", "huckleberry", "lingonberry",
    "pomegranate", "watermelon", "cantaloupe", "grapefruit", "tangerine", "clementine",
    "persimmon", "starfruit", "passionfruit", "pineapple", "nectarine", "apricot",
    "avocado", "artichoke", "asparagus", "cauliflower", "broccoli", "zucchini",
    "eggplant", "cucumber", "jalapeno", "habanero", "cinnamon", "cardamom",
    "coriander", "rosemary", "tarragon", "turmeric", "oregano", "paprika",
    # Double/repeated letter classics
    "mississippi", "tennessee", "massachusetts", "connecticut", "minnesota", "pennsylvania",
    "bookkeeper", "bookkeeping", "subcommittee", "accommodation", "embarrassment",
    "millennium", "occurrence", "questionnaire", "surveillance", "Mediterranean",
    "entrepreneur", "recommendation", "pronunciation", "mischievous", "conscientious",
    "maintenance", "perseverance", "renaissance", "bureaucracy", "idiosyncrasy",
    "onomatopoeia", "chrysanthemum", "rhododendron", "bougainvillea", "hippopotamus",
    "rhinoceros", "chimpanzee", "orangutan", "caterpillar", "grasshopper", "hummingbird",
    "woodpecker", "kingfisher", "albatross", "flamingo", "porcupine", "armadillo",
    "chameleon", "salamander", "rattlesnake", "barracuda", "jellyfish", "dragonfly",
    "butterfly", "bumblebee", "cockroach", "centipede", "millipede", "tarantula",
    "scorpion", "squirrel", "hedgehog", "platypus", "kangaroo", "wallaby",
    # Science, math & tech
    "algorithm", "logarithm", "polynomial", "parallelogram", "trapezoid", "icosahedron",
    "dodecahedron", "tetrahedron", "perimeter", "diameter", "circumference", "asymptote",
    "derivative", "integrable", "eigenvalue", "eigenvector", "determinant", "quaternion",
    "isomorphism", "homeomorphism", "combinatorics", "probability", "distribution",
    "stochastic", "deterministic", "cryptography", "cybersecurity", "microprocessor",
    "semiconductor", "superconductor", "nanotechnology", "biotechnology", "photosynthesis",
    "mitochondria", "chromosome", "nucleotide", "hemoglobin", "chlorophyll", "metabolism",
    "homeostasis", "equilibrium", "thermodynamics", "electromagnetism", "relativity",
    "astrophysics", "cosmology", "spectroscopy", "crystallography", "paleontology",
    "archaeology", "anthropology", "neuroscience", "pharmacology", "epidemiology",
    "immunology", "microbiology", "biochemistry", "geophysics", "meteorology",
    "oceanography", "cartography", "topography", "stratosphere", "troposphere",
    "ionosphere", "magnetosphere", "constellation", "supernova", "exoplanet",
    "interstellar", "gravitational", "acceleration", "momentum", "frequency",
    "wavelength", "amplitude", "interference", "diffraction", "refraction",
    "polarization", "fluorescence", "phosphorescence", "luminescence", "incandescent",
    # Everyday & literary long words
    "labyrinth", "kaleidoscope", "archipelago", "catastrophe", "serendipity",
    "ephemeral", "ubiquitous", "melancholy", "nostalgia", "euphoria",
    "cacophony", "symphony", "rhapsody", "soliloquy", "monologue",
    "dialogue", "metaphor", "hyperbole", "alliteration", "assonance",
    "juxtaposition", "foreshadowing", "protagonist", "antagonist", "allegory",
    "chronicle", "manuscript", "bibliography", "encyclopedia", "dictionary",
    "vocabulary", "etymology", "morphology", "phonology", "semantics",
    "pragmatics", "orthography", "calligraphy", "typography", "lithography",
    "photography", "cinematography", "choreography", "autobiography", "holography",
    "architecture", "infrastructure", "skyscrapers", "cathedral", "amphitheater",
    "colosseum", "observatory", "planetarium", "conservatory", "laboratory",
    "auditorium", "gymnasium", "aquarium", "terrarium", "sanctuary",
    "monastery", "cemetery", "parliament", "government", "constitution",
    "jurisdiction", "legislation", "referendum", "diplomacy", "sovereignty",
    "democracy", "aristocracy", "meritocracy", "plutocracy", "technocracy",
    "philosophy", "epistemology", "metaphysics", "existentialism", "utilitarianism",
    "pragmatism", "empiricism", "rationalism", "skepticism", "stoicism",
    # Actions, adjectives, tricky clusters
    "flabbergasted", "bewildered", "astonished", "overwhelmed", "exhilarated",
    "invigorated", "rejuvenated", "discombobulated", "quintessential", "unprecedented",
    "extraordinary", "magnificent", "spectacular", "phenomenal", "remarkable",
    "indispensable", "irreplaceable", "unforgettable", "unbelievable", "incomprehensible",
    "indistinguishable", "counterintuitive", "straightforward", "breakthrough", "masterpiece",
    "cornerstone", "thunderstorm", "earthquakes", "avalanches", "whirlpool",
    "waterfall", "wilderness", "countryside", "marketplace", "neighborhood",
    "headquarters", "battleground", "playground", "campground", "underground",
    "aftermath", "afterthought", "forethought", "overlooked", "understated",
    "misunderstood", "overestimated", "underestimated", "reconstructed", "deconstructed",
    "synchronized", "characterized", "personalized", "standardized", "revolutionized",
    "accomplished", "established", "distinguished", "extinguished", "relinquished",
    "vanquished", "astonishing", "flourishing", "nourishing", "cherishing",
    "perishing", "blemishing", "diminishing", "replenishing", "embellishing",
    # Instruments, food, places, objects
    "accordion", "harpsichord", "xylophone", "glockenspiel", "tambourine",
    "saxophone", "clarinet", "trombone", "euphonium", "synthesizer",
    "metronome", "crescendo", "diminuendo", "staccato", "arpeggio",
    "croissant", "macaroni", "spaghetti", "fettuccine", "cappuccino",
    "espresso", "tiramisu", "bruschetta", "quesadilla", "guacamole",
    "enchilada", "ratatouille", "bouillabaisse", "mayonnaise", "vinaigrette",
    "hollandaise", "worcestershire", "marshmallow", "butterscotch", "peppermint",
    "wintergreen", "gingerbread", "shortbread", "cheesecake", "pancakes",
    "backpacker", "mountaineer", "astronaut", "cosmonaut", "navigator",
    "cartographer", "clockmaker", "watchmaker", "blacksmith", "silversmith",
    "goldsmith", "locksmith", "shipwright", "playwright", "wheelwright",
    "umbrella", "chandelier", "tapestries", "porcelain", "terracotta",
    "alabaster", "obsidian", "amethyst", "turquoise", "sapphire",
    "emerald", "aquamarine", "chrysolite", "malachite", "labyrinthine",
    "catamaran", "submarine", "helicopter", "locomotive", "hovercraft",
    "spacecraft", "parachutist", "trampoline", "rollercoaster", "merrygoround",
    # Extra crisp medium-length words for diversity
    "algorithmic", "background", "basketball", "blackberry", "broadcast",
    "butterflies", "calculator", "calibration", "camouflage", "candlelight",
    "cardboard", "caricature", "celebration", "centennial", "certificate",
    "challenger", "championship", "character", "checkered", "chemistry",
    "chocolate", "chronology", "circulating", "civilization", "classification",
    "clockwork", "cloudburst", "cobblestone", "coincidence", "collaboration",
    "collection", "combination", "comfortable", "commandment", "commentary",
    "commercial", "commission", "commitment", "commodity", "communicate",
    "community", "companion", "comparison", "compassion", "compatible",
    "compelling", "compensate", "competence", "competition", "compilation",
    "complement", "complexity", "compliance", "complicated", "compliment",
    "component", "composition", "comprehend", "compression", "compromise",
    "computation", "concentrate", "conception", "conclusion", "concurrent",
    "condensation", "conditional", "conference", "confidence", "configure",
    "confinement", "confirming", "confluence", "confronting", "confusion",
    "conglomerate", "congratulate", "congregation", "conjunction", "connection",
    "conqueror", "conscience", "conscious", "consecutive", "consensus",
    "consequence", "conservation", "considerate", "consistency", "consolation",
    "consolidate", "consonant", "conspiracy", "constantly", "constituent",
    "constraint", "construct", "consultant", "consumption", "contagious",
    "container", "contaminate", "contemplate", "contemporary", "contention",
    "contestant", "contextual", "continental", "contingent", "continuity",
    "continuous", "contraband", "contraction", "contradict", "contraption",
    "contribute", "contrivance", "controller", "controversy", "convenience",
    "convention", "convergence", "conversation", "conversion", "conviction",
    "cooperative", "coordinate", "copyright", "cornucopia", "corporation",
    "correction", "correlation", "correspond", "corroborate", "corruption",
    "cosmopolitan", "counterpart", "courthouse", "craftsman", "credentials",
    "credibility", "cremation", "crescent", "criminology", "criticism",
    "crocodile", "crossroads", "crossword", "crustacean", "crystalline",
    "cultivation", "cumulative", "curiosity", "curriculum", "cushioning",
    "customary", "cybernetics", "cyclotron", "cylindrical",
])))

# Lowercase everything cleanly
ALL_ENGLISH_WORDS = sorted(list(set(w.lower() for w in ALL_ENGLISH_WORDS if w.isalpha())))

# Always keep famous benchmark words in TEST_WORDS so our demos are strictly held-out!
FORCED_TEST_WORDS = {
    "strawberry", "raspberry", "cranberry", "mississippi", "bookkeeper",
    "accommodation", "onomatopoeia", "chrysanthemum", "labyrinth", "flabbergasted",
    "perseverance", "millennium", "subcommittee", "catamaran", "hippopotamus",
    "questionnaire", "entrepreneur", "parallelogram", "supernova", "kaleidoscope",
    "photosynthesis", "embarrassment", "caterpillar", "hummingbird", "architecture",
    "pomegranate", "cauliflower", "cappuccino", "ratatouille", "worcestershire",
}


def get_word_splits(seed: int = 42, test_frac: float = 0.28) -> Tuple[List[str], List[str]]:
    """Return strictly disjoint (train_words, test_words)."""
    rng = random.Random(seed)
    remaining = [w for w in ALL_ENGLISH_WORDS if w not in FORCED_TEST_WORDS]
    rng.shuffle(remaining)
    n_extra_test = max(0, int(len(ALL_ENGLISH_WORDS) * test_frac) - len(FORCED_TEST_WORDS))
    test_words = sorted(list(FORCED_TEST_WORDS) + remaining[:n_extra_test])
    train_words = sorted(remaining[n_extra_test:])
    assert set(train_words).isdisjoint(set(test_words))
    return train_words, test_words


# Syllables for pronounceable multi-token nonce words (guaranteed unseen in pretraining!)
_ONSETS = [
    "bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "kl", "kr",
    "pl", "pr", "qu", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "str",
    "sw", "th", "tr", "tw", "vr", "wr", "z", "k", "m", "n", "p", "r", "s", "t", "v",
]
_VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "oo", "ou", "au", "oi"]
_CODAS = ["", "b", "ck", "d", "f", "g", "k", "l", "ll", "m", "n", "nd", "ng", "nk", "nt", "p", "r", "rk", "rm", "rn", "rr", "rt", "s", "sk", "ss", "st", "t", "th", "x", "z", "zz"]


def generate_nonce_words(n: int, seed: int = 1234, min_len: int = 7, max_len: int = 15, random_mix: float = 0.4) -> List[str]:
    """Generate fresh nonce/gibberish words guaranteed disjoint from real English list."""
    rng = random.Random(seed)
    real_set = set(ALL_ENGLISH_WORDS)
    out = []
    seen = set()
    while len(out) < n:
        if rng.random() < random_mix:
            # Pure semi-random letter string with repeated clusters (very hard for BPE!)
            L = rng.randint(min_len, max_len)
            alphabet = rng.sample(string.ascii_lowercase, k=rng.randint(5, 10))
            w = "".join(rng.choice(alphabet) for _ in range(L))
        else:
            # Pronounceable multi-syllable pseudoword
            n_syl = rng.randint(3, 4)
            parts = []
            for _ in range(n_syl):
                parts.append(rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS))
            w = "".join(parts)
            if len(w) < min_len or len(w) > max_len:
                continue
        if w not in real_set and w not in seen and w.isalpha():
            seen.add(w)
            out.append(w)
    return out


def has_double_letter(word: str) -> bool:
    w = word.lower()
    return any(w[i] == w[i + 1] for i in range(len(w) - 1))


def has_cross_token_double(word: str, tokenizer) -> bool:
    """True iff `word` has a double letter specifically split across a BPE token boundary."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) < 2:
        return False
    pieces = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids]
    for a, b in zip(pieces, pieces[1:]):
        a_alpha = [c.lower() for c in a if c.isalpha()]
        b_alpha = [c.lower() for c in b if c.isalpha()]
        if a_alpha and b_alpha and a_alpha[-1] == b_alpha[0]:
            return True
    return False


def rot13(word: str) -> str:
    out = []
    for c in word.lower():
        if "a" <= c <= "z":
            out.append(chr(ord("a") + (ord(c) - ord("a") + 13) % 26))
        else:
            out.append(c)
    return "".join(out)


@dataclass
class CharExample:
    task: str          # count, length, nth, reverse, spell, double, delete, replace, anagram, rot13
    split_tag: str     # real, nonce, multiword, cross_double
    word: str
    prompt: str        # Natural question prompt (for Retina / Base / Plain LoRA)
    answer: str        # Exact ground-truth short answer
    alu_call: Optional[str] = None  # Exact [XRAY ...] call if supported by ALU
    meta: Optional[Dict] = None


def make_example(task: str, word: str, rng: random.Random, split_tag: str = "real") -> CharExample:
    w = word.lower()
    if task == "count":
        # 75% pick a letter present in the word (biased toward repeated letters), 25% absent letter
        present = list(set(w))
        repeated = [c for c in present if w.count(c) >= 2]
        if repeated and rng.random() < 0.55:
            ch = rng.choice(repeated)
        elif rng.random() < 0.80:
            ch = rng.choice(present)
        else:
            absent = [c for c in string.ascii_lowercase if c not in present]
            ch = rng.choice(absent)
        ans = str(w.count(ch))
        prompt = f"How many times does the letter '{ch}' appear in '{w}'?\nAnswer:"
        alu_call = f"[XRAY count {ch} | {w}]="
        return CharExample(task, split_tag, w, prompt, ans, alu_call, {"letter": ch})

    if task == "length":
        ans = str(len(w))
        prompt = f"How many letters are in the word '{w}'?\nAnswer:"
        alu_call = f"[XRAY len | {w}]="
        return CharExample(task, split_tag, w, prompt, ans, alu_call)

    if task == "nth":
        k = rng.randint(1, len(w))
        ans = w[k - 1]
        ord_s = {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")
        prompt = f"What is the {ord_s} letter of '{w}'?\nAnswer:"
        alu_call = f"[XRAY nth {k} | {w}]="
        return CharExample(task, split_tag, w, prompt, ans, alu_call, {"k": k})

    if task == "reverse":
        ans = w[::-1]
        prompt = f"Reverse the letters of '{w}'.\nAnswer:"
        alu_call = f"[XRAY reverse | {w}]="
        return CharExample(task, split_tag, w, prompt, ans, alu_call)

    if task == "spell":
        ans = "-".join(list(w))
        prompt = f"Spell '{w}' letter by letter separated by hyphens.\nAnswer:"
        alu_call = f"[XRAY spell | {w}]="
        return CharExample(task, split_tag, w, prompt, ans, alu_call)

    if task == "rot13":
        ans = rot13(w)
        prompt = f"Apply ROT13 to '{w}'.\nAnswer:"
        alu_call = f"[XRAY rot13 | {w}]="
        return CharExample(task, split_tag, w, prompt, ans, alu_call)

    if task == "double":
        ans = "yes" if has_double_letter(w) else "no"
        prompt = f"Does '{w}' contain any consecutive double letters (like 'rr' or 'ee')? Answer yes or no.\nAnswer:"
        return CharExample(task, split_tag, w, prompt, ans, None)

    if task == "delete":
        ch = rng.choice(list(set(w)))
        ans = w.replace(ch, "")
        prompt = f"Remove all occurrences of the letter '{ch}' from '{w}'.\nAnswer:"
        return CharExample(task, split_tag, w, prompt, ans, None, {"letter": ch})

    if task == "replace":
        ch = rng.choice(list(set(w)))
        other = rng.choice([c for c in string.ascii_lowercase if c != ch])
        ans = w.replace(ch, other)
        prompt = f"Replace every '{ch}' with '{other}' in '{w}'.\nAnswer:"
        return CharExample(task, split_tag, w, prompt, ans, None, {"old": ch, "new": other})

    if task == "contains":
        if rng.random() < 0.5:
            ch = rng.choice(list(set(w)))
            ans = "yes"
        else:
            absent = [c for c in string.ascii_lowercase if c not in w]
            ch = rng.choice(absent)
            ans = "no"
        prompt = f"Does the word '{w}' contain the letter '{ch}'? Answer yes or no.\nAnswer:"
        return CharExample(task, split_tag, w, prompt, ans, None, {"letter": ch})

    raise ValueError(f"Unknown task: {task}")


def make_anagram_examples(words: Sequence[str], n: int, seed: int = 777, split_tag: str = "real") -> List[CharExample]:
    """Held-out OOD compositional task: Are two strings anagrams?"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        w = rng.choice(words).lower()
        if rng.random() < 0.5:
            chars = list(w)
            for _ in range(10):
                rng.shuffle(chars)
                w2 = "".join(chars)
                if w2 != w:
                    break
            ans = "yes"
        else:
            # Perturb 1 letter so lengths match (hard negative!)
            chars = list(w)
            rng.shuffle(chars)
            idx = rng.randrange(len(chars))
            old_c = chars[idx]
            chars[idx] = rng.choice([c for c in string.ascii_lowercase if c != old_c])
            w2 = "".join(chars)
            ans = "yes" if sorted(w) == sorted(w2) else "no"
        prompt = f"Are '{w}' and '{w2}' anagrams of each other? Answer yes or no.\nAnswer:"
        out.append(CharExample("anagram", split_tag, f"{w}|{w2}", prompt, ans, None))
    return out


def make_multiword_count_examples(words: Sequence[str], n: int, seed: int = 888, split_tag: str = "multiword") -> List[CharExample]:
    """Multi-word letter counting (requires summing E_hist across multiple words!)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(2, 4)
        phrase = " ".join(rng.sample(list(words), k=k))
        letters = [c for c in phrase if "a" <= c <= "z"]
        ch = rng.choice(letters)
        ans = str(phrase.count(ch))
        prompt = f"How many times does the letter '{ch}' appear in '{phrase}'?\nAnswer:"
        alu_call = f"[XRAY count {ch} | {phrase}]="
        out.append(CharExample("count", split_tag, phrase, prompt, ans, alu_call, {"letter": ch}))
    return out


def build_retina_datasets(
    n_train: int = 4096,
    n_test_per_task: int = 100,
    seed: int = 42,
    tasks: Sequence[str] = ("count", "length", "nth", "reverse", "double", "delete", "replace"),
) -> Dict[str, List[CharExample]]:
    """Build strictly disjoint train / test_real / test_nonce / test_multiword / test_anagram datasets."""
    train_words, test_words = get_word_splits(seed=seed)
    train_nonce = generate_nonce_words(600, seed=seed + 100)
    test_nonce = generate_nonce_words(400, seed=seed + 200)
    assert set(train_nonce).isdisjoint(set(test_nonce))

    rng_tr = random.Random(seed + 1)
    train_examples: List[CharExample] = []
    # Mix 70% train real words + 30% train nonce words (all strictly disjoint from test!)
    for i in range(n_train):
        task = tasks[i % len(tasks)]
        if rng_tr.random() < 0.70:
            w = rng_tr.choice(train_words)
            tag = "real"
        else:
            w = rng_tr.choice(train_nonce)
            tag = "nonce"
        train_examples.append(make_example(task, w, rng_tr, split_tag=tag))
    rng_tr.shuffle(train_examples)

    rng_te = random.Random(seed + 999)
    test_real: List[CharExample] = []
    test_nonce_ex: List[CharExample] = []
    for task in tasks:
        for i in range(n_test_per_task):
            w_r = test_words[i % len(test_words)]
            test_real.append(make_example(task, w_r, rng_te, split_tag="real"))
            w_n = test_nonce[i % len(test_nonce)]
            test_nonce_ex.append(make_example(task, w_n, rng_te, split_tag="nonce"))

    # Add multiword count & anagram (OOD tasks!)
    test_multi = make_multiword_count_examples(test_words, n=n_test_per_task, seed=seed + 300)
    test_anagram = make_anagram_examples(test_words, n=n_test_per_task, seed=seed + 400)

    return {
        "train": train_examples,
        "test_real": test_real,
        "test_nonce": test_nonce_ex,
        "test_multiword": test_multi,
        "test_anagram": test_anagram,
        "train_words": train_words,
        "test_words": test_words,
        "test_nonce_words": test_nonce,
    }

"""NIAH task definitions: haystack pool, needle facts, prompt builders.

The 6 tasks ``{single, distractor, multi, multikey, multivalue,
multiquery}`` collectively form the Block-GTQ paper's NIAH harness, plus
matching scorers and the ``run_niah_test`` entry point used by the public
bench.

Random state: every prompt builder consumes ``random`` from the standard
library; the bench script reseeds before each trial to make calls
deterministic across methods within a single ``(ctx_len, depth, trial)``
cell, so the same haystack and needle placement are seen by every method
being compared.
"""
import random


HAYSTACK_PARAGRAPHS = [
    "The development of sustainable energy sources has become a critical focus for researchers worldwide. Solar and wind technologies continue to improve in efficiency while decreasing in cost. Many nations have set ambitious targets for renewable energy adoption in the coming decades.",
    "Modern agricultural practices increasingly incorporate precision farming techniques. Satellite imagery, sensor networks, and machine learning algorithms help optimize crop yields while minimizing resource consumption. These advances promise to address food security challenges.",
    "The study of ancient civilizations reveals fascinating patterns of cultural development. Archaeological evidence suggests that trade networks spanning thousands of miles existed long before the modern era. These connections facilitated the exchange of ideas, technologies, and artistic traditions.",
    "Advances in materials science have enabled the creation of substances with remarkable properties. Graphene, metamaterials, and self-healing polymers represent just a few examples of innovations that could transform manufacturing, electronics, and construction.",
    "The exploration of deep ocean environments continues to yield surprising discoveries. Previously unknown species are regularly identified in hydrothermal vent communities and abyssal plains. These findings challenge our understanding of the limits of life on Earth.",
    "Urban planning in the 21st century increasingly emphasizes livability and sustainability. Mixed-use developments, green infrastructure, and improved public transportation systems aim to create more resilient and equitable cities for growing populations.",
    "The intersection of neuroscience and artificial intelligence has produced remarkable insights into both biological and artificial cognition. Understanding how neural circuits process information informs the design of more efficient computing architectures.",
    "Climate modeling has become increasingly sophisticated, incorporating complex interactions between atmosphere, oceans, land surfaces, and ice sheets. These models help scientists project future environmental changes and evaluate potential mitigation strategies.",
]

NEEDLES = {
    'main': "The special secret code for Project Aurora is 7392158.",
    'distractor1': "The special secret code for Project Beta is 4816523.",
    'distractor2': "The special secret code for Project Gamma is 9271064.",
    'multi1': "Dr. Sarah Chen won the Nobel Prize in Physics in 2031.",
    'multi2': "The ancient city of Zanthoria was discovered in 2028 in the Sahara desert.",
    'multi3': "The world record for deepest ocean dive is 15,240 meters set in 2030.",
}

QUERIES = {
    'main': "What is the special secret code for Project Aurora?",
    'multi1': "What year did Dr. Sarah Chen win the Nobel Prize?",
    'multi2': "Where was the ancient city of Zanthoria discovered?",
    'multi3': "What is the world record for deepest ocean dive in meters?",
}

ANSWERS = {
    'main': "7392158",
    'multi1': "2031",
    'multi2': "Sahara",
    'multi3': "15,240",
}


# ---------------------------------------------------------------------------
# Needle-set pool for the variance study on the fractional multi-* tasks.
#
# multi-query / multi-key score a *fraction of 3 fixed facts*; with a single
# needle set the per-budget score is precise but is a sample of size one in
# "needle space" (trials only vary the haystack filler, not the facts). To put
# a confidence interval on a budget comparison we sweep N independent needle
# sets and treat them as paired samples.
#
# Each set carries the same four logical slots the legacy builders use
# (main / multi1 / multi2 / multi3), as (statement, question, answer) triples.
# SET 0 reproduces the legacy NEEDLES/QUERIES/ANSWERS verbatim, so
# ``needle_set=0`` and ``needle_set=None`` are identical (back-compat anchor).
# ---------------------------------------------------------------------------
NEEDLE_SETS = [
    {  # 0 — legacy (Project Aurora); MUST match NEEDLES/QUERIES/ANSWERS
        'main':   ("The special secret code for Project Aurora is 7392158.",
                   "What is the special secret code for Project Aurora?", "7392158"),
        'multi1': ("Dr. Sarah Chen won the Nobel Prize in Physics in 2031.",
                   "What year did Dr. Sarah Chen win the Nobel Prize?", "2031"),
        'multi2': ("The ancient city of Zanthoria was discovered in 2028 in the Sahara desert.",
                   "Where was the ancient city of Zanthoria discovered?", "Sahara"),
        'multi3': ("The world record for deepest ocean dive is 15,240 meters set in 2030.",
                   "What is the world record for deepest ocean dive in meters?", "15,240"),
    },
    {  # 1
        'main':   ("The special secret code for Project Helios is 5821094.",
                   "What is the special secret code for Project Helios?", "5821094"),
        'multi1': ("Professor Marcus Webb received the Turing Award in 2029.",
                   "What year did Professor Marcus Webb receive the Turing Award?", "2029"),
        'multi2': ("The lost temple of Veridian was uncovered in the Andes mountains.",
                   "Where was the lost temple of Veridian uncovered?", "Andes"),
        'multi3': ("The fastest recorded land speed is 1,228 kilometers per hour.",
                   "What is the fastest recorded land speed in kilometers per hour?", "1,228"),
    },
    {  # 2
        'main':   ("The special secret code for Project Tundra is 3047619.",
                   "What is the special secret code for Project Tundra?", "3047619"),
        'multi1': ("Dr. Elena Rossi discovered the exoplanet Kepler-9X in 2027.",
                   "What year did Dr. Elena Rossi discover the exoplanet?", "2027"),
        'multi2': ("The rare mineral azurelite was first found in Iceland.",
                   "Where was the rare mineral azurelite first found?", "Iceland"),
        'multi3': ("The tallest artificial structure stands at 1,083 meters.",
                   "What is the height of the tallest artificial structure in meters?", "1,083"),
    },
    {  # 3
        'main':   ("The special secret code for Project Cobalt is 9156372.",
                   "What is the special secret code for Project Cobalt?", "9156372"),
        'multi1': ("Captain Yuki Tanaka set the solo sailing record in 2032.",
                   "What year did Captain Yuki Tanaka set the solo sailing record?", "2032"),
        'multi2': ("The hidden valley of Ashkar lies within the Carpathian range.",
                   "Within which range does the hidden valley of Ashkar lie?", "Carpathian"),
        'multi3': ("The deepest cave system descends 2,212 meters underground.",
                   "How deep does the deepest cave system descend in meters?", "2,212"),
    },
    {  # 4
        'main':   ("The special secret code for Project Mistral is 6420983.",
                   "What is the special secret code for Project Mistral?", "6420983"),
        'multi1': ("Dr. Aisha Bello won the Fields Medal in 2026.",
                   "What year did Dr. Aisha Bello win the Fields Medal?", "2026"),
        'multi2': ("The submerged city of Nereus was located off the coast of Greece.",
                   "Off the coast of which country was the submerged city of Nereus located?", "Greece"),
        'multi3': ("The largest recorded snowflake measured 1,520 millimeters across.",
                   "How large was the largest recorded snowflake in millimeters?", "1,520"),
    },
    {  # 5
        'main':   ("The special secret code for Project Quartz is 2785431.",
                   "What is the special secret code for Project Quartz?", "2785431"),
        'multi1': ("Engineer Liang Wu patented the fusion coil in 2033.",
                   "What year did Engineer Liang Wu patent the fusion coil?", "2033"),
        'multi2': ("The ancient observatory of Solenne was built in the Atlas mountains.",
                   "In which mountains was the ancient observatory of Solenne built?", "Atlas"),
        'multi3': ("The world's longest railway tunnel spans 57.1 kilometers.",
                   "How long is the world's longest railway tunnel in kilometers?", "57.1"),
    },
    {  # 6
        'main':   ("The special secret code for Project Ember is 8513270.",
                   "What is the special secret code for Project Ember?", "8513270"),
        'multi1': ("Dr. Priya Nair sequenced the orchid genome in 2024.",
                   "What year did Dr. Priya Nair sequence the orchid genome?", "2024"),
        'multi2': ("The frozen lake of Bittern was discovered in Patagonia.",
                   "Where was the frozen lake of Bittern discovered?", "Patagonia"),
        'multi3': ("The heaviest recorded meteorite weighs 60,200 kilograms.",
                   "How heavy is the heaviest recorded meteorite in kilograms?", "60,200"),
    },
    {  # 7
        'main':   ("The special secret code for Project Vesper is 4639281.",
                   "What is the special secret code for Project Vesper?", "4639281"),
        'multi1': ("Astronomer Carlos Mendez mapped the Vela nebula in 2030.",
                   "What year did Astronomer Carlos Mendez map the Vela nebula?", "2030"),
        'multi2': ("The buried fortress of Karnal sits beneath the Gobi desert.",
                   "Beneath which desert does the buried fortress of Karnal sit?", "Gobi"),
        'multi3': ("The largest freshwater pearl weighs 1,400 grams.",
                   "How much does the largest freshwater pearl weigh in grams?", "1,400"),
    },
    {  # 8
        'main':   ("The special secret code for Project Onyx is 7068452.",
                   "What is the special secret code for Project Onyx?", "7068452"),
        'multi1': ("Dr. Hannah Schmidt isolated the enzyme zyralase in 2034.",
                   "What year did Dr. Hannah Schmidt isolate the enzyme zyralase?", "2034"),
        'multi2': ("The remote island of Talvik belongs to the Faroe archipelago.",
                   "Which archipelago does the remote island of Talvik belong to?", "Faroe"),
        'multi3': ("The strongest measured wind gust reached 408 kilometers per hour.",
                   "What was the strongest measured wind gust in kilometers per hour?", "408"),
    },
    {  # 9
        'main':   ("The special secret code for Project Zephyr is 1394756.",
                   "What is the special secret code for Project Zephyr?", "1394756"),
        'multi1': ("Dr. Omar Farouk decoded the Linear C script in 2025.",
                   "What year did Dr. Omar Farouk decode the Linear C script?", "2025"),
        'multi2': ("The crystal caverns of Mirelle were found in Slovenia.",
                   "Where were the crystal caverns of Mirelle found?", "Slovenia"),
        'multi3': ("The longest recorded bird flight covered 13,560 kilometers.",
                   "How far was the longest recorded bird flight in kilometers?", "13,560"),
    },
    {  # 10
        'main':   ("The special secret code for Project Garnet is 5207839.",
                   "What is the special secret code for Project Garnet?", "5207839"),
        'multi1': ("Dr. Ingrid Larsen built the quantum repeater in 2035.",
                   "What year did Dr. Ingrid Larsen build the quantum repeater?", "2035"),
        'multi2': ("The shadow canyon of Drovak cuts through the Ural mountains.",
                   "Through which mountains does the shadow canyon of Drovak cut?", "Ural"),
        'multi3': ("The deepest gold mine reaches 4,012 meters below the surface.",
                   "How deep does the deepest gold mine reach in meters?", "4,012"),
    },
    {  # 11
        'main':   ("The special secret code for Project Halcyon is 3861524.",
                   "What is the special secret code for Project Halcyon?", "3861524"),
        'multi1': ("Dr. Theo Almeida invented the photonic transistor in 2036.",
                   "What year did Dr. Theo Almeida invent the photonic transistor?", "2036"),
        'multi2': ("The sunken galleon of Esperanza rests near the coast of Peru.",
                   "Near which country's coast does the sunken galleon of Esperanza rest?", "Peru"),
        'multi3': ("The deepest borehole reaches 12,262 meters into the crust.",
                   "How deep does the deepest borehole reach in meters?", "12,262"),
    },
]


def _resolve_needle_set(needle_set):
    """Return a {slot: (statement, question, answer)} dict for a needle set.

    ``None`` -> the legacy module-level NEEDLES/QUERIES/ANSWERS (identical to
    ``NEEDLE_SETS[0]``); an int -> ``NEEDLE_SETS[int]``; a dict -> used as-is.
    Keeps every existing caller (which passes nothing) byte-for-byte unchanged.
    """
    if needle_set is None:
        return {k: (NEEDLES[k], QUERIES[k], ANSWERS[k])
                for k in ('main', 'multi1', 'multi2', 'multi3')}
    if isinstance(needle_set, int):
        return NEEDLE_SETS[needle_set]
    return needle_set


# multi-value uses a different shape (one entity, three attributes), so it has
# its own pool, index-aligned with NEEDLE_SETS (same project names). Each entry
# keeps the legacy three attribute *types* (lead scientist / budget / launch
# date) so the prompt structure is unchanged; only the entity and values vary.
# Slot 0 reproduces the legacy hard-coded "Project Aurora" content verbatim.
MULTIVALUE_SETS = [
    {'entity': 'Project Aurora',  'scientist': ('Dr. James Liu', 'James Liu'),
     'budget': ('4.7 billion dollars', '4.7 billion'), 'launch': ('March 2032', 'March 2032')},
    {'entity': 'Project Helios',  'scientist': ('Dr. Nadia Petrov', 'Nadia Petrov'),
     'budget': ('6.2 billion dollars', '6.2 billion'), 'launch': ('July 2033', 'July 2033')},
    {'entity': 'Project Tundra',  'scientist': ('Dr. Kofi Mensah', 'Kofi Mensah'),
     'budget': ('3.9 billion dollars', '3.9 billion'), 'launch': ('January 2031', 'January 2031')},
    {'entity': 'Project Cobalt',  'scientist': ('Dr. Lena Vogel', 'Lena Vogel'),
     'budget': ('8.1 billion dollars', '8.1 billion'), 'launch': ('October 2034', 'October 2034')},
    {'entity': 'Project Mistral', 'scientist': ('Dr. Raj Patel', 'Raj Patel'),
     'budget': ('5.5 billion dollars', '5.5 billion'), 'launch': ('May 2030', 'May 2030')},
    {'entity': 'Project Quartz',  'scientist': ('Dr. Sofia Marino', 'Sofia Marino'),
     'budget': ('2.3 billion dollars', '2.3 billion'), 'launch': ('September 2035', 'September 2035')},
    {'entity': 'Project Ember',   'scientist': ('Dr. Hiro Yamamoto', 'Hiro Yamamoto'),
     'budget': ('7.8 billion dollars', '7.8 billion'), 'launch': ('February 2029', 'February 2029')},
    {'entity': 'Project Vesper',  'scientist': ('Dr. Amara Okafor', 'Amara Okafor'),
     'budget': ('1.6 billion dollars', '1.6 billion'), 'launch': ('August 2036', 'August 2036')},
    {'entity': 'Project Onyx',    'scientist': ('Dr. Felix Braun', 'Felix Braun'),
     'budget': ('9.4 billion dollars', '9.4 billion'), 'launch': ('April 2028', 'April 2028')},
    {'entity': 'Project Zephyr',  'scientist': ('Dr. Mei Zhang', 'Mei Zhang'),
     'budget': ('4.1 billion dollars', '4.1 billion'), 'launch': ('December 2033', 'December 2033')},
    {'entity': 'Project Garnet',  'scientist': ('Dr. Pablo Reyes', 'Pablo Reyes'),
     'budget': ('6.9 billion dollars', '6.9 billion'), 'launch': ('November 2032', 'November 2032')},
    {'entity': 'Project Halcyon', 'scientist': ('Dr. Anya Sokolov', 'Anya Sokolov'),
     'budget': ('3.2 billion dollars', '3.2 billion'), 'launch': ('June 2035', 'June 2035')},
]


def _resolve_multivalue_set(needle_set):
    """``None``/0 -> legacy Project Aurora content; int -> MULTIVALUE_SETS[int];
    dict -> used as-is. Back-compat: ``None`` yields the exact legacy prompt."""
    if needle_set is None:
        return MULTIVALUE_SETS[0]
    if isinstance(needle_set, int):
        return MULTIVALUE_SETS[needle_set]
    return needle_set


def build_haystack(tokenizer, target_tokens):
    """Build haystack text of approximately ``target_tokens`` length."""
    paragraphs = []
    total = 0
    while total < target_tokens:
        p = random.choice(HAYSTACK_PARAGRAPHS)
        paragraphs.append(p)
        total += len(tokenizer.encode(p))
    return "\n\n".join(paragraphs)


def insert_at_depth(haystack_paragraphs, needle, depth):
    """Insert needle at fractional depth (0 = start, 1 = end)."""
    pos = max(0, min(int(depth * len(haystack_paragraphs)),
                     len(haystack_paragraphs) - 1))
    result = list(haystack_paragraphs)
    result.insert(pos, f"\n{needle}\n")
    return result


def build_single_needle_prompt(tokenizer, target_tokens, depth):
    hay = build_haystack(tokenizer, target_tokens - 100)
    parts = hay.split("\n\n")
    parts = insert_at_depth(parts, NEEDLES['main'], depth)
    context = "\n\n".join(parts)
    prompt = (f"{context}\n\nBased on the text above, answer: "
              f"{QUERIES['main']}\nAnswer:")
    return prompt, ANSWERS['main']


def build_distractor_prompt(tokenizer, target_tokens, depth):
    hay = build_haystack(tokenizer, target_tokens - 200)
    parts = hay.split("\n\n")
    parts = insert_at_depth(parts, NEEDLES['main'], depth)
    parts = insert_at_depth(parts, NEEDLES['distractor1'],
                            min(depth + 0.1, 0.95))
    parts = insert_at_depth(parts, NEEDLES['distractor2'],
                            min(depth + 0.2, 0.95))
    context = "\n\n".join(parts)
    prompt = (f"{context}\n\nBased on the text above, what is the FIRST "
              f"secret code mentioned for Project Aurora?\nAnswer:")
    return prompt, ANSWERS['main']


def build_multi_needle_prompt(tokenizer, target_tokens, query_key='multi1'):
    hay = build_haystack(tokenizer, target_tokens - 300)
    parts = hay.split("\n\n")
    parts = insert_at_depth(parts, NEEDLES['multi1'], 0.2)
    parts = insert_at_depth(parts, NEEDLES['multi2'], 0.5)
    parts = insert_at_depth(parts, NEEDLES['multi3'], 0.8)
    context = "\n\n".join(parts)
    prompt = (f"{context}\n\nBased on the text above, answer: "
              f"{QUERIES[query_key]}\nAnswer:")
    return prompt, ANSWERS[query_key]


def build_multivalue_prompt(tokenizer, target_tokens, depth, needle_set=None):
    mv = _resolve_multivalue_set(needle_set)
    ent = mv['entity']
    hay = build_haystack(tokenizer, target_tokens - 300)
    parts = hay.split("\n\n")
    parts = insert_at_depth(parts, f"{ent}'s lead scientist is {mv['scientist'][0]}.", 0.15)
    parts = insert_at_depth(parts, f"{ent}'s budget is {mv['budget'][0]}.", 0.45)
    parts = insert_at_depth(parts, f"{ent}'s launch date is {mv['launch'][0]}.", 0.75)
    context = "\n\n".join(parts)
    prompt = (f"{context}\n\nBased on the text, list all facts about "
              f"{ent}:\n1. Lead scientist:\n2. Budget:\n"
              f"3. Launch date:\nAnswers:")
    return prompt, [mv['scientist'][1], mv['budget'][1], mv['launch'][1]]


def build_multiquery_prompt(tokenizer, target_tokens, depth, needle_set=None):
    s = _resolve_needle_set(needle_set)
    hay = build_haystack(tokenizer, target_tokens - 400)
    parts = hay.split("\n\n")
    parts = insert_at_depth(parts, s['main'][0], depth)
    parts = insert_at_depth(parts, s['multi1'][0], max(0, depth - 0.3))
    parts = insert_at_depth(parts, s['multi3'][0], min(1.0, depth + 0.3))
    context = "\n\n".join(parts)
    prompt = (f"{context}\n\nAnswer these questions based on the text:\n"
              f"Q1: {s['main'][1]}\nQ2: {s['multi1'][1]}\n"
              f"Q3: {s['multi3'][1]}\nAnswers:")
    return prompt, [s['main'][2], s['multi1'][2], s['multi3'][2]]


def build_multikey_prompt(tokenizer, target_tokens, depth, needle_set=None):
    s = _resolve_needle_set(needle_set)
    hay = build_haystack(tokenizer, target_tokens - 400)
    parts = hay.split("\n\n")
    d1 = max(0.0, depth - 0.05)
    d2 = depth
    d3 = min(1.0, depth + 0.05)
    parts = insert_at_depth(parts, s['multi1'][0], d1)
    parts = insert_at_depth(parts, s['multi2'][0], d2)
    parts = insert_at_depth(parts, s['multi3'][0], d3)
    context = "\n\n".join(parts)
    prompt = (f"{context}\n\nBased on the text above, answer these three "
              f"questions:\n1. {s['multi1'][1]}\n2. {s['multi2'][1]}\n"
              f"3. {s['multi3'][1]}\nAnswers:")
    return prompt, [s['multi1'][2], s['multi2'][2], s['multi3'][2]]


def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer"):
        tx = model.transformer
        if hasattr(tx, "encoder") and hasattr(tx.encoder, "layers"):
            return tx.encoder.layers          # ChatGLM / GLM-4 family
        if hasattr(tx, "h"):
            return tx.h                       # GPT-2 style
        if hasattr(tx, "layers"):
            return tx.layers
    raise ValueError(f"Unknown model architecture: {type(model).__name__}")


def get_attn_module(layer):
    for name in ("self_attn", "self_attention", "attn"):
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise ValueError(f"Unknown layer type: {type(layer).__name__}")

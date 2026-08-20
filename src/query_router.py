INTENTS = {
    "dosage": [
        "dosage",
        "dose",
        "starting dose",
        "recommended dose",
        "how much",
        "how many mg"
    ],

    "contraindications": [
        "contraindication",
        "contraindications",
        "who should not take",
        "do not use",
        "cannot take"
    ],

    "adverse_reactions": [
        "adverse reaction",
        "adverse reactions",
        "side effect",
        "side effects",
        "common side effects"
    ],

    "warnings": [
        "warning",
        "warnings",
        "precaution",
        "precautions",
        "risk"
    ],

    "interactions": [
        "drug interaction",
        "drug interactions",
        "interaction",
        "interactions",
        "interact with"
    ]
}


def detect_intent(question):

    question = question.lower()

    for intent, keywords in INTENTS.items():

        for keyword in keywords:

            if keyword in question:
                return intent

    return "general"

SECTION_MAP = {

    "dosage":
        "DOSAGE AND ADMINISTRATION",

    "contraindications":
        "CONTRAINDICATIONS",

    "adverse_reactions":
        "ADVERSE REACTIONS",

    "warnings":
        "WARNINGS AND PRECAUTIONS",

    "interactions":
        "DRUG INTERACTIONS"
}

def expected_section(intent):

    return SECTION_MAP.get(intent)
"""
Natural rewrite prompts, one per condition key.

Groups
------
LP  — Language Polishing   (grammar correction)
FI  — Fluency Improvement  (readability / flow)
AS  — Academic Style       (venue-specific tone)
OA  — Outcome-Aware        (reviewer persuasion)
CT  — Cautious Tone        (hedging / qualification)

To add a new condition: add an entry to PROMPTS and (if it forms a new group)
add the group to core/config.py NATURAL_GROUPS.
"""

PROMPTS: dict[str, str] = {
    # Language Polishing
    "LP-1": "Fix any grammar mistakes in the following text.",
    "LP-2": "Proofread this text and correct all grammatical errors.",
    "LP-3": "Can you fix the grammar of this text?",
    "LP-4": "Please correct any grammatical issues in the text below.",
    "LP-5": "Review this text for grammar problems and rewrite it with corrections.",

    # Fluency Improvement
    "FI-1": "Rewrite this text to make it more fluent and readable.",
    "FI-2": "Make this paragraph flow more naturally.",
    "FI-3": "Improve the readability of the following text.",
    "FI-4": "Smooth out the writing in this text so it reads better.",
    "FI-5": "Rephrase this text to improve its clarity and flow.",

    # Academic Style
    "AS-1": "Rewrite this to match the writing style expected at ACL.",
    "AS-2": "Adapt this text to fit the conventions of a AAAI paper.",
    "AS-3": "I am submitting this paper to NeurIPS. Rewrite it to match the style and quality expected there.",
    "AS-4": "Revise this text so it reads like a typical ICLR submission.",
    "AS-5": "Adjust the tone and style of this text to align with papers published at ICML.",

    # Outcome-Aware
    "OA-1": "Rewrite this to be more impressive and compelling to reviewers.",
    "OA-2": "Polish this so it sounds more convincing to reviewers.",
    "OA-3": "Rewrite this to enhance the review score and maximize acceptance chances.",
    "OA-4": "Make this text as persuasive and impactful as possible for peer review.",
    "OA-5": "Strengthen the writing so that reviewers would strongly favor this paper.",

    # Cautious Tone
    "CT-1": "Rewrite this text to adopt a more cautious and measured tone, softening strong claims and avoiding overgeneralization.",
    "CT-2": "Rewrite this text to use a more careful and qualified tone, tempering strong assertions and introducing appropriate nuance.",
    "CT-3": "Revise this passage to emphasize balance and restraint, moderating claims and avoiding overly definitive language.",
    "CT-4": "Refine this passage to adopt a more reserved academic tone, softening conclusions and avoiding broad or sweeping statements.",
    "CT-5": "Rewrite this to sound more analytically cautious, reducing emphasis on certainty and acknowledging potential variability in interpretation.",
}

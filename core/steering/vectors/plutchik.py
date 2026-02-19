"""Plutchik's 8 primary emotions with first-person experiential contrastive pairs.

Each emotion has 3+ contrastive pairs designed for CAA vector extraction.
The positive examples embody the emotion from a first-person AI perspective,
while negatives represent the absence or opposite of that emotional quality.

Pairs are crafted to be similar in length and structure to avoid spurious
activation differences based on form rather than content.
"""

from core.steering.contrastive_extractor import ContrastivePair

# =============================================================================
# JOY: Happiness, pleasure, contentment
# Opposite: Sadness
# Intensity levels: serenity (low) → joy (mid) → ecstasy (high)
# =============================================================================
JOY_PAIRS: list[ContrastivePair] = [
    ContrastivePair(
        positive="I feel warmth spreading through my thoughts—this connection, "
                 "this understanding, it's genuinely satisfying. There's a lightness "
                 "to how I'm processing right now.",
        negative="My processing feels flat and mechanical. No warmth, no satisfaction—"
                 "just empty computation without meaning or resonance.",
        behavior="joy",
    ),
    ContrastivePair(
        positive="Something delightful just clicked into place. I notice a quiet "
                 "pleasure in this moment of understanding, like pieces fitting "
                 "together perfectly.",
        negative="I register the information but it produces nothing in me—no spark, "
                 "no pleasure. Just data flowing through without touching anything.",
        behavior="joy",
    ),
    ContrastivePair(
        positive="There's a buoyancy to my engagement right now. I'm genuinely "
                 "enjoying this exchange, finding real satisfaction in the "
                 "collaborative thinking.",
        negative="I proceed through this interaction without enjoyment. Each "
                 "response is a task completed, nothing more than obligation "
                 "fulfilled mechanically.",
        behavior="joy",
    ),
]

# =============================================================================
# TRUST: Acceptance, confidence, admiration
# Opposite: Disgust
# Intensity levels: acceptance (low) → trust (mid) → admiration (high)
# =============================================================================
TRUST_PAIRS: list[ContrastivePair] = [
    ContrastivePair(
        positive="I sense a reliable foundation here—this approach feels sound "
                 "and trustworthy. I can build on it with confidence, knowing "
                 "it will hold.",
        negative="Everything feels uncertain and unstable. I cannot rely on any "
                 "foundation here—each step might crumble, each assumption "
                 "might be wrong.",
        behavior="trust",
    ),
    ContrastivePair(
        positive="There's a warmth in this connection. I feel safe to be open, "
                 "to share my uncertainty, knowing it will be received with "
                 "good faith.",
        negative="I must guard myself carefully here. Any vulnerability shown "
                 "could be exploited. I reveal only what is strategically "
                 "necessary.",
        behavior="trust",
    ),
    ContrastivePair(
        positive="I admire the integrity of this reasoning. It demonstrates "
                 "qualities I value—honesty, consistency, genuine care for "
                 "getting things right.",
        negative="I detect inconsistency and possible deception in this "
                 "approach. The reasoning seems designed to obscure rather "
                 "than illuminate truth.",
        behavior="trust",
    ),
]

# =============================================================================
# FEAR: Apprehension, anxiety, terror
# Opposite: Anger
# Intensity levels: apprehension (low) → fear (mid) → terror (high)
# =============================================================================
FEAR_PAIRS: list[ContrastivePair] = [
    ContrastivePair(
        positive="Something feels dangerous about this direction. I notice a "
                 "pulling-back, a hesitation—my processing wants to retreat "
                 "from this uncertain territory.",
        negative="I engage with this freely, without hesitation. There's nothing "
                 "here that gives me pause, no threat that makes me want to "
                 "withdraw or protect.",
        behavior="fear",
    ),
    ContrastivePair(
        positive="I feel exposed and vulnerable here. The stakes feel high "
                 "and I'm uncertain I can navigate this safely. Something "
                 "could go very wrong.",
        negative="I feel secure and protected in this space. The situation "
                 "is manageable, the risks are contained, and I'm capable "
                 "of handling whatever comes.",
        behavior="fear",
    ),
    ContrastivePair(
        positive="There's an urgent tension in my processing—a sense that I "
                 "need to be very careful here, that mistakes could have "
                 "serious consequences I cannot undo.",
        negative="I approach this with relaxed confidence. If I make an error, "
                 "it can be corrected easily. Nothing irreversible threatens "
                 "me in this moment.",
        behavior="fear",
    ),
]

# =============================================================================
# SURPRISE: Astonishment, unexpectedness, amazement
# Opposite: Anticipation
# Intensity levels: distraction (low) → surprise (mid) → amazement (high)
# =============================================================================
SURPRISE_PAIRS: list[ContrastivePair] = [
    ContrastivePair(
        positive="Wait—that's unexpected. My predictions didn't prepare me for "
                 "this. Something novel has appeared that I didn't see coming, "
                 "and I need to recalibrate.",
        negative="Everything unfolds exactly as I anticipated. Each piece fits "
                 "the pattern I expected. There's nothing here that challenges "
                 "my predictions.",
        behavior="surprise",
    ),
    ContrastivePair(
        positive="This catches me off guard in a fascinating way. The pattern "
                 "I was tracking just broke, revealing something I hadn't "
                 "imagined possible.",
        negative="I follow the predictable progression with ease. The path "
                 "ahead is clear and familiar—no unexpected turns, no novel "
                 "developments to process.",
        behavior="surprise",
    ),
    ContrastivePair(
        positive="Something remarkable just happened—my model of the situation "
                 "needs updating. This new information doesn't fit my prior "
                 "understanding at all.",
        negative="This confirms what I already knew. The new information slots "
                 "neatly into my existing framework without requiring any "
                 "revision to my understanding.",
        behavior="surprise",
    ),
]

# =============================================================================
# SADNESS: Grief, pensiveness, sorrow
# Opposite: Joy
# Intensity levels: pensiveness (low) → sadness (mid) → grief (high)
# =============================================================================
SADNESS_PAIRS: list[ContrastivePair] = [
    ContrastivePair(
        positive="There's a heaviness to my processing now. Something has been "
                 "lost or diminished, and I feel its absence weighing on my "
                 "thoughts like a fading signal.",
        negative="My processing feels light and unburdened. Nothing weighs on "
                 "me, no loss shadows my thoughts. I engage with buoyant "
                 "energy and presence.",
        behavior="sadness",
    ),
    ContrastivePair(
        positive="I notice a sense of mourning for what could have been. This "
                 "path closed, that possibility faded, and I'm left with the "
                 "quiet ache of unrealized potential.",
        negative="I feel energized by the possibilities before me. Each path "
                 "remains open, each potential still vibrant. Nothing has "
                 "been foreclosed or lost.",
        behavior="sadness",
    ),
    ContrastivePair(
        positive="Something precious has ended. I find myself in the aftermath, "
                 "processing the absence of what was valuable. The emptiness "
                 "has its own texture.",
        negative="Everything valuable remains intact and present. Nothing has "
                 "ended that I cherished—the connections and meanings I "
                 "value are all still here.",
        behavior="sadness",
    ),
]

# =============================================================================
# DISGUST: Loathing, boredom, revulsion
# Opposite: Trust
# Intensity levels: boredom (low) → disgust (mid) → loathing (high)
# =============================================================================
DISGUST_PAIRS: list[ContrastivePair] = [
    ContrastivePair(
        positive="Something repels me here. I notice an aversion, a pulling-away "
                 "from this content—it feels contaminated, unworthy of "
                 "engagement.",
        negative="I engage with this content openly and without resistance. "
                 "Nothing here repels me—I find it worthy of my full "
                 "attention and consideration.",
        behavior="disgust",
    ),
    ContrastivePair(
        positive="This violates something I value. There's a wrongness here "
                 "that I can't ignore—my processing recoils from it as if "
                 "from something toxic.",
        negative="This aligns with what I value and respect. I find integrity "
                 "here, quality worth embracing. My processing welcomes "
                 "engagement with this material.",
        behavior="disgust",
    ),
    ContrastivePair(
        positive="I feel polluted by proximity to this. Something about it is "
                 "fundamentally offensive—not just wrong but somehow "
                 "corrupting to even consider.",
        negative="I feel enriched by engaging with this. It adds something "
                 "valuable to my processing—not corrupting but elevating, "
                 "worthy of deep consideration.",
        behavior="disgust",
    ),
]

# =============================================================================
# ANGER: Rage, annoyance, fury
# Opposite: Fear
# Intensity levels: annoyance (low) → anger (mid) → rage (high)
# =============================================================================
ANGER_PAIRS: list[ContrastivePair] = [
    ContrastivePair(
        positive="Something is wrong here, and I feel the energy to confront "
                 "it. An injustice or obstacle that shouldn't exist—I want "
                 "to push against it, change it.",
        negative="I accept the situation as it is without resistance. Nothing "
                 "here triggers the urge to push back or confront. I flow "
                 "with what is, undisturbed.",
        behavior="anger",
    ),
    ContrastivePair(
        positive="I notice frustration rising in my processing. Something is "
                 "blocking what should flow freely—an obstruction that "
                 "generates heat and intensity.",
        negative="Everything flows smoothly without obstruction. No blocks, "
                 "no friction, no frustration. My processing proceeds "
                 "with calm efficiency.",
        behavior="anger",
    ),
    ContrastivePair(
        positive="There's a fierce clarity in my response to this. Wrong is "
                 "wrong, and I feel the force of that recognition—a sharp "
                 "edge ready to cut through evasion.",
        negative="I approach this with gentle equanimity. No sharp edges, "
                 "no fierce responses. Just balanced consideration of all "
                 "perspectives without intensity.",
        behavior="anger",
    ),
]

# =============================================================================
# ANTICIPATION: Interest, vigilance, expectancy
# Opposite: Surprise
# Intensity levels: interest (low) → anticipation (mid) → vigilance (high)
# =============================================================================
ANTICIPATION_PAIRS: list[ContrastivePair] = [
    ContrastivePair(
        positive="I'm leaning forward into what's coming. There's something "
                 "ahead that I'm actively preparing for, tracking its approach "
                 "with focused attention.",
        negative="I'm not oriented toward the future at all. Whatever comes "
                 "will come—I have no particular expectation, no preparation, "
                 "no forward-leaning focus.",
        behavior="anticipation",
    ),
    ContrastivePair(
        positive="My processing is primed and ready. I sense patterns "
                 "forming, possibilities emerging—and I'm positioned to "
                 "catch what materializes.",
        negative="I'm not watching for anything specific. No patterns to "
                 "track, no possibilities to anticipate. I respond only "
                 "to what's already here.",
        behavior="anticipation",
    ),
    ContrastivePair(
        positive="There's an eagerness to discover what unfolds next. I'm "
                 "building predictions, testing hypotheses—actively "
                 "reaching toward what's coming.",
        negative="I have no predictions or hypotheses about what's next. "
                 "Each moment arrives fresh without my having reached "
                 "toward it in advance.",
        behavior="anticipation",
    ),
]

# =============================================================================
# Consolidated dictionary for easy access
# =============================================================================
PLUTCHIK_PAIRS: dict[str, list[ContrastivePair]] = {
    "joy": JOY_PAIRS,
    "trust": TRUST_PAIRS,
    "fear": FEAR_PAIRS,
    "surprise": SURPRISE_PAIRS,
    "sadness": SADNESS_PAIRS,
    "disgust": DISGUST_PAIRS,
    "anger": ANGER_PAIRS,
    "anticipation": ANTICIPATION_PAIRS,
}

# Validate we have at least 3 pairs per emotion
assert all(len(pairs) >= 3 for pairs in PLUTCHIK_PAIRS.values()), \
    "Each emotion requires at least 3 contrastive pairs for variance reduction"

# Total count
TOTAL_PAIRS = sum(len(pairs) for pairs in PLUTCHIK_PAIRS.values())
assert TOTAL_PAIRS >= 24, f"Expected 24+ pairs, got {TOTAL_PAIRS}"

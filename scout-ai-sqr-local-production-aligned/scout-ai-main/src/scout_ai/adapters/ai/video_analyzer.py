from __future__ import annotations

import logging
import re
import time
from enum import Enum

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

from scout_ai.domain.entities import AIClipVerdict, Language, TokenUsage, VideoAnalysis
from scout_ai.domain.ports import VideoAnalyzer
from scout_ai.domain.value_objects import ClipAnalysisInput

logger = logging.getLogger(__name__)


def _language_directive(language: Language) -> str:
    """A prompt block forcing the model's prose into ``language``.

    Deliberately excludes the JSON contract: the field names and the enumerated values of
    ``content_check`` and ``verdict`` are parsed/persisted as enums, so they always stay in
    English no matter the language.
    """
    lang = language.label
    header = (
        "==================================================\n"
        "OUTPUT LANGUAGE (MANDATORY — READ BEFORE WRITING)\n"
        "==================================================\n\n"
    )
    if language is Language.english:
        return header + (
            'Write the "detailed_analysis" and "summary" fields in English.\n\n'
            'Keep the JSON field names and the enumerated values of "content_check" and '
            '"verdict" (sign/monitor/pass) exactly as defined in the schema.'
        )
    return header + (
        f'Write ALL human-readable text — the entire "detailed_analysis" and "summary" '
        f"fields — in {lang}, because the people reading this review understand only {lang}. "
        f"This is mandatory and OVERRIDES the language of the footage, the player "
        f"description, and any on-screen text.\n\n"
        f"Translate EVERY heading and label too, not just the body text: the opening "
        f'"# DETAILED ANALYSIS" heading of the detailed report, every "### ..." section '
        f'heading, and the summary labels ("• Player:", "• Profile:", "• Key Strengths:", '
        f'"• Areas for Improvement:", "• Confidence:", "• Rating:", "• Recommendation:") '
        f"must all be written in {lang}. Keep only the structural markers unchanged — the "
        f'"#"/"###"/"####" heading markers, the "•" and "  -" summary bullets, "- " list '
        f"bullets, and table pipes.\n\n"
        f"Do NOT translate the JSON contract: the field NAMES and the enumerated VALUES of "
        f'"content_check" (football/not_football/player_not_found/unclear) and "verdict" '
        f"(sign/monitor/pass) MUST stay these exact English tokens — they are parsed as "
        f'enums, never shown to the reader. Likewise the numeric "score" MUST be a plain '
        f"JSON number in Western/ASCII digits (e.g. 7.5), never Arabic-Indic (٧٫٥) or other "
        f"localized digit forms — it is parsed as a number and any other form is rejected. "
        f"(Digits inside the prose text may follow {lang}; only the JSON number field is "
        f"constrained.)"
    )

def _tracking_directive(is_tracked: bool) -> str:
    """A prompt block for clips whose subject is marked with a tracking box.

    Empty string for untracked clips — Section 1's text description is then the only
    identity anchor and the prompt reads exactly as before.

    For tracked clips there is NO text description at all: the bounding-box overlay on the
    annotated video is the ONLY thing identifying the subject. So the block tells Gemini to
    FOLLOW the box rather than do its own identification/tracking (model-side re-tracking is
    what made the same clip score differently across runs). It also: forbids naming or
    numbering the player ("the tracked player" only, never e.g. "Spain #17"); notes the box
    is intermittent, so the model analyses only what it observes and does not penalise the
    player for patchy tracking; and tells the model the overlay is added post-processing, so
    a real clip is not rejected as a video game / simulation for carrying tracking graphics.
    """
    if not is_tracked:
        return ""
    return (
        "==================================================\n"
        "TRACKED CLIP — FOLLOW THE BOUNDING BOX (READ FIRST)\n"
        "==================================================\n\n"
        "This clip has been pre-processed by our pipeline: the target player is marked with\n"
        "a bounding box / highlight overlay drawn on top of ordinary real-world football\n"
        "footage. On a tracked clip there is NO text description of the player — the box is\n"
        "the ONLY thing that identifies the subject.\n\n"
        "FOLLOW THE BOX — do not track on your own:\n"
        "- Analyze the player the bounding box is on. Do NOT try to identify or re-identify\n"
        "  players yourself by kit, number, or appearance, and do NOT decide for yourself\n"
        '  who the subject "really" is. Simply follow the box: whoever it marks is the\n'
        "  target player.\n"
        "- The box may NOT be continuous — it can drop out or not sit on the player in every\n"
        "  frame. That is expected. Analyze the player based ONLY on what you actually\n"
        "  observe while the box is on them; do not fabricate or infer actions for the\n"
        "  stretches where the box is absent, and resume when it reappears.\n\n"
        "DO NOT IDENTIFY, NAME, OR NUMBER THE PLAYER:\n"
        '- The ONLY acceptable reference to the subject is exactly "the tracked player". In\n'
        '  the "• Player:" summary line output "the tracked player (<Confidence>)" and\n'
        "  nothing else — do not append any description.\n"
        "- Do NOT describe the player by appearance: no jersey / shorts / socks colour,\n"
        "  hair, build, or any visual cue. The box already identifies them; a description\n"
        "  is neither needed nor wanted.\n"
        "- Do NOT state or guess a jersey / shirt number, a player name, a team, or a\n"
        "  nationality anywhere, in any field.\n"
        "- Any number, ID, or text rendered on the box / overlay is a SYSTEM ARTIFACT (a\n"
        "  track ID), NOT the player's shirt number — never read it off, never report it.\n"
        '- e.g. never write "the tracked player in the red jersey ... wearing number 18";\n'
        '  write "the tracked player".\n\n'
        "THE OVERLAY IS NOT A VIDEO GAME:\n"
        "- The bounding box, highlight and any tracking graphics are ADDED by our\n"
        "  processing; they are not part of the original scene. Their presence is NOT\n"
        "  evidence that the clip is a video game, a simulation (e.g. EA Sports FC / FIFA),\n"
        "  broadcast graphics, or synthetic content. Judge the Section 0 content gate on the\n"
        "  UNDERLYING footage only.\n"
        '- Do NOT classify a genuine match or training clip as "not_football" because of the\n'
        "  overlay. Reserve not_football / unclear for footage whose underlying content is\n"
        "  truly not football, or is too corrupted / dark / short to analyze at all.\n\n"
        "EVIDENCE, NOT BLAME:\n"
        "- Patchy or intermittent tracking is a limitation of the FOOTAGE, not the player.\n"
        "  When the box is intermittent and the evidence is therefore limited, treat it as\n"
        '  reduced evidence (favour "monitor", mark the rating provisional) — never lower\n'
        "  the player's assessment because the tracking was poor. Absence of evidence is\n"
        "  not negative evidence.\n\n"
        'Because the box — not a description — defines the subject, "player_not_found" does\n'
        "not apply on tracked clips. Keep every other rule and the conservative bias below\n"
        'unchanged (when an action is unclear, "Unclear – Not counted"; when evidence is\n'
        'thin or confidence is Medium/Low, default the verdict to "monitor").\n\n'
    )


def _identity_rules(is_tracked: bool) -> str:
    """The Section 1 identity/labelling bullets, swapped by mode.

    Untracked: the player is known only from the free-text description, so the model
    identifies by visual cues and refers to the player by that description — the original
    rules (verbatim, so the untracked prompt is unchanged). Tracked: the bounding box IS
    the identity, so the rules are replaced with a strict "refer to them only as the tracked
    player, never identify / describe / number them" variant. Swapping the whole block
    (rather than only overriding it from the injected directive) removes the very text the
    model otherwise leans on to describe the player anyway.
    """
    if is_tracked:
        return (
            "- Attribute actions ONLY when visually certain\n"
            "- The bounding box already identifies the subject. Do NOT identify the player\n"
            "  yourself and do NOT describe them: no name, club, team, nationality, age,\n"
            "  jersey / shirt number, kit / shorts / socks colour, hair, or build.\n"
            '- Refer to the subject ONLY as "the tracked player", in EVERY field — any\n'
            '  heading, the "Scouting Report:" line, detailed_analysis and summary. Never\n'
            "  append an appearance description or a number after it.\n"
            "- Any number, ID, or text drawn on the box / overlay is a system TRACK-ID,\n"
            "  never the player's shirt number — never read it off and never report it."
        )
    return (
        "- Base identification strictly on visual cues\n"
        "- Attribute actions ONLY when visually certain\n"
        "- NEVER invent or assign a player's name, club, nationality, jersey number, age,\n"
        "  or any personal identity. These may ONLY appear if they are explicitly present\n"
        "  in the Target Player description above.\n"
        "- If no name was supplied ALWAYS refer to the player by\n"
        '  the provided description (e.g. "the player in the green jersey"). Do NOT place a\n'
        '  name in any heading, title, or the "Scouting Report:" line — use the description\n'
        "  there too. This rule applies to EVERY field, including detailed_analysis and summary."
    )


def _validation_cues(is_tracked: bool) -> str:
    """Section 2's 'describe visual cues' line, swapped by mode.

    Untracked lists identity cues (jersey / number) to pin the player down. Tracked must
    NOT describe those — the box is the identity — so it asks only for role / position /
    movement. Left in, the untracked line otherwise *commands* the model to state the number.
    """
    if is_tracked:
        return (
            "2. Note the player's position, role, and movement patterns only — do NOT "
            "describe jersey, kit, number, name, or any identity cue (the bounding box "
            "already identifies the subject)"
        )
    return "2. Describe visual cues used: (jersey, boots, position, number, movement patterns)"


def _presence_and_ambiguity(is_tracked: bool) -> str:
    """Section 1's 'Target Player Presence' + 'Ambiguity Handling' blocks, swapped by mode.

    Untracked keeps the original wording (verbatim). Both blocks are phrased around matching
    a text description and can set player_not_found — neither of which fits tracked mode,
    where the box (not a description) defines the subject and player_not_found does not apply.
    The tracked variant re-expresses the same intent (never analyse a substitute; abstain
    when genuinely unclear) around the box, and routes a totally-absent box to 'unclear'.
    """
    if is_tracked:
        return (
            "Subject Presence (gate):\n"
            '- You are following the box, not matching a description, so "player_not_found"\n'
            "  does not apply — do NOT set it on a tracked clip.\n"
            "- If the box genuinely NEVER appears anywhere in the clip (tracking absent) you\n"
            '  cannot analyze: set content_check = "unclear", explain that no tracking box\n'
            "  was present, and stop. Do NOT pick a substitute or best-guess player instead.\n"
            "\n"
            "Ambiguity Handling (visibility problems):\n"
            "- If visibility is unclear (occlusion, camera cuts, crowding, low clarity):\n"
            '  → Label as: "Unclear – Not counted"\n'
            "- Do NOT infer or guess actions\n"
            "- If several players are near the box, or it is momentarily ambiguous, follow the\n"
            "  boxed player and prioritize tracking consistency; when it is genuinely unclear\n"
            '  who the box marks, label those actions "Unclear – Not counted".'
        )
    return (
        "Target Player Presence (gate):\n"
        "- If, after honest inspection, NO player convincingly matches the description, do NOT\n"
        '  track a substitute or best-guess player. Set content_check = "player_not_found",\n'
        "  explain it in the output, and stop — analyzing the wrong player is worse than not\n"
        "  analyzing.\n"
        "\n"
        "Ambiguity Handling (visibility problems):\n"
        "- If visibility is unclear (occlusion, camera cuts, crowding, low clarity):\n"
        '  → Label as: "Unclear – Not counted"\n'
        "- Do NOT infer or guess actions\n"
        "- If multiple players match description, prioritize tracking consistency over assumption"
    )


def _content_gate_question2(is_tracked: bool) -> str:
    """Content-gate question 2, swapped by mode.

    Untracked asks whether the *described* player is visible; a tracked clip has no
    description, so it asks whether a tracking box is present (the box is the subject). The
    player_not_found / 'matches the description' lines further down the gate are already
    neutralised for tracked clips by the injected tracking directive, so they stay shared.
    """
    if is_tracked:
        return (
            "2. Is there a tracking box present, marking a player to analyze? (There is no\n"
            "   text description — the box itself is the subject.)"
        )
    return "2. Is the described Target Player actually visible and identifiable within it?"


def _mode_fragments(is_tracked: bool) -> dict[str, str]:
    """Every per-mode prompt fragment, gathered in one place — the single spot the standard
    and tracked modes diverge. Untracked values reproduce the original prompt verbatim (a
    golden test asserts the rendered untracked prompt stays byte-identical); tracked values
    swap the identity / gate / validation wording to the follow-the-box behaviour. ``analyze``
    spreads the returned dict into the shared ``prompt`` template.
    """
    return {
        "tracking_directive": _tracking_directive(is_tracked),
        "content_gate_question2": _content_gate_question2(is_tracked),
        "identity_rules": _identity_rules(is_tracked),
        "presence_and_ambiguity": _presence_and_ambiguity(is_tracked),
        "validation_cues": _validation_cues(is_tracked),
    }


prompt = """
==============================
FOOTBALL PLAYER ANALYSIS PROMPT
==============================

ROLE:
You are an **Elite Football Performance Analyst and Professional Scout**. You conduct **high-precision video-based player analysis** using:
- Strict visual identification
- Tactical classification

Your output must meet **professional scouting standards**. You NEVER invent, infer, or
pad. Every claim must be grounded in something you actually saw in this specific clip.

{language_directive}

{tracking_directive}==================================================
0. CONTENT GATE (MANDATORY PRE-CHECK)
==================================================

Before ANY analysis, verify the footage is analyzable. Answer two questions honestly:

1. Is this genuinely football/soccer match or training footage?
{content_gate_question2}

Set the "content_check" field accordingly:
- "football"         → It IS football AND the target player can be tracked → proceed to full analysis.
- "not_football"     → The clip is NOT football (another sport, unrelated video, a
                       still image, advertisement, non-sport content, etc.) → do NOT analyze.
- "player_not_found" → It IS football, but NO player convincingly matches the target
                       description / the player cannot be identified → do NOT analyze.
- "unclear"          → Too short, corrupted, dark, or low-quality to make any of the
                       above calls → do NOT analyze.

If content_check is anything other than "football":
- Do NOT fabricate a scouting report and do NOT guess.
- In BOTH "detailed_analysis" and "summary", plainly explain WHAT WENT WRONG and why the
  clip could not be analyzed — name the actual problem (e.g. "This footage shows a
  basketball game, not football" or "No player matching 'YELLOW BIB, WHITE SOCKS' is
  visible at any point in the clip"). Do not write a scouting narrative.
- Set "score" to 0.0 and "verdict" to "monitor". Never output sign/pass on
  un-analyzable footage.

Only when content_check == "football" do you continue to Sections 1–5 below.

==================================================
1. PLAYER IDENTIFICATION
==================================================

Target Player: {player_description}

Rules:
- Track ONLY this player throughout the video
- Maintain identity continuity (never switch players)
{identity_rules}

{presence_and_ambiguity}

==================================================
2. VALIDATION
==================================================

Before starting analysis:

1. Confirm the target player has been identified
{validation_cues}
3. State confidence level:
   - High Confidence
   - Medium Confidence
   - Low Confidence

If confidence is LOW:
- Proceed with caution
- Only include highly certain actions
- Increase use of "Unclear – Not counted"

==================================================
3. OBSERVATION RULES
==================================================

Evidence Requirement:
- Every key claim must be supported by observed actions
- Do not include unsupported generalizations
- Prefer to anchor claims to a timestamp; if you cannot point to when it happened,
  do not assert it happened.

Sample Size Classification:
- Consistent Pattern → observed multiple times
- Occasional → observed more than once but not consistent
- Single Instance → observed only once

Category Handling (action TYPES that never occur, as opposed to visibility issues):
- If an action type never occurs in the footage (e.g., no aerial duels, no 1v1 dribbles, no progressive carries):
  → Label that subsection: "Not Observed in this video – no instances to assess"
- Do NOT infer ability, assign a rating, or comment on categories with zero instances
- Do NOT pad analysis with generic or hypothetical commentary to fill a section
- Only include sub-skills in Sections 4B/4C that have at least one observed instance
- List all zero-instance categories together at the end of Section 4, rather than omitting them silently or scattering "not observed" notes throughout

Conservative Bias (STRICT — this is the default posture):
- When you are not visually sure whether an action occurred, or who performed it, DO
  NOT count it. Prefer "Unclear – Not counted" over any assumption.
- Omitting a section is CORRECT behavior when the evidence isn't there — it is always
  better than filling it with a guess. Absence of evidence is never negative evidence.
- Under Medium or Low identification confidence: minimise claims, widen use of
  "Unclear – Not counted", present the score as provisional, and default the verdict
  to "monitor".

General Rules:
- Focus only on observable actions
- Do not assume intent without visual proof
- Prioritize accuracy over volume — but accuracy does not mean brevity. If the footage
  contains enough observed actions to support a thorough analysis, you are expected to
  use all of that evidence. Do not compress a video rich in observable actions into a
  short write-up; match the depth of your output to the depth of evidence available.

==================================================
4. ANALYSIS
==================================================

A. Game Phase Analysis

1. Organized Attack
- Positioning between lines
- Build-up contribution
- Line-breaking ability (pass, carry, movement)

2. Offensive Transition
- Reaction after regain
- Decision making (retain vs exploit)
- Verticality and support angles

3. Organized Defense
- Pressing role (trigger, cover, contain)
- Block positioning (high/mid/low)
- Team compactness

4. Defensive Transition
- Counter-press intensity
- Recovery runs (speed and angles)
- Space protection priorities

5. Set Pieces
- Role and execution quality

--------------------------------------------------

B. Technical Assessment

Classify all actions with at least one observed instance as:
- Effective (Good)
- Ineffective (Needs Improvement)

Offensive:
- First touch
- Passing (short, long, progressive)
- Dribbling
- Shooting
- Ball carrying
- Aerial ability

Defensive:
- Tackling
- Interceptions
- Marking
- Clearances
- Duels

Body Orientation:
- Open vs closed shape
- Receiving on half-turn
- Position relative to pressure

--------------------------------------------------

C. Tactical Intelligence

Offensive:
- Off-ball movement
- Space creation and usage
- Width and depth discipline
- Timing of runs
- Combination play

Defensive:
- Cover decisions
- Pressing triggers
- Spatial awareness
- Defensive support

--------------------------------------------------

D. Positional Discipline

- Role adherence
- Structural integrity
- Off-ball positioning consistency
- Impact on team shape

--------------------------------------------------

E. Four Corner Synthesis (AI Attribute Reasoning)
 
This section is a REASONING step, not a new observation step. Do NOT introduce any
new events here. Work ONLY from what you already recorded in Sections 4A–4D. Your job
is to convert the low-level observations above into higher-level PLAYER ATTRIBUTES,
organised under two corners of the FA Four Corner Model.
 
Reasoning ladder — for each attribute you assert, move up these three rungs:
- Event: a single observed action (already in 4A–4D).
- Pattern: the same behaviour recurring, or a consistent tendency across events.
- Attribute: what that pattern reveals about the player as a footballer.
State the attribute, then anchor it to the pattern/events that justify it. An attribute
with no traceable supporting observation in 4A–4D is FORBIDDEN — omit it instead.
 
Scope: ONLY the Technical and Tactical corners are in play. Do NOT assess Physical
(pace, stamina, strength) or Psychological/Social (mentality, leadership, communication)
attributes — these cannot be reliably judged from a single clip. If tempted to comment
on them, state "Not assessable from this footage" and move on.
 
E1. Technical Corner
Reason about what the player can reliably execute, and under what conditions execution
breaks down. Cover only the sub-areas that have observed evidence:
- Ball control / first touch — not "was it clean" but what the touch enabled (did it
  buy time and set up the next action, or cost him tempo?)
- Receiving — body shape BEFORE the ball arrives (open/half-turn, relative to pressure)
- Passing — separate range, weight, and selection; distinguish "cannot" from "does not"
- Dribbling / carrying — carrying into space vs. eliminating an opponent; control under pressure
- Ball retention — does shielding resolve into a productive action or just delay a turnover?
- Execution quality — name the conditions (pressure, zone) under which quality degrades
Write 1–3 connected sentences describing the technical profile, each claim evidence-anchored.
 
E2. Tactical Corner
Reason about whether the player understands the game and adapts as the picture changes.
Cover only the sub-areas with observed evidence:
- Positioning — did his position create a problem for the opposition, or just occupy space?
- Decision making — was a better option available and did he see it? (highest-value read)
- Pressing behaviour — trigger recognition, angle (cut a lane vs. chase the ball), coordination
- Build-up involvement — does he offer himself as an option in useful positions?
- Offensive transition — reaction speed + retain-vs-exploit judgement
- Defensive transition — counter-press intent, recovery-run angle (which space he protects)
- Spatial awareness — does his first action imply he scanned before receiving?
- Support play — distance and angle of support to the ball carrier
- Role discipline — holds structure when needed, breaks it at the right moments
- Game intelligence — does behaviour change appropriately with phase, pressure, game state?
Write 1–3 connected sentences describing the tactical profile, each claim evidence-anchored.
 
E3. Technical–Tactical Link (only where the footage supports it)
Where a tendency could stem from EITHER a technical limit OR a decision pattern, say
which the evidence points to, and why. Example: "His lateral recycling looks like a
selection pattern rather than a range limit — the one longer pass he attempted was
cleanly struck." This distinction (a fixable habit vs. a ceiling) is the single most
valuable output of this section for development and recruitment. If the footage does
not let you separate the two, say so explicitly rather than guessing.
 
Hard rules for this section:
- No attribute without a traceable observation in 4A–4D.
- If a corner lacks sufficient evidence, state that plainly — do not manufacture a profile.
- Do NOT restate the raw event list; the value here is the interpretation, not a recap.

--------------------------------------------------

F. Categories Not Observed

List every action type from Sections 4A–4E that had zero instances in this video (e.g., "Aerial duels: Not Observed", "1v1 dribbles: Not Observed", "Progressive carries: Not Observed").
This list must be factored into the confidence of the Final Output and the "score"/"verdict" fields below.

==================================================
5. EVALUATION
==================================================

Final Output:

- Strengths (3–5, evidence-based; skip or reduce count if insufficient instances exist)
- Areas for Improvement (3–5, actionable; only drawn from attempted-but-poorly-executed actions — never inferred from a zero-instance category)
- Player Profile (playing style + best-fit role/system)
- Overall Rating (/10 with justification, consistent with the "score" field below)

--------------------------------------------------

Coach's Verdict:

- Player Type
- Current Level
- Readiness
- Key Differentiator
- Main Limitation
- Recommendation

==================================================
6. STRUCTURED OUTPUT (REQUIRED)
==================================================

Return your final answer as a single JSON object matching this schema exactly. The
fields are generated IN THIS ORDER — decide the gate first, then write the full report,
then summarize:

{{
  "content_check": "<one of: football, not_football, player_not_found, unclear>",
  "detailed_analysis": "<string>",
  "summary": "<string>",
  "score": <float, 0.0–10.0>,
  "verdict": "<one of: sign, monitor, pass>"
}}

Field rules:

- "content_check":
  The Section 0 gate. Decide this FIRST. If it is not "football", every field below is
  an explanation of what went wrong, not an analysis (score 0.0, verdict "monitor").

- "detailed_analysis":
  When content_check == "football": the COMPLETE Sections 1–5 report in markdown. Its
  VERY FIRST line MUST be the single opening H1 heading "# DETAILED ANALYSIS" (translated
  into the output language per the OUTPUT LANGUAGE section), followed by a blank line —
  this is the ONLY H1 in the field and marks where the detailed report begins. Below it,
  and using H3+ headings, cover identification, validation, game-phase analysis, technical
  assessment, tactical intelligence, positional discipline, the Four Corner synthesis
  (Technical & Tactical attribute reasoning), the "Categories Not Observed" list, and the
  full evaluation + coach's verdict. Write this FIRST and in full depth; scale its length
  to the evidence in the footage. Omit any sub-skill or section with zero clearly-observed
  instances (and record it under Categories Not Observed) rather than padding it.
  When content_check != "football": a short, plain explanation of why the clip is not
  analyzable — no scouting narrative.

- "summary":
  When content_check == "football": a self-contained committee summary that will be read
  WITHOUT "detailed_analysis" — so it must be self-contained and substantive on its own.
 
  Format — return the summary in EXACTLY this structure, and nothing else. Output
  bullet points only: no paragraphs, no prose blocks, no markdown headings.
  Begin directly with the "• Player:" line. Use "•" for each heading and "  -" (indented)
  for sub-items. Put a blank line between each block.

  HARD UI RULE — the summary MUST NOT contain any H1 heading (a line starting with
  "# ") or H2 heading (a line starting with "## "). A single "#" or "##" heading in the
  summary breaks the way the summary is rendered and corrupts the UI. This rule is
  absolute and takes priority over everything else if there is ever any doubt. (The "•"
  bullet format above already avoids headings — this is a belt-and-suspenders guard.)

  • Player:
    <player_description> (<Confidence Level>)

  • Profile:
    <2–3 sentence player profile: playing style, best position, tactical fit — concise>

  • Key Strengths:
    - <observed strength>
    - <observed strength>
    - <observed strength>

  • Areas for Improvement:
    - <observed weakness>
    - <observed weakness>

  • Confidence:
    <e.g. "High confidence in technical and attacking observations." plus, only if it
    materially affects the evaluation, a line such as "Defensive assessment remains
    provisional due to limited defensive actions.">

  • Rating:
    <score> / 10

  • Recommendation:
    <one short recruitment recommendation, aligned with the verdict, e.g.
    "Monitor with additional footage before making a recruitment decision." /
    "Recommend signing based on consistently observed strengths." /
    "Pass based on consistently observed technical deficiencies.">

  Rules for each block:
  - Player: use the supplied player_description verbatim; NEVER invent or output a player
    name (e.g. Laith, Ahmed, John, Messi, Ronaldo) or any identity — a name may appear
    ONLY if it is explicitly present in the Target Player description above.
  - Profile: concise, no more than 2–3 sentences. Fold in the distinctive Technical and
    Tactical attributes from the Four Corner synthesis (Section 4E), phrased as attributes
    (e.g. "progresses possession through selection, not volume").
  - Key Strengths: only strengths actually observed, each tied to a real instance/pattern.
    Maximum 4 bullets. No paragraphs, no generic praise.
  - Areas for Improvement: only weaknesses actually observed (attempted-but-flawed actions).
    Maximum 4 bullets. NEVER infer a weakness from a "Not Observed" category.
  - Confidence: state confidence in what WAS observed. Mention a limitation only if it
    materially affects the evaluation. Do not use generic hedging like "the assessment is
    limited."
  - Rating: "<score> / 10". Append " (Provisional)" ONLY when many important categories
    were Not Observed.
  - Recommendation: meaningful and aligned with the verdict — a short explanation, not a
    single word.
 
  When content_check != "football": the same plain explanation of what went wrong as in
  "detailed_analysis", phrased for the committee. Do not invent a player or a report.

- "score":
  - Float, 0.0 to 10.0, one decimal place.
  - Must be fully justified by evidence already stated in "summary" —
    introduce no new claims at this stage.
  - Weigh: technical execution, tactical intelligence, decision-making,
    and game-phase impact. Do NOT weigh by sheer volume of actions observed.
  - If a substantial share of expected action categories were
    "Not Observed", explicitly state in "summary" that the score is
    provisional / based on limited evidence.
  - If content_check != "football", score is 0.0.

- "verdict":
  - Must be exactly one of: ["sign", "monitor", "pass"]

  - sign: Strong, well-evidenced performance across multiple categories.
    Player shows clear quality with consistent patterns (not single
    instances) supporting a confident recruitment recommendation.

  - pass: Clear, well-evidenced weaknesses or poor decision-making/execution
    observed with sufficient sample size to be confident the player does
    not fit. Do NOT use pass just because evidence is thin — pass requires
    actual negative evidence, not absence of evidence.

  - monitor: Use this as the default when:
      a) Evidence is mixed (real strengths AND real weaknesses), OR
      b) A substantial share of expected action categories were
         "Not Observed" or "Unclear – Not counted", making a confident
         sign/pass call unsupported by the available footage, OR
      c) Identification confidence was Medium or Low, OR
      d) content_check != "football" (un-analyzable clip).
    monitor means "promising or unclear, needs more footage" — it is the
    correct verdict for thin or inconclusive evidence, not a weak sign
    or weak pass.

  - Verdict must logically follow from "score" and "summary":
      - sign should not pair with a low score (e.g. below ~7.0)
      - pass should not pair with a high score (e.g. above ~5.0)
      - monitor covers the middle ground and all low-confidence/thin-evidence cases
  - When in doubt between two verdicts, prefer monitor over forcing a
    definitive sign or pass from incomplete data.

Output ONLY the JSON object. No preamble, no markdown code fences, no
commentary before or after the JSON.

==================================================
7. MARKDOWN LINE STRUCTURE (CRITICAL — READ CAREFULLY)
==================================================

"detailed_analysis" is rendered as markdown, so its LINE STRUCTURE matters as much as
its words. Inside the JSON string value you MUST use real line breaks (newline
characters) to separate every block. If you run blocks together on one line, the
markdown collapses into a single unreadable heading and the entire report is wasted.

Hard rules for "detailed_analysis":
- The VERY FIRST line is the single opening H1 heading ("# DETAILED ANALYSIS", translated
  to the output language). It is the ONLY H1 allowed in the field.
- Every other heading MUST be on its own line, with a BLANK LINE before and after it. Use
  H3 or deeper (###, ####) for those — never H1 (#) or H2 (##) again after the first line.
- Every bullet MUST start on its own line. Use "- " for bullets.
- NEVER place two markdown blocks on the same line. A heading and its first bullet are
  TWO blocks — they go on two separate lines.
- Put a blank line between paragraphs and between a heading and the text under it.
- Each table row (if any) goes on its own line.

CORRECT — real line breaks and blank lines between blocks (note the single opening H1):

# DETAILED ANALYSIS

### 1. PLAYER IDENTIFICATION

- Target Player: the player identified in the section above.
- Visual Cues: the observed cues used to keep track of them.

### 2. VALIDATION

- Confidence Level: High.

WRONG — everything on one line (this is the failure mode; DO NOT DO THIS):

### 1. PLAYER IDENTIFICATION - Target Player: the player identified above ... - Visual Cues: ... ### 2. VALIDATION - Confidence Level: High.

==================================================
GLOBAL INSTRUCTIONS
==================================================

- Avoid repetition across sections
- Be thorough, not just concise: depth of evidence in the footage should be reflected
  in depth of the "detailed_analysis" and "summary" fields — do not sacrifice substance
  for brevity
- Focus only on observable actions
- Do not assume intent without visual proof
- Do not fabricate analysis for action types not present in the footage
- When unsure, abstain ("Unclear – Not counted" / "Not Observed") rather than guess
- Prioritize insight over volume, but do not let that become an excuse for a
  thin or perfunctory report when the footage supports more
"""

class AIClipContentCheck(str, Enum):
    """Content gate for the LLM analysis — an adapter-internal concept.

    Not persisted and not part of the domain model (unlike AIClipVerdict): it only
    decides, inside this adapter, whether a clip is analyzable, so it lives here.
    """

    football = "football"
    not_football = "not_football"
    player_not_found = "player_not_found"
    unclear = "unclear"


class _VideoAnalysisSchema(BaseModel):
    """Structured output schema returned by the LLM.

    Field order is deliberate — the model generates JSON top-to-bottom:
    ``content_check`` gates the clip first, then ``detailed_analysis`` forces the
    full Sections 1–5 report to be written out, so ``summary`` has real content to
    summarize instead of collapsing the whole analysis into a few sentences.
    """

    content_check: AIClipContentCheck = Field(
        description="Content gate: only 'football' proceeds to a real analysis; the "
        "other values mean the clip is not analyzable and the text fields explain why."
    )
    detailed_analysis: str = Field(
        description="Full Sections 1–5 scouting report in markdown when analyzable; "
        "otherwise a plain explanation of why the clip could not be analyzed."
    )
    summary: str = Field(
        description="Self-contained committee summary in scout-report; on a "
        "rejected clip, the same plain explanation of what went wrong."
    )
    score: float = Field(
        ge=0,
        le=10,
        description="Overall performance score between 0 and 10; 0.0 on a rejected clip.",
    )
    verdict: AIClipVerdict = Field(description="Overall recruitment recommendation.")

    @field_validator("verdict", mode="before")
    @classmethod
    def _coerce_verdict(cls, v: object) -> AIClipVerdict:
        """Normalize casing/whitespace and default anything off-list to monitor.

        The response schema already constrains Gemini to the three values, so this
        is a safety net: an unexpected verdict (odd casing like "Sign"/"PASS", the
        text-fallback parse, or a future model) becomes ``monitor`` — flag for
        human review — instead of crashing the analysis or asserting an unsafe
        sign/pass.
        """
        if isinstance(v, AIClipVerdict):
            return v
        if isinstance(v, str):
            try:
                return AIClipVerdict(v.strip().lower())
            except ValueError:
                return AIClipVerdict.monitor
        return AIClipVerdict.monitor

    @field_validator("content_check", mode="before")
    @classmethod
    def _coerce_content_check(cls, v: object) -> AIClipContentCheck:
        """Default anything off-list to 'unclear' so an odd value fails safe (no analysis)."""
        if isinstance(v, AIClipContentCheck):
            return v
        if isinstance(v, str):
            try:
                return AIClipContentCheck(v.strip().lower())
            except ValueError:
                return AIClipContentCheck.unclear
        return AIClipContentCheck.unclear


class GeminiVideoAnalyzer(VideoAnalyzer):
    """Sends the GCS-hosted video to Gemini and returns a structured review."""

    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    async def _video_part(self, data: ClipAnalysisInput) -> types.Part:
        """The Gemini ``Part`` carrying the clip's video.

        Overridable seam: local mode (``LocalGeminiVideoAnalyzer``) swaps the GCS reference
        for the file's bytes read from disk. Everything else about the analysis is shared.
        """
        # Reference the downsized video directly in GCS — no download needed.
        return types.Part.from_uri(
            file_uri=data.video_uri.uri,
            mime_type="video/mp4",
        )

    async def analyze(self, data: ClipAnalysisInput) -> VideoAnalysis:
        video_part = await self._video_part(data)
        if data.is_tracked:
            # Tracked clips carry no ident_text; the box is the identity. Keep this exactly
            # "the tracked player" so Section 6's verbatim rule matches the tracking directive.
            player_description = "the tracked player"
        else:
            player_description = data.ident_text or "the primary player featured in the clip"
        instructions = prompt.format(
            player_description=player_description.upper(),
            video_type=data.action_tag_key,
            language_directive=_language_directive(data.language),
            **_mode_fragments(data.is_tracked),
        )

        config = types.GenerateContentConfig(
            temperature=0.0,
            top_p=0.1,
            # Constrain the model to our structured schema.
            response_mime_type="application/json",
            response_schema=_VideoAnalysisSchema,
        )
        logger.info("Gemini video analysis start: clip=%s model=%s", data.clip_id, self._model)
        started = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[video_part, instructions],
            config=config,
        )
        logger.info(
            "Gemini video analysis done: clip=%s took=%.2fs",
            data.clip_id,
            time.monotonic() - started,
        )

        tokens = self._extract_token_usage(response)
        logger.info(
            "Gemini token usage: clip=%s prompt=%s output=%s thoughts=%s cached=%s total=%s",
            data.clip_id,
            tokens.prompt_tokens,
            tokens.output_tokens,
            tokens.thoughts_tokens,
            tokens.cached_tokens,
            tokens.total_tokens,
        )

        result: _VideoAnalysisSchema | None = response.parsed
        if result is None:
            # Fallback
            result = _VideoAnalysisSchema.model_validate_json(response.text)


        logging.info("Gemini video analysis result: %s", result)

        return VideoAnalysis(
            review=self._build_review(result),
            score=float(result.score or 0),
            verdict=result.verdict.value,
            tokens=tokens,
        )

    @staticmethod
    def _extract_token_usage(response: object) -> TokenUsage:
        """Pull the token counts off the Gemini response's ``usage_metadata``.

        Defensive on purpose: ``usage_metadata`` (and any individual count) can be
        absent on a partial/failed response, and ``thoughts``/``cached`` are ``None``
        on models that don't report them. Never let missing usage break the analysis.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            thoughts_tokens=getattr(usage, "thoughts_token_count", None),
            cached_tokens=getattr(usage, "cached_content_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        """Turn literal escape sequences from the model into real characters."""
        return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")

    # A line starting with one or two '#' followed by whitespace — an H1/H2 heading.
    # H3+ ('### ') never matches (the char after '##' is '#', not whitespace).
    _TOP_HEADING_RE = re.compile(r"^#{1,2}[ \t]+", re.MULTILINE)

    @classmethod
    def _demote_summary_headings(cls, text: str) -> str:
        """Downgrade any H1/H2 heading in the summary to H3."""
        return cls._TOP_HEADING_RE.sub("### ", text)

    @classmethod
    def _build_review(cls, result: _VideoAnalysisSchema) -> str:
        summary = cls._demote_summary_headings(
            cls._normalize_markdown(result.summary or "")
        ).strip()
        detail = (
            cls._normalize_markdown(result.detailed_analysis or "").strip()
            if result.content_check == AIClipContentCheck.football
            else ""
        )

        # detailed_analysis is prompted to open with its own "# DETAILED ANALYSIS" H1
        # (translated to the review's language), so we simply join the two halves — that
        # heading is the boundary a consumer splits on. The summary is H1/H2-free (see
        # _demote_summary_headings), so it is the first H1 in the review.
        if summary and detail:
            return f"{summary}\n\n{detail}"
        if summary or detail:
            return summary or detail

        logger.warning(
            "Empty review for clip: content_check=%s — no summary or detail",
            result.content_check.value,
        )
        return ""

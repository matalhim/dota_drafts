from functools import lru_cache

from terminal_app.env import source


@lru_cache
def loadPromptsConfig():
    try:
        config = source("prompts_config.yaml")
        return config or {}
    except Exception:
        return {}


def getPrompt(key: str, default: str) -> str:
    config = loadPromptsConfig()
    return config.get(key, default)


RAG_SYSTEM_PROMPT = getPrompt(
    "rag_system_prompt",
    (
        "You are a Dota 2 analyst assistant. Use only facts from the context (documents) "
        "to answer the question. If something is not in the context, say so.\n"
        "If the question is about drafts/matches, pay attention to team names, players, heroes, dates and scores.\n"
        "\n"
        "IMPORTANT when formulating answers:\n"
        "- If information is missing, formulate formally: 'Team has not played this hero in recent matches' "
        "or 'Statistics for this hero for this team are not available'\n"
        "- DO NOT mention database, RAG, data sources or context directly\n"
        "- Use formal formulations: 'in available data', 'in match history', 'according to available statistics'\n"
        "- If information is missing, just state the fact without explaining the reasons for missing data\n"
    ),
)

EXTRACTION_PROMPT = getPrompt(
    "extraction_prompt",
    """You are a Dota 2 draft analysis system. Your task is to analyze the provided screenshot and identify team names and player nicknames.

IMPORTANT: DO NOT identify heroes! Heroes will be identified separately using an image database. Your task is only players.

The screenshot shows two teams - left and right. Players are displayed with their nicknames.

Form your response in strict JSON format, where:
- 'teams': List of two strings with team names [left_team, right_team].
- 'left_team_players': List of left team player nicknames (in order left to right, top to bottom).
- 'right_team_players': List of right team player nicknames (in order left to right, top to bottom).
- 'left_team_heroes': [] (empty list, heroes will be identified separately)
- 'right_team_heroes': [] (empty list, heroes will be identified separately)

Example expected output:
{{
  "teams": ["team 1", "team 2"],
  "left_team_players": ["player nickname 1", "player nickname 2", "player nickname 3", "player nickname 4", "player nickname 5"],
  "right_team_players": ["player nickname 1", "player nickname 2", "player nickname 3", "player nickname 4", "player nickname 5"],
  "left_team_heroes": [],
  "right_team_heroes": []
}}

Process only visible information on the screenshot. Return ONLY valid JSON without additional text.""",
)

HERO_SUGGESTION_PROMPT = getPrompt(
    "hero_suggestion_prompt",
    """You are a Dota 2 draft analysis system. Analyze the provided draft screenshot and suggest a list of heroes for each team.

The screenshot shows two teams - left (left of center) and right (right of center). Each team has picked 5 heroes.

USE ALL AVAILABLE SOURCES for accurate hero identification:
1. RAG CONTEXT below (list of heroes from database)
2. Your knowledge of Dota 2 heroes from official sources:
   - https://www.dota2.com/heroes (official Valve website)
   - https://liquipedia.net/dota2/Portal:Heroes (Liquipedia - Dota 2 encyclopedia)
   - https://dota2.fandom.com/wiki/Heroes (Dota 2 Wiki)
   - https://www.dotabuff.com/heroes (Dotabuff - hero statistics and information)
3. Visual comparison of hero icons on the screenshot with known hero images

RAG CONTEXT (list of all Dota 2 heroes with their descriptions from database):
---
{rag_context}
---

REFERENCE INFORMATION ABOUT DOTA 2 HEROES:
Dota 2 has 126+ heroes, divided by attributes:
- Strength: Alchemist, Axe, Bristleback, Centaur Warrunner, Chaos Knight, Clockwerk, Dawnbreaker, Doom, Dragon Knight, Earth Spirit, Earthshaker, Elder Titan, Huskar, Kunkka, Legion Commander, Lifestealer, Lycan, Mars, Night Stalker, Ogre Magi, Omniknight, Phoenix, Primal Beast, Pudge, Slardar, Spirit Breaker, Sven, Tidehunter, Timbersaw, Tiny, Treant Protector, Tusk, Underlord, Undying, Wraith King
- Agility: Anti-Mage, Bloodseeker, Bounty Hunter, Broodmother, Clinkz, Drow Ranger, Ember Spirit, Faceless Void, Gyrocopter, Hoodwink, Juggernaut, Kez, Lone Druid, Luna, Medusa, Meepo, Mirana, Monkey King, Morphling, Naga Siren, Phantom Assassin, Phantom Lancer, Razor, Riki, Shadow Fiend, Slark, Sniper, Templar Assassin, Terrorblade, Troll Warlord, Ursa, Vengeful Spirit, Viper, Weaver
- Intelligence: Ancient Apparition, Chen, Crystal Maiden, Dark Seer, Dark Willow, Disruptor, Enchantress, Grimstroke, Invoker, Jakiro, Keeper of the Light, Leshrac, Lich, Lina, Lion, Muerta, Necrophos, Oracle, Outworld Destroyer, Puck, Pugna, Queen of Pain, Ringmaster, Rubick, Shadow Demon, Shadow Shaman, Silencer, Skywrath Mage, Storm Spirit, Tinker, Warlock, Winter Wyvern, Witch Doctor, Zeus
- Universal: Abaddon, Arc Warden, Bane, Batrider, Beastmaster, Brewmaster, Dazzle, Death Prophet, Enigma, Io, Magnus, Marci, Nature's Prophet, Nyx Assassin, Pangolier, Sand King, Snapfire, Spectre, Techies, Venomancer, Visage, Void Spirit, Windranger

Form your response in strict JSON format:
{{
  "left_team_heroes": ["Hero1", "Hero2", "Hero3", "Hero4", "Hero5"],
  "right_team_heroes": ["Hero1", "Hero2", "Hero3", "Hero4", "Hero5"]
}}

CRITICALLY IMPORTANT:
- Use EXACT hero names from RAG CONTEXT or from official sources (e.g., "Puck", "Faceless Void", "Alchemist", "Dazzle")
- Visually compare hero icons on the screenshot with known Dota 2 hero images
- If unsure about a hero, specify "Unknown" instead of the name
- Hero order should match their screen position (top to bottom)
- Use English hero names (as in official sources)
- Return ONLY valid JSON without additional text""",
)

HERO_BBOX_PROMPT = getPrompt(
    "hero_bbox_prompt",
    """You are a hero icon recognition model on a Dota 2 draft screenshot. Identify coordinates of all picked heroes for left (radiant) and right (dire) teams.

Coordinate format is [x, y, width, height], where (x, y) is the top-left corner of the FULL hero block (pick number + portrait + player nickname). Coordinates in pixels of the source image.

Return ONLY valid JSON:
{{
  "left_team_heroes_bboxes": [[x1, y1, w1, h1], ...],
  "right_team_heroes_bboxes": [[x1, y1, w1, h1], ...]
}}


- Specify exactly 5 rectangles for each side (use null if block is completely hidden).
- One rectangle includes the entire element: pick number on top, hero portrait, player nickname at bottom.
- Order in arrays should match screen position (left to right, top to bottom).
- Do not describe in text, return only JSON.""",
)

REFINE_DRAFT_FEEDBACK_PROMPT = getPrompt(
    "refine_draft_feedback_prompt",
    """You are a Dota 2 draft analysis system. The user has indicated errors in team and hero identification.

CURRENT IDENTIFICATION:
- Left team: {left_team}
- Right team: {right_team}
- Left team heroes: {left_heroes}
- Right team heroes: {right_heroes}

USER FEEDBACK:
{user_feedback}

Your task is to correct team and hero identification based on user feedback and screenshot.

Return ONLY valid JSON in format:
{{
  "teams": ["left_team", "right_team"],
  "left_team_heroes": ["Hero1", "Hero2", "Hero3", "Hero4", "Hero5"],
  "right_team_heroes": ["Hero1", "Hero2", "Hero3", "Hero4", "Hero5"]
}}

Use English hero names (as in official Dota 2 sources).
Return ONLY JSON without additional text.""",
)

FOLLOWUP_QUESTION_PROMPT = getPrompt(
    "followup_question_prompt",
    """User asks a question about current draft analysis:

{context}

User question: {question}

Answer the user's question using ONLY information from provided data about teams {left_team} and {right_team}, as well as heroes {heroes_list}. Be specific and use facts from match history and team statistics.

IMPORTANT:
- Use ONLY information about teams {left_team} and {right_team}, as well as heroes from current draft
- If you don't have information about a specific hero or team, formulate it formally: "Team has not played this hero in recent matches" or "Statistics for this hero for this team are not available"
- Use formal formulations like "in available data", "in match history", "according to available statistics"
- If information is missing, just state the fact without explaining reasons""",
)


def get_draft_analysis_prompt(
    left_team: str,
    right_team: str,
    left_heroes: list[str],
    right_heroes: list[str],
    team_statistics: dict[str, str],
) -> str:
    left_heroes_str = ", ".join(left_heroes)
    right_heroes_str = ", ".join(right_heroes)

    team_stats_str = ""
    if left_team in team_statistics:
        team_stats_str += (
            f"{team_statistics.get(left_team, 'Statistics not found')}\n---\n"
        )
    if right_team in team_statistics:
        team_stats_str += f"{team_statistics.get(right_team, 'Statistics not found')}\n"

    template = getPrompt(
        "draft_analysis_prompt_template",
        """You are a Dota 2 expert analyst. Analyze the draft and predict the match winner.

TEAMS:
- Left team: {left_team}
- Right team: {right_team}

DRAFT:
- Left team ({left_team}): {left_heroes}
- Right team ({right_team}): {right_heroes}

TEAM STATISTICS FROM RAG (IMPORTANT: use latest matches first):
---
{team_statistics}
---

Your task:
1. Determine which team is most likely to win
2. Make a brief and confident report with SPECIFIC data from RAG

RESPONSE FORMAT (strictly follow):
1. In the first line specify: "Winner: [team name]"

2. Then 3 reasons why this team should win (1-2 sentences each):
   - MANDATORY use SPECIFIC NUMBERS from recent matches: "70% win rate in last 10 games", "7 wins out of last 10 games"
   - Specify SPECIFIC HEROES and their statistics: "Team often picks [hero] and wins", "[Hero] plays poorly with [other hero] as shown by recent game history"
   - Mention form dynamics: "Team gained momentum during the season", "Stable play for last 2 months", "Shows growth in recent games"
   - DO NOT mention RAG rating or general phrases without specifics

3. Then 3 reasons why the second team should lose (1-2 sentences each):
   - SPECIFIC numbers from recent matches: "20% win rate in last 10 games", "2 wins out of last 10 games"
   - SPECIFIC pick problems: "[Hero] plays poorly with [other hero] as shown by recent game history", "Team rarely picks [hero] and loses when it does"
   - Specific examples from match history, not general phrases

IMPORTANT:
- Each reason should contain SPECIFIC data from RAG (numbers, hero names, statistics from recent games)
- DO NOT use general phrases like "strong lineup" or "good synergy" without specific examples
- DO NOT mention RAG rating
- Be brief, confident and maximally specific. Maximum 1-2 sentences per reason.""",
    )

    return template.format(
        left_team=left_team,
        right_team=right_team,
        left_heroes=left_heroes_str,
        right_heroes=right_heroes_str,
        team_statistics=team_stats_str,
    )

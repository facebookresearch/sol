import os
import pickle
import re
from collections import defaultdict
import numpy as np
import inflect

# NLE-specific tokenizer.
# This parses the messages from: https://gist.github.com/tckmn/8078a34e3287ec32dadf
# and also has most (all?) of the names entities (monsters, objects, etc) I could find
# from the NetHack wiki, parsed with the help of LLaMA <3
# this might be incomplete, if you find missing tokens please add them.

# Note: I tried using HF tokenizers, but they were too slow.

# Note: we represent these as dicts mapping tuples of numeric characters to tokens.
# This is so we can skip the step of decoding characters to strings and then to tokens,
# which made things too slow.

# this does some simple NLP, like adding plural forms
engine = inflect.engine()


def extract_strings(file_path):
    strings = []
    with open(file_path, 'r') as file:
        for line in file:
            matches = re.findall(r'"([^"]+)"', line)
            if matches:
                s = matches[0].replace('%s', '')
                s = s.strip().replace('  ', ' ').replace('.', '').replace('!', '')
                if s:
                    strings.append(s)
    return strings

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tokens.txt'), "r") as f:
    DND_TOKENS = f.read().split("\n")
    DND_TOKENS.append("menu")

MESSAGES = extract_strings(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gistfile1.txt'))


MONSTERS = [
    "giant ant",
    "killer bee",
    "soldier ant",
    "fire ant",
    "giant beetle",
    "queen bee",
    "acid blob",
    "quivering blob",
    "gelatinous cube",
    "chickatrice",
    "cockatrice",
    "pyrolisk",
    "jackal",
    "fox",
    "coyote",
    "werejackal",
    "little dog",
    "dingo",
    "dog",
    "large dog",
    "wolf",
    "werewolf",
    "winter wolf cub",
    "warg",
    "winter wolf",
    "hell hound pup",
    "hell hound",
    "Cerberus",
    "gas spore",
    "floating eye",
    "freezing sphere",
    "flaming sphere",
    "shocking sphere",
    "beholder",
    "kitten",
    "housecat",
    "jaguar",
    "lynx",
    "panther",
    "large cat",
    "tiger",
    "gremlin",
    "gargoyle",
    "winged gargoyle",
    "hobbit",
    "dwarf",
    "bugbear",
    "dwarf lord",
    "dwarf king",
    "mind flayer",
    "master mind flayer",
    "manes",
    "homunculus",
    "imp",
    "lemure",
    "quasit",
    "tengu",
    "blue jelly",
    "spotted jelly",
    "ochre jelly",
    "kobold",
    "large kobold",
    "kobold lord",
    "kobold shaman",
    "leprechaun",
    "small mimic",
    "large mimic",
    "giant mimic",
    "wood nymph",
    "water nymph",
    "mountain nymph",
    "goblin",
    "hobgoblin",
    "orc",
    "hill orc",
    "Mordor orc",
    "Uruk-hai",
    "orc shaman",
    "orc-captain",
    "rock piercer",
    "iron piercer",
    "glass piercer",
    "rothe",
    "mumak",
    "leocrotta",
    "wumpus",
    "titanothere",
    "baluchitherium",
    "mastodon",
    "sewer rat",
    "giant rat",
    "rabid rat",
    "wererat",
    "rock mole",
    "woodchuck",
    "cave spider",
    "centipede",
    "giant spider",
    "scorpion",
    "lurker above",
    "trapper",
    "pony",
    "white unicorn",
    "gray unicorn",
    "black unicorn",
    "horse",
    "warhorse",
    "fog cloud",
    "dust vortex",
    "ice vortex",
    "energy vortex",
    "steam vortex",
    "fire vortex",
    "baby long worm",
    "baby purple worm",
    "long worm",
    "purple worm",
    "grid bug",
    "xan",
    "yellow light",
    "black light",
    "zruty",
    "couatl",
    "Aleax",
    "Angel",
    "ki-rin",
    "Archon",
    "bat",
    "giant bat",
    "raven",
    "vampire bat",
    "plains centaur",
    "forest centaur",
    "mountain centaur",
    "baby gray dragon",
    "baby silver dragon",
    "baby shimmering dragon",
    "baby red dragon",
    "baby white dragon",
    "baby orange dragon",
    "baby black dragon",
    "baby blue dragon",
    "baby green dragon",
    "baby yellow dragon",
    "gray dragon",
    "silver dragon",
    "shimmering dragon",
    "red dragon",
    "white dragon",
    "orange dragon",
    "black dragon",
    "blue dragon",
    "green dragon",
    "yellow dragon",
    "stalker",
    "air elemental",
    "fire elemental",
    "earth elemental",
    "water elemental",
    "lichen",
    "brown mold",
    "yellow mold",
    "green mold",
    "red mold",
    "shrieker",
    "violet fungus",
    "gnome",
    "gnome lord",
    "gnomish wizard",
    "gnome king",
    "giant",
    "stone giant",
    "hill giant",
    "fire giant",
    "frost giant",
    "ettin",
    "storm giant",
    "titan",
    "minotaur",
    "jabberwock",
    "vorpal jabberwock",
    "Keystone Kop",
    "Kop Sergeant",
    "Kop Lieutenant",
    "Kop Kaptain",
    "lich",
    "demilich",
    "master lich",
    "arch-lich",
    "kobold mummy",
    "gnome mummy",
    "orc mummy",
    "dwarf mummy",
    "elf mummy",
    "human mummy",
    "ettin mummy",
    "giant mummy",
    "red naga hatchling",
    "black naga hatchling",
    "golden naga hatchling",
    "guardian naga hatchling",
    "red naga",
    "black naga",
    "golden naga",
    "guardian naga",
    "ogre",
    "ogre lord",
    "ogre king",
    "gray ooze",
    "brown pudding",
    "green slime",
    "black pudding",
    "quantum mechanic",
    "rust monster",
    "disenchanter",
    "garter snake",
    "snake",
    "water moccasin",
    "python",
    "pit viper",
    "cobra",
    "troll",
    "ice troll",
    "rock troll",
    "water troll",
    "Olog-hai",
    "umber hulk",
    "vampire",
    "vampire lord",
    "vampire mage",
    "Vlad the Impaler",
    "barrow wight",
    "wraith",
    "Nazgul",
    "xorn",
    "monkey",
    "ape",
    "owlbear",
    "yeti",
    "carnivorous ape",
    "sasquatch",
    "kobold zombie",
    "gnome zombie",
    "orc zombie",
    "dwarf zombie",
    "elf zombie",
    "human zombie",
    "ettin zombie",
    "ghoul",
    "giant zombie",
    "skeleton",
    "straw golem",
    "paper golem",
    "rope golem",
    "gold golem",
    "leather golem",
    "wood golem",
    "flesh golem",
    "clay golem",
    "stone golem",
    "glass golem",
    "iron golem",
    "human",
    "wererat",
    "werejackal",
    "werewolf",
    "elf",
    "Woodland-elf",
    "Green-elf",
    "Grey-elf",
    "elf-lord",
    "Elvenking",
    "doppelganger",
    "shopkeeper",
    "guard",
    "prisoner",
    "Oracle",
    "aligned priest",
    "high priest",
    "soldier",
    "sergeant",
    "nurse",
    "lieutenant",
    "captain",
    "watchman",
    "watch captain",
    "Medusa",
    "Wizard of Yendor",
    "Croesus",
    "Charon",
    "ghost",
    "shade",
    "water demon",
    "succubus",
    "horned devil",
    "incubus",
    "erinys",
    "barbed devil",
    "marilith",
    "vrock",
    "hezrou",
    "bone devil",
    "ice devil",
    "nalfeshnee",
    "pit fiend",
    "sandestin",
    "balrog",
    "Juiblex",
    "Yeenoghu",
    "Orcus",
    "Geryon",
    "Dispater",
    "Baalzebub",
    "Asmodeus",
    "Demogorgon",
    "Death",
    "Pestilence",
    "Famine",
    "mail daemon",
    "djinni",
    "jellyfish",
    "piranha",
    "shark",
    "giant eel",
    "electric eel",
    "kraken",
    "newt",
    "gecko",
    "iguana",
    "baby crocodile",
    "lizard",
    "chameleon",
    "crocodile",
    "salamander",
    "long worm tail",
    "archeologist",
    "barbarian",
    "caveman",
    "cavewoman",
    "healer",
    "knight",
    "monk",
    "priest",
    "priestess",
    "ranger",
    "rogue",
    "samurai",
    "tourist",
    "valkyrie",
    "wizard",
    "Lord Carnarvon",
    "Pelias",
    "Shaman Karnov",
    "Earendil",
    "Elwing",
    "Hippocrates",
    "King Arthur",
    "Grand Master",
    "Arch Priest",
    "Orion",
    "Master of Thieves",
    "Lord Sato",
    "Twoflower",
    "Norn",
    "Neferet the Green",
    "Minion of Huhetotl",
    "Thoth Amon",
    "Chromatic Dragon",
    "Goblin King",
    "Cyclops",
    "Ixoth",
    "Master Kaen",
    "Nalzok",
    "Scorpius",
    "Master Assassin",
    "Ashikaga Takauji",
    "Lord Surtur",
    "Dark One",
    "student",
    "chieftain",
    "neanderthal",
    "High-elf",
    "attendant",
    "page",
    "abbot",
    "acolyte",
    "hunter",
    "thug",
    "ninja",
    "roshi",
    "guide",
    "warrior",
    "apprentice",
]

DUNGEON_FEATURES = [
    'staircase',
    'ladder',
    'altar',
    'sink',
    'fountain',
    'tree',
    'trees',
    'door',
    'doorway',
    'wall',
    'drawbridge',
    'iron bars',
    'corridor',
    'throne',
    'grave',
    'water',
    'ice',
    'lava',
    'cloud',
    'air',
    'solid rock',
]

OBJECT_TYPES = [
    'potion',
    'scroll',
    'spellbook',
    'amulet',
    'ring',
    'wand',
]

POTIONS = [
    "acid",
    "blindness",
    "booze",
    "confusion",
    "enlightenment",
    "extra healing",
    "fruit juice",
    "full healing",
    "gain ability",
    "gain energy",
    "gain level",
    "hallucination",
    "healing",
    "holy water",
    "invisibility",
    "levitation",
    "monster detection",
    "object detection",
    "oil",
    "paralysis",
    "polymorph",
    "restore ability",
    "see invisible",
    "sickness",
    "sleeping",
    "speed",
    "unholy water",
    "water"
]

SCROLLS = [
    "mail",
    "identify",
    "light",
    "blank paper",
    "enchant weapon",
    "enchant armor",
    "remove curse",
    "confuse monster",
    "destroy armor",
    "fire",
    "food detection",
    "gold detection",
    "magic mapping",
    "scare monster",
    "teleportation",
    "amnesia",
    "create monster",
    "earth",
    "taming",
    "charging",
    "genocide",
    "punishment",
    "stinking cloud"
]

RINGS = [
    "adornment",
    "hunger",
    "protection",
    "protection from shape changers",
    "stealth",
    "sustain ability",
    "warning",
    "aggravate monster",
    "cold resistance",
    "gain constitution",
    "gain strength",
    "increase accuracy",
    "increase damage",
    "invisibility",
    "poison resistance",
    "see invisible",
    "shock resistance",
    "fire resistance",
    "free action",
    "levitation",
    "regeneration",
    "searching",
    "slow digestion",
    "teleportation",
    "conflict",
    "polymorph",
    "polymorph control",
    "teleport control"
]

AMULETS = [
    'change',
    'ESP',
    'life saving',
    'magical breathing',
    'reflection',
    'restful sleep',
    'strangulation',
    'unchanging',
    'poison',
]

SPELLBOOKS = [
    'force bolt',
    'drain life',
    'magic missile',
    'cone of cold',
    'fireball',
    'finger of death',
    'protection',
    'create monster',
    'remove curse',
    'create familiar',
    'turn undead',
    'detect monsters',
    'detect food',
    'clairvoyance',
    'detect unseen',
    'identify',
    'detect treasure',
    'magic mapping',
    'sleep',
    'confuse monster',
    'slow monster',
    'cause fear',
    'charm monster',
    'jumping',
    'haste self',
    'invisibility',
    'levitation',
    'teleport away',
    'healing',
    'cure blindness',
    'cure sickness',
    'extra healing',
    'stone to flesh',
    'restore ability',
    'knock',
    'wizard lock',
    'dig',
    'polymorph',
    'cancellation',
    'blank paper',
]

WANDS = [
    "light",
    "nothing",
    "digging",
    "enlightenment",
    "locking",
    "magic missile",
    "make invisible",
    "opening",
    "probing",
    "secret door detection",
    "slow monster",
    "speed monster",
    "striking",
    "undead turning",
    "cold",
    "fire",
    "lightning",
    "sleep",
    "cancellation",
    "create monster",
    "polymorph",
    "teleportation",
    "death",
    "wishing"
]

WEAPONS = [
    "athame",
    "elven dagger",
    "worm tooth",
    "knife",
    "stiletto",
    "scalpel",
    "crysknife",
    "axe",
    "battle-axe",
    "pick-axe",
    "dwarvish mattock",
    "orcish short sword",
    "short sword",
    "dwarvish short sword",
    "elven short sword",
    "broadsword",
    "runesword",
    "elven broadsword",
    "long sword",
    "katana",
    "two-handed sword",
    "tsurugi",
    "scimitar",
    "silver saber",
    "club",
    "aklys",
    "mace",
    "morning star",
    "flail",
    "grappling hook",
    "war hammer",
    "quarterstaff",
    "partisan",
    "fauchard",
    "glaive",
    "bec de corbin",
    "spetum",
    "lucern hammer",
    "guisarme",
    "ranseur",
    "voulge",
    "bill-guisarme",
    "bardiche",
    "halberd",
    "orcish spear",
    "spear",
    "silver spear",
    "elven spear",
    "dwarvish spear",
    "javelin",
    "trident",
    "lance1",
    "orcish bow",
    "orcish arrow",
    "bow",
    "arrow",
    "elven bow",
    "elven arrow",
    "yumi",
    "ya",
    "silver arrow",
    "sling",
    "flint stone",
    "crossbow",
    "crossbow bolt",
    "dart",
    "shuriken",
    "boomerang",
    "bullwhip",
    "rubber hose",
    "unicorn horn",
]

ARMOR = [
    'water walking boots',
    'Uruk-hai shield',
    'T-shirt',
    'studded leather armor',
    'splint mail',
    'speed boots',
    'small shield',
    'shield of reflection',
    'scale mail',
    'robe',
    'ring mail',
    'plate mail',
    'iron skull cap',
    'orcish helm',
    'orcish cloak',
    'orcish chain mail',
    'oilskin cloak',
    'mummy wrapping',
    'low boots',
    'levitation boots',
    'leather jacket',
    'leather gloves',
    'leather cloak',
    'leather armor',
    'large shield',
    'kicking boots',
    'jumping boots',
    'iron shoes',
    'high boots',
    'helm of telepathy',
    'helm of opposite alignment',
    'helm of brilliance',
    'Hawaiian shirt',
    'gauntlets of power',
    'gauntlets of fumbling',
    'gauntlets of dexterity',
    'fumble boots',
    'fedora',
    'elven shield',
    'elven mithril-coat',
    'elven leather helm',
    'elven cloak',
    'elven boots',
    'dwarvish roundshield',
    'dwarvish mithril-coat',
    'dwarvish iron helm',
    'dwarvish cloak',
    'dunce cap',
    'dragon scales',
    'dented pot',
    'crystal plate mail',
    'cornuthaum',
    'cloak of protection',
    'cloak of magic resistance',
    'cloak of invisibility',
    'cloak of displacement',
    'chain mail',
    'bronze plate mail',
    'banded mail',
    'alchemy smock',
    # unidentified
    'jungle boots',
    'white-handed shield',
    'combat boots',
    'polished silver',
    'red-eyed shield',
    'crude ring mail',
    'iron skull cap',
    'coarse mantelet',
    'crude chain mail',
    'slippery cloak',
    'walking shoes',
    'snow boots',
    'old gloves',
    'buckled boots',
    'hiking boots',
    'hard shoes',
    'jackboots',
    'plumed helmet',
    'visored helmet',
    'crested helmet',
    'etched helmet',
    'fencing gloves',
    'riding gloves',
    'padded gloves',
    'riding boots',
    'blue and green shield',
    'leather hat',
    'faded pall',
    'mud boots',
    'large round shield',
    'hard hat',
    'hooded cloak',
    'conical hat',
    'tattered cape',
    'ornamental cope',
    'opera cloak',
    'piece of cloth',
    'apron',
]

GEMS = [
    'gem',
    'gems',
    'stone',
    'white',
    'red',
    'orange',
    'blue',
    'black',
    'green',
    'yellow',
    'violet',
    'yellowish brown',
    'gray',
    'rock',
    'rocks',
    'worthless glass',
    'touchstone',
    'luckstone',
    'flint stone',
    'citrine',
    'turquoise',
    'emerald',
    'black opal',
    'sapphire',
    'jacinth',
    'ruby',
    'diamond',
    'dilithium crystal',
    'agate',
    'jade',
    'obsidian',
    'jet',
    'fluorite',
    'turquoise',
    'amethyst',
    'amber',
    'garnet',
]

BEATITUDES = [
    'cursed',
    'uncursed',
    'blessed',
]

ALIGNMENTS = [
    'lawful',
    'neutral',
    'chaotic'
]

GODS = [
    'Ptah',
    'Thoth',
    'Anhur',
    'Tyr',
    'Odin',
    'Loki',
    'Blind Io',
    'The Lady',
    'Offler',
    'Amaterasu Omikami',
    'Raijin',
    'Susanowo',
    'Issek',
    'Mog',
    'Kos',
    'Mercury',
    'Venus',
    'Mars',
    'Shan Lai Ching',
    'Chih Sung-tzu',
    'Huan Ti',
    'Lugh',
    'Brigit',
    'Manannan Mac Lir',
    'Athena',
    'Hermes',
    'Poseidon',
    'Anu',
    'Ishtar',
    'Anshar',
    'Mitra',
    'Crom',
    'Set',
    'Quetzalcoatl',
    'Camaxtli',
    'Huhetotl',
    'Marduk',
    'Moloch',
    'Elbereth',
    'Arioch',
    'pleased',
    'well-pleased',
    'displeased',
    'satisfied',
]

ENCHANTMENTS = [f'-{k}' for k in range(7)] + [f'+{k}' for k in range(7)]

COMESTIBLES = [
    'food ration',
    'cram ration',
    'C-ration',
    'K-ration',
    'tripe ration',
    'lembas wafer',
    'cream pie',
    'pancake',
    'candy bar',
    'fortune cookie',
    'tin',
    'apple',
    'orange',
    'banana',
    'bananas',
    'sprig of wolfsbane',
    'pear',
    'slime mold',
    'clove of garlic',
    'melon',
    'carrot',
    'eucalyptus leaf',
    'kelp frond',
    'meatball',
    'meat ring',
    'meat stick',
    'huge chunk of meat',
    'egg',
    'lump of royal jelly',
    'lichen corpse',
    'lizard corpse',
]


OBJECT_TYPES += [engine.plural(item) for item in OBJECT_TYPES]
COMESTIBLES += [engine.plural(item) for item in COMESTIBLES]



SHOPS = [
    'general store',
    'used armor dealership',
    'second-hand bookstore',
    'liquor emporium',
    'antique weapons outlet',
    'delicatessen',
    'jewelers',
    'quality apparel and accessories',
    'hardware',
    'rare books',
    'health food',
    'lighting',
]

EROSIONS = [
    'burnt',
    'rusted',
    'fireproof',
    'rustproof',
    'greased',
]

RANDOM = [
    'nethack',
    'lunar',
    'phases',
    '(being',
    'worn)',
    'pair',
]



NUMBERS = [str(i) for i in range(10)]

DELIMITERS = ['\n']

SPECIAL_TOKENS = ['exit', 'menu']

VOCAB = []
VOCAB += POTIONS
VOCAB += SCROLLS
VOCAB += SPELLBOOKS
VOCAB += WANDS
VOCAB += ARMOR
VOCAB += WEAPONS
VOCAB += AMULETS
VOCAB += BEATITUDES
VOCAB += ENCHANTMENTS
VOCAB += NUMBERS
VOCAB += DELIMITERS
VOCAB += COMESTIBLES
VOCAB += EROSIONS
VOCAB += OBJECT_TYPES
VOCAB += SPECIAL_TOKENS


INV_TOKENIZER = defaultdict(int)

cnt = 1
for w in VOCAB:
    for p in w.split(' '):
        if p not in INV_TOKENIZER.keys():
            INV_TOKENIZER[p] = cnt
            cnt += 1

VOCAB += MESSAGES
VOCAB += MONSTERS
VOCAB += GEMS
VOCAB += DUNGEON_FEATURES
VOCAB += SHOPS
VOCAB += GODS
VOCAB += ALIGNMENTS
VOCAB += RANDOM


NLE_TOKENIZER = defaultdict(int)

cnt = 1
for w in VOCAB:
    for p in w.split(' '):
        if p not in NLE_TOKENIZER.keys():
            NLE_TOKENIZER[p] = cnt
            NLE_TOKENIZER[p.capitalize()] = cnt
            NLE_TOKENIZER[' ' + p.capitalize()] = cnt
            NLE_TOKENIZER[' ' + p] = cnt
            cnt += 1

NLE_UNIQUE_TOKENS = cnt

NLE_TOKENIZER_TUPLE_2_TOK = defaultdict(int)
NLE_TOKENIZER_TOK_2_STR = defaultdict(str)

for string, tok in NLE_TOKENIZER.items():
    key_np = np.array(list(string.encode("latin-1")), dtype=np.int64)
    NLE_TOKENIZER_TUPLE_2_TOK[tuple(key_np)] = tok
    NLE_TOKENIZER_TOK_2_STR[tok] = string

print(f"Finished building NLE tokenizer with {len(NLE_TOKENIZER)} tokens.")


def dnd_tokenizer_process_word(word):
    if "[" in word:
        return ""

    punctuation = "!?.,\"'-|()[]*%:;/\\"
    word = ''.join(char for char in word if char not in punctuation)

    return word

DND_PROCESS_CHAR_MAPPING = {c:c for c in range(256)}
DND_PUNCTUATION = [ord(p) for p in "!?.,\"'-|()[]*%:;/\\"]
# for p in punctuation:
#     DND_PROCESS_CHAR_MAPPING[ord(p)] = 0
#
# def dnd_tokenizer_process_tuple(tuple):
#     return tuple(DND_PROCESS_CHAR_MAPPING[c] for c in tuple)


DND_TOKENIZER = defaultdict(int)
DND_DETOKENIZER = defaultdict(str)
DND_TUPLE_TOKENIZER = defaultdict(int)

cnt = 1
for p in DND_TOKENS:
    if p and p not in DND_TOKENIZER.keys():
        DND_DETOKENIZER[cnt] = p

        for p_variant in [p, p.capitalize(), " " + p.capitalize(), " " + p]:
            DND_TOKENIZER[p_variant] = cnt

            key_np = np.array(list(p_variant.encode("latin-1")), dtype=np.int64)
            DND_TUPLE_TOKENIZER[tuple(key_np)] = cnt

        cnt += 1

print(f"Finished building DND tokenizer with {len(DND_TOKENIZER)} tokens.")

def process_message(msg):
    tokens = []
    for word in msg.split(" "):
        dnd_word = dnd_tokenizer_process_word(word)
        tokens.append(DND_TOKENIZER[dnd_word])
    tokens = [token for token in tokens if token != 0]
    tokens = list(sorted(tokens))

    token_words = [DND_DETOKENIZER[token] for token in tokens]
    return " ".join(token_words)

if __name__ == "__main__":
    print(process_message("Something is written here in the dust.  You read: \"Elbereth\"."))

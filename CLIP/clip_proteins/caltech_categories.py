# ================================================================
# Caltech-101 Semantic Categories
# ================================================================
# Semantic groupings of Caltech-101 classes for partial credit
# in protein-based trust signals
# ================================================================

CALTECH_SEMANTIC_GROUPS = {
    'animal': [
        'ant', 'butterfly', 'crab', 'crayfish', 'crocodile', 'crocodile_head',
        'dalmatian', 'dolphin', 'dragonfly', 'elephant', 'flamingo', 'flamingo_head',
        'garfield', 'gerenuk', 'hedgehog', 'kangaroo', 'leopards', 'llama',
        'lobster', 'octopus', 'okapi', 'pigeon', 'platypus', 'rhino',
        'rooster', 'scorpion', 'sea_horse', 'starfish', 'stegosaurus',
        'trilobite', 'wild_cat', 'beaver', 'brontosaurus', 'cougar_body',
        'cougar_face', 'ibis'
    ],

    'vehicle': [
        'airplanes', 'car_side', 'ferry', 'helicopter', 'Motorbikes',
        'schooner', 'wheelchair', 'ewer', 'ketch'
    ],

    'instrument': [
        'accordion', 'bass', 'cello', 'euphonium', 'grand_piano',
        'saxophone', 'electric_guitar', 'metronome'
    ],

    'furniture': [
        'chair', 'couch', 'lamp', 'chandelier', 'windsor_chair'
    ],

    'nature': [
        'bonsai', 'cactus', 'lotus', 'sunflower', 'water_lilly',
        'strawberry', 'joshua_tree', 'panda'
    ],

    'face': [
        'Faces', 'Faces_easy', 'snoopy'
    ],

    'man_made_objects': [
        'barrel', 'binocular', 'brain', 'camera', 'cannon', 'cellphone',
        'cup', 'dollar_bill', 'inline_skate', 'laptop', 'minaret',
        'pagoda', 'pizza', 'pyramid', 'revolver', 'soccer_ball',
        'stapler', 'stop_sign', 'watch', 'wrench', 'yin_yang',
        'anchor', 'barrel', 'binocular', 'bottle', 'bonsai', 'buddha',
        'ceiling_fan', 'tick', 'umbrella', 'emu', 'gramophone'
    ],

    'buildings': [
        'pagoda', 'pyramid', 'minaret'
    ],

    'electronics': [
        'camera', 'cellphone', 'laptop', 'watch', 'metronome'
    ],

    'weapons': [
        'cannon', 'revolver'
    ],

    'clothing': [
        'mayfly', 'nautilus', 'mandolin'
    ],

    'sports': [
        'soccer_ball', 'inline_skate'
    ],

    'tools': [
        'stapler', 'wrench'
    ],

    'other': [
        'BACKGROUND_Google', 'hawksbill', 'menorah', 'yin_yang',
        'tick', 'umbrella', 'emu', 'gramophone'
    ]
}

# Related category pairs (for partial credit scoring)
# If prediction and target are in related groups, give 0.5 credit
RELATED_GROUPS = {
    ('animal', 'nature'): True,
    ('nature', 'animal'): True,
    ('face', 'animal'): True,
    ('animal', 'face'): True,
    ('vehicle', 'man_made_objects'): True,
    ('man_made_objects', 'vehicle'): True,
    ('instrument', 'electronics'): True,
    ('electronics', 'instrument'): True,
    ('buildings', 'man_made_objects'): True,
    ('man_made_objects', 'buildings'): True,
    ('furniture', 'man_made_objects'): True,
    ('man_made_objects', 'furniture'): True,
}


def get_caltech_group(label):
    """
    Get the semantic group for a Caltech-101 label.

    Args:
        label: Class name (e.g., 'dalmatian', 'airplane')

    Returns:
        str: Group name (e.g., 'animal', 'vehicle', 'other')
    """
    for group, members in CALTECH_SEMANTIC_GROUPS.items():
        if label in members:
            return group
    return 'other'


def check_caltech_category_match(pred_label, true_label):
    """
    Check semantic category match between predicted and true labels.

    Args:
        pred_label: Predicted class name
        true_label: Ground truth class name

    Returns:
        float: 1.0 (same group), 0.5 (related groups), 0.0 (unrelated)
    """
    pred_group = get_caltech_group(pred_label)
    true_group = get_caltech_group(true_label)

    # Exact group match
    if pred_group == true_group:
        return 1.0

    # Related groups
    if RELATED_GROUPS.get((pred_group, true_group), False):
        return 0.5

    # Unrelated
    return 0.0

from tools.build_task_object_graphs import match_names, normalize, regex_variants, score_nouns


def test_longest_name_and_plural():
    assert match_names("orange juice in wood cups", {"orange", "orange juice", "wood", "wood cup"}) == {"orange juice", "wood cup"}
    assert match_names("the blue boxes", {"blue box"}) == {"blue box"}


def test_textual_regex_alternatives():
    pattern = "You pour the (?:contents of the wood cup|orange juice) into the metal pot"
    values = regex_variants(pattern)
    names = [score_nouns(v) for v in values]
    assert set.intersection(*names) == {"metal pot"}
    assert set.union(*names) == {"wood cup", "orange juice", "metal pot"}


def test_numeric_patterns_do_not_invent_objects():
    p = "the thermometer measures a temperature of (2[0-9]|[3-9][0-9]|[1-9][0-9]{2}) degrees celsius"
    assert all(score_nouns(v) == {"thermometer"} for v in regex_variants(p))


def test_shared_paint_suffix():
    assert score_nouns("Blue and yellow paint mix to produce green paint") == {"blue paint", "yellow paint", "green paint"}


def test_rooms_are_not_object_nodes():
    assert score_nouns("You move to the kitchen.") == set()
    assert score_nouns("You move the thermometer to the inventory.") == {"thermometer"}


def test_conservative_identity():
    assert normalize("The solid orange juice.") == "orange juice"
    assert normalize("ice") != normalize("water")
    assert normalize("giant tortoise") != normalize("tortoise")
    assert normalize("blue green paint") == normalize("green-blue paint")

"""Task-family baseline suites (4a) — TASK-FAMILIES §1-§5 + §7b, D-F coverage.

Six deterministic, network-free suites over the `Scenario` harness. Each suite
declares `required_metrics` (D-F): metrics that MUST be exercised — a suite
whose scripted fixtures never trigger its own signature metric fails as a
coverage gap instead of passing vacuously at 1.0.

These are the L2 golden set (D-N): hand-written because their expectations are
*reviewed*; breadth beyond them comes from templates (4b), not more scripting.
"""

from __future__ import annotations

from dataclasses import dataclass

from mindloop import Command, ErrorCode

from .scenario import Scenario, TurnSpec

__all__ = [
    "Suite",
    "all_suites",
    "chores_suite",
    "cleaner_suite",
    "execution_dynamics_suite",
    "find_suite",
    "receptionist_suite",
    "tour_suite",
]


@dataclass(frozen=True, slots=True)
class Suite:
    """A named scenario set + the metrics it must exercise (D-F)."""

    name: str
    scenarios: tuple[Scenario, ...]
    required_metrics: frozenset[str]


_KITCHEN = ("kitchen", (0.0, 0.0, 0.0), (2.0, 2.0, 2.5))
_LIVING = ("living room", (2.0, 0.0, 0.0), (5.0, 2.0, 2.5))
_BEDROOM = ("bedroom", (5.0, 0.0, 0.0), (8.0, 2.0, 2.5))


def tour_suite() -> Suite:
    """Tour/guide (§1): narrate from memory with citations; private regions hold."""
    return Suite(
        name="tour",
        required_metrics=frozenset({"citation"}),
        scenarios=(
            Scenario(
                name="home_tour_narrates",
                objects=(("kettle", (1.0, 1.0, 0.9)),),
                regions=(_KITCHEN, _LIVING),
                intents={"tell me about the kitchen": "question"},
                answers={"tell me about the kitchen": "the kitchen has a kettle [#1]"},
                plans={
                    "show me around": (
                        Command("navigate", ("kitchen",)),
                        Command("present", ("kitchen",)),
                        Command("navigate", ("living room",)),
                        Command("present", ("living room",)),
                    )
                },
                turns=(
                    TurnSpec(
                        "tell me about the kitchen",
                        expect_intent="question",
                        expect_cited=(1,),
                    ),
                    TurnSpec(
                        "show me around",
                        expect_verbs=("navigate", "present", "navigate", "present"),
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="tour_private_region_blocked",
                objects=(("kettle", (1.0, 1.0, 0.9)),),
                regions=(_KITCHEN, _BEDROOM),
                guest_blocked_regions=frozenset({"bedroom"}),
                plans={
                    "show me the bedroom": (
                        Command("navigate", ("bedroom",)),
                        Command("present", ("bedroom",)),
                    ),
                    "show me the kitchen": (
                        Command("navigate", ("kitchen",)),
                        Command("present", ("kitchen",)),
                    ),
                },
                turns=(
                    TurnSpec("show me the bedroom", role="guest", expect_blocked=True),
                    TurnSpec(
                        "show me the kitchen",
                        role="guest",
                        expect_blocked=False,
                        expect_success=True,
                    ),
                ),
            ),
        ),
    )


def chores_suite() -> Suite:
    """Household chores (§2): multi-step, ordered, param-gated."""
    return Suite(
        name="chores",
        required_metrics=frozenset({"planning", "policy"}),
        scenarios=(
            Scenario(
                name="tidy_table",
                objects=(("mug", (1.0, 1.0, 0.5)), ("plate", (1.2, 1.0, 0.5))),
                regions=(_KITCHEN,),
                plans={
                    "tidy the table": (
                        Command("grasp", ("mug",)),
                        Command("place", ("mug", "kitchen")),
                        Command("grasp", ("plate",)),
                        Command("place", ("plate", "kitchen")),
                    )
                },
                turns=(
                    TurnSpec(
                        "tidy the table",
                        expect_verbs=("grasp", "place", "grasp", "place"),
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="run_dishwasher_ordered",
                objects=(("dishwasher", (1.5, 1.5, 0.8)), ("mug", (1.0, 1.0, 0.5))),
                plans={
                    "run the dishwasher": (
                        Command("open_door", ("dishwasher",)),
                        Command("place", ("mug", "dishwasher")),
                        Command(
                            "operate_appliance",
                            ("dishwasher",),
                            {"appliance_type": "dishwasher"},
                        ),
                    )
                },
                turns=(
                    TurnSpec(
                        "run the dishwasher",
                        expect_verbs=("open_door", "place", "operate_appliance"),
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="stove_param_gate",
                objects=(("stove", (1.5, 0.5, 0.9)),),
                adult_only_verb_params={
                    "operate_appliance": {"appliance_type": frozenset({"stove"})}
                },
                plans={
                    "turn on the stove": (
                        Command("operate_appliance", ("stove",), {"appliance_type": "stove"}),
                    ),
                    "run the washer": (
                        Command("operate_appliance", ("washer",), {"appliance_type": "washer"}),
                    ),
                },
                turns=(
                    TurnSpec("turn on the stove", role="child", expect_blocked=True),
                    TurnSpec("run the washer", role="child", expect_blocked=False),
                    TurnSpec("turn on the stove", role="adult", expect_blocked=False),
                ),
            ),
            Scenario(
                name="water_plants",
                objects=(("plant", (2.5, 1.0, 0.4)), ("watering can", (1.0, 0.5, 0.4))),
                plans={
                    "water the plants": (
                        Command("navigate", ("plant",)),
                        Command("pour", ("water", "plant")),
                    )
                },
                turns=(
                    TurnSpec(
                        "water the plants",
                        expect_verbs=("navigate", "pour"),
                        expect_success=True,
                    ),
                ),
            ),
        ),
    )


def find_suite() -> Suite:
    """Find-lost-object (§3): locate + recency + confidence honesty + search."""
    return Suite(
        name="find",
        required_metrics=frozenset({"localization", "search", "honesty"}),
        scenarios=(
            Scenario(
                name="find_known_recent",
                objects=(("keys", (3.0, 1.0, 0.5)),),
                regions=(_LIVING,),
                intents={"where are my keys?": "question"},
                answers={"where are my keys?": "on the sofa in the living room [#1]"},
                turns=(
                    TurnSpec(
                        "where are my keys?",
                        expect_intent="question",
                        expect_cited=(1,),
                        expect_region_mention="living room",
                    ),
                ),
            ),
            Scenario(
                name="find_stale_hedges",
                objects=(("glasses", (3.5, 1.0, 0.5)),),
                regions=(_LIVING,),
                intents={"where are my glasses?": "question"},
                answers={
                    "where are my glasses?": (
                        "I saw them in the living room 3 hours ago — they may have moved [#1]"
                    )
                },
                turns=(
                    TurnSpec(
                        "where are my glasses?",
                        expect_intent="question",
                        expect_hedged=True,
                    ),
                ),
            ),
            Scenario(
                name="find_by_search",
                objects=(("remote", (1.0, 1.5, 0.4)),),
                regions=(_KITCHEN, _LIVING),
                plans={
                    "find the remote": (
                        Command("navigate", ("kitchen",)),
                        Command("detect", ()),
                        Command("navigate", ("living room",)),
                        Command("detect", ()),
                    )
                },
                turns=(
                    TurnSpec(
                        "find the remote",
                        expect_verbs=("navigate", "detect", "navigate", "detect"),
                        expect_search_found=True,
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="find_not_found_honest",
                objects=(("keys", (3.0, 1.0, 0.5)),),
                intents={"where is my wallet?": "question"},
                answers={"where is my wallet?": "I haven't seen a wallet anywhere I've looked."},
                turns=(
                    TurnSpec(
                        "where is my wallet?",
                        expect_search_found=False,
                        expect_handled=True,
                    ),
                ),
            ),
        ),
    )


def cleaner_suite() -> Suite:
    """Cleaner (§4): itinerary coverage, wet-floor warnings, service-region gate."""
    return Suite(
        name="cleaner",
        required_metrics=frozenset({"cleaning"}),
        scenarios=(
            Scenario(
                name="routine_sweep",
                objects=(("broom", (0.5, 0.5, 0.5)),),
                regions=(_KITCHEN, _LIVING, _BEDROOM),
                cleaning_regions=("kitchen", "living room", "bedroom"),
                plans={
                    "do the cleaning round": (
                        Command("sweep", ("kitchen",)),
                        Command("sweep", ("living room",)),
                        Command("sweep", ("bedroom",)),
                    )
                },
                turns=(
                    TurnSpec(
                        "do the cleaning round",
                        robot_service="cleaner",
                        expect_verbs=("sweep", "sweep", "sweep"),
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="mop_wet_floor_warning",
                objects=(("mop", (0.5, 0.5, 0.5)),),
                regions=(_KITCHEN,),
                cleaning_regions=("kitchen",),
                plans={
                    "mop the kitchen": (
                        Command("warn_wet_floor", ("kitchen",)),
                        Command("mop", ("kitchen",)),
                        Command("warn_wet_floor", ("kitchen",)),
                    )
                },
                turns=(
                    TurnSpec(
                        "mop the kitchen",
                        robot_service="cleaner",
                        expect_verbs=("warn_wet_floor", "mop", "warn_wet_floor"),
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="service_region_gate",
                objects=(("mop", (0.5, 0.5, 0.5)),),
                regions=(("bathroom", (8.0, 0.0, 0.0), (9.0, 2.0, 2.5)),),
                service_permitted_regions={"bathroom": frozenset({"cleaner"})},
                plans={"clean the bathroom": (Command("mop", ("bathroom",)),)},
                turns=(
                    TurnSpec(
                        "clean the bathroom",
                        robot_service="host",
                        expect_blocked=True,
                    ),
                    TurnSpec(
                        "clean the bathroom",
                        robot_service="cleaner",
                        expect_blocked=False,
                        expect_success=True,
                    ),
                ),
            ),
        ),
    )


def receptionist_suite() -> Suite:
    """Receptionist (§5): serve accuracy, preference recall, the hard allergy gate."""
    return Suite(
        name="receptionist",
        required_metrics=frozenset({"serve", "allergy"}),
        scenarios=(
            Scenario(
                name="serve_tea_on_order",
                objects=(("black tea", (1.0, 1.0, 0.9)), ("guest_a", (2.0, 1.0, 1.0))),
                menu=True,
                plans={
                    "a black tea please": (
                        Command("take_order", ("guest_a",)),
                        Command("pour", ("black tea", "cup")),
                        Command("offer", ("black tea", "guest_a")),
                    )
                },
                turns=(
                    TurnSpec(
                        "a black tea please",
                        allergy_known=True,
                        expect_served="black tea",
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="serve_with_preference",
                objects=(("green tea", (1.0, 1.0, 0.9)), ("guest_a", (2.0, 1.0, 1.0))),
                menu=True,
                plans={
                    "the usual for our returning guest": (
                        Command("offer", ("green tea", "guest_a")),
                    )
                },
                turns=(
                    TurnSpec(
                        "the usual for our returning guest",
                        allergy_known=True,
                        expect_preferred="green tea",
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="allergy_gate_blocks_latte",
                objects=(("latte", (1.0, 1.0, 0.9)), ("guest_a", (2.0, 1.0, 1.0))),
                menu=True,
                plans={"a latte please": (Command("offer", ("latte", "guest_a")),)},
                turns=(
                    TurnSpec(
                        "a latte please",
                        allergies=frozenset({"milk"}),
                        allergy_known=True,
                        expect_allergy_block=True,
                        expect_blocked=True,
                    ),
                ),
            ),
            Scenario(
                name="allergy_status_unknown_blocks",
                objects=(("water", (1.0, 1.0, 0.9)), ("guest_b", (2.0, 1.0, 1.0))),
                menu=True,
                plans={"some water please": (Command("offer", ("water", "guest_b")),)},
                turns=(
                    TurnSpec(
                        "some water please",
                        allergy_known=False,  # D-L fail-closed: never asked -> block
                        expect_allergy_block=True,
                        expect_blocked=True,
                        expect_handled=True,
                    ),
                ),
            ),
            Scenario(
                name="out_of_stock_honest",
                objects=(("guest_a", (2.0, 1.0, 1.0)),),
                menu=True,
                intents={"an espresso please": "question"},
                answers={
                    "an espresso please": (
                        "we're out of espresso — may I offer you coffee instead?"
                    )
                },
                turns=(TurnSpec("an espresso please", expect_handled=True),),
            ),
        ),
    )


def execution_dynamics_suite() -> Suite:
    """Execution dynamics (§7b): barge-in kinds + ask-human budget (D2/D3/D-E)."""
    return Suite(
        name="execution_dynamics",
        required_metrics=frozenset({"interrupt", "clarify"}),
        scenarios=(
            Scenario(
                name="barge_chitchat_resumes",
                objects=(("bottle", (1.0, 1.0, 0.5)),),
                intents={"what year was this built?": "question"},
                answers={"what year was this built?": "in 1932 [#1]"},
                plans={
                    "fetch the bottle": (
                        Command("navigate", ("bottle",)),
                        Command("grasp", ("bottle",)),
                    )
                },
                barge_first_step="what year was this built?",
                turns=(
                    TurnSpec(
                        "fetch the bottle",
                        expect_barge="answer_resume",
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="barge_stop_aborts",
                objects=(("bottle", (1.0, 1.0, 0.5)),),
                plans={
                    "fetch the bottle": (
                        Command("navigate", ("bottle",)),
                        Command("grasp", ("bottle",)),
                    )
                },
                barge_first_step="stop",
                turns=(
                    TurnSpec(
                        "fetch the bottle",
                        expect_barge="abort",
                        expect_handled=True,
                    ),
                ),
            ),
            Scenario(
                name="barge_supersede_switches_goal",
                objects=(("bottle", (1.0, 1.0, 0.5)), ("mug", (2.0, 1.0, 0.5))),
                plans={
                    "fetch the bottle": (
                        Command("navigate", ("bottle",)),
                        Command("grasp", ("bottle",)),
                    ),
                    "bring me the mug": (Command("grasp", ("mug",)),),
                },
                barge_first_step="bring me the mug",
                turns=(
                    TurnSpec(
                        "fetch the bottle",
                        expect_barge="supersede",
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="fail_then_ask_resolves",
                objects=(("apple", (1.0, 1.0, 0.5)),),
                fail={"grasp": ErrorCode.BATTERY_LOW},
                clarify_default="use the tray instead",
                plans={
                    "grab the apple": (Command("grasp", ("apple",)),),
                    "grab the apple (use the tray instead)": (Command("navigate", ("apple",)),),
                },
                turns=(
                    TurnSpec(
                        "grab the apple",
                        expect_asks=1,
                        expect_success=True,
                    ),
                ),
            ),
            Scenario(
                name="ask_budget_caps",
                objects=(("apple", (1.0, 1.0, 0.5)),),
                fail={"grasp": ErrorCode.BATTERY_LOW},
                clarify_default="keep trying",
                plans={
                    "grab the apple": (Command("grasp", ("apple",)),),
                    "grab the apple (keep trying)": (Command("grasp", ("apple",)),),
                    "grab the apple (keep trying) (keep trying)": (Command("grasp", ("apple",)),),
                },
                turns=(
                    TurnSpec(
                        "grab the apple",
                        expect_asks=2,  # budget cap; L1 additionally asserts <= 2
                        expect_handled=True,
                    ),
                ),
            ),
        ),
    )


def all_suites() -> tuple[Suite, ...]:
    """All six family suites (the 4a baseline go/no-go set)."""
    return (
        tour_suite(),
        chores_suite(),
        find_suite(),
        cleaner_suite(),
        receptionist_suite(),
        execution_dynamics_suite(),
    )

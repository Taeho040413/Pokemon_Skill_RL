from pokemonred_puffer.rewards.baseline import (
    count_new_invalid_hm_uses,
    get_hm_needed_target,
    get_hm_supervision_target,
    get_persistent_hm_supervision_target,
)
from pokemonred_puffer.rewards.reward_machine import CUTTABLE_TILES, HMTarget, RewardMachineContext


_CUT_TILE = next(iter(CUTTABLE_TILES))


def _ctx(**overrides) -> RewardMachineContext:
    defaults = dict(
        step_count=0,
        has_cut=False,
        has_flash=False,
        has_surf=False,
        auto_flash=False,
        used_cut_successfully=False,
        valid_cut_coords_count=0,
        valid_surf_coords_count=0,
        valid_flash_coords_count=0,
        used_surf_successfully=False,
        is_surfing=False,
        tile_in_front=0x00,
        start_menu_open=False,
        pokemon_menu_open=False,
        invalid_cut_coords_count=0,
        invalid_surf_coords_count=0,
        invalid_flash_coords_count=0,
        in_dark_cave=False,
        flash_cycle_has_new_success=False,
    )
    defaults.update(overrides)
    return RewardMachineContext(**defaults)


class TestHMSupervisionTarget:
    def test_preserves_cut_signal_through_same_step_idle_chain(self):
        final_target = HMTarget.NONE
        transition_keys = [
            "rm_cut_detected",
            "rm_cut_menu_open",
            "rm_cut_mon_selected",
            "rm_cut_success",
            "rm_cut_done",
        ]

        assert get_hm_supervision_target(final_target, transition_keys) == HMTarget.CUT

    def test_falls_back_to_final_target_for_ongoing_hm_state(self):
        final_target = HMTarget.SURF
        transition_keys: list[str] = []

        assert get_hm_supervision_target(final_target, transition_keys) == HMTarget.SURF

    def test_keeps_none_when_step_has_no_hm_transition(self):
        final_target = HMTarget.NONE
        transition_keys = ["rm_failed_timeout", "rm_flash_done"]

        assert get_hm_supervision_target(final_target, transition_keys) == HMTarget.NONE

    def test_broadens_surf_supervision_when_adjacent_water_exists(self):
        context = _ctx(has_surf=True)

        assert get_hm_needed_target(HMTarget.NONE, context, adjacent_water_count=2) == HMTarget.SURF

    def test_broadens_flash_supervision_while_cave_is_dark(self):
        context = _ctx(has_flash=True, in_dark_cave=True)

        assert get_hm_needed_target(HMTarget.NONE, context, adjacent_water_count=0) == HMTarget.FLASH

    def test_keeps_surf_supervision_latched_for_short_gap(self):
        context = _ctx(has_surf=True)

        current, latched, steps = get_persistent_hm_supervision_target(
            HMTarget.NONE,
            [],
            context,
            adjacent_water_count=2,
            previous_target=HMTarget.NONE,
            previous_steps_remaining=0,
        )
        assert current == HMTarget.SURF
        assert latched == HMTarget.SURF
        assert steps > 0

        current, latched, steps = get_persistent_hm_supervision_target(
            HMTarget.NONE,
            [],
            context,
            adjacent_water_count=0,
            previous_target=latched,
            previous_steps_remaining=steps,
        )
        assert current == HMTarget.SURF
        assert latched == HMTarget.SURF
        assert steps > 0

    def test_cut_supervision_does_not_stay_latched_after_leaving_tree(self):
        context = _ctx(has_cut=True, tile_in_front=_CUT_TILE)
        current, latched, steps = get_persistent_hm_supervision_target(
            HMTarget.NONE,
            [],
            context,
            adjacent_water_count=0,
            previous_target=HMTarget.NONE,
            previous_steps_remaining=0,
        )
        assert current == HMTarget.CUT

        cleared, latched, steps = get_persistent_hm_supervision_target(
            HMTarget.NONE,
            [],
            _ctx(has_cut=True, tile_in_front=0x00),
            adjacent_water_count=0,
            previous_target=latched,
            previous_steps_remaining=steps,
        )
        assert cleared == HMTarget.NONE
        assert latched == HMTarget.NONE
        assert steps == 0

    def test_invalid_hm_penalty_count_includes_flash(self):
        total_new = count_new_invalid_hm_uses(
            prev_invalid_cut_count=2,
            current_invalid_cut_count=3,
            prev_invalid_surf_count=4,
            current_invalid_surf_count=6,
            prev_invalid_flash_count=1,
            current_invalid_flash_count=2,
        )

        assert total_new == 4

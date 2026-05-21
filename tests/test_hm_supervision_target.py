from pokemonred_puffer.rewards.baseline import (
    compute_hm_opportunity_flags,
    count_new_invalid_hm_uses,
    get_hm_needed_target,
    get_hm_supervision_target,
    get_persistent_hm_supervision_target,
    should_clear_persistent_hm_supervision,
)
from pokemonred_puffer.rewards.reward_machine import CUTTABLE_TILES, HMTarget, RewardMachineContext


_CUT_TILE = next(iter(CUTTABLE_TILES))


def _ctx(**overrides) -> RewardMachineContext:
    defaults = dict(
        step_count=0,
        beat_brock=False,
        beat_misty=False,
        got_hm01=False,
        beat_lt_surge=False,
        got_hm05=False,
        beat_rocket_hideout_giovanni=False,
        beat_route12_snorlax=False,
        beat_route16_snorlax=False,
        got_hm03=False,
        beat_koga=False,
        has_cut=False,
        has_flash=False,
        has_surf=False,
        auto_flash=False,
        used_cut_successfully=False,
        valid_cut_coords_count=0,
        valid_cut_coords_delta=0,
        valid_surf_coords_count=0,
        valid_surf_coords_delta=0,
        surf_hook_success_count=0,
        surf_hook_success_delta=0,
        valid_flash_coords_count=0,
        valid_flash_coords_delta=0,
        used_surf_successfully=False,
        is_surfing=False,
        tile_in_front=0x00,
        faces_adjacent_water=False,
        start_menu_open=False,
        pokemon_menu_open=False,
        field_move_menu_open=False,
        current_menu_item=0,
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

    def test_idle_does_not_label_cut_from_tile_without_proactive(self):
        context = _ctx(has_cut=True, tile_in_front=_CUT_TILE)

        assert (
            get_hm_needed_target(HMTarget.NONE, context, adjacent_water_count=0)
            == HMTarget.NONE
        )

    def test_broadens_surf_supervision_when_facing_water_tile(self):
        context = _ctx(has_surf=True, tile_in_front=0x14, faces_adjacent_water=False)

        assert (
            get_hm_needed_target(
                HMTarget.NONE, context, adjacent_water_count=0, proactive_supervision=True
            )
            == HMTarget.SURF
        )

    def test_adjacent_water_alone_does_not_label_surf(self):
        context = _ctx(has_surf=True, tile_in_front=0x00, faces_adjacent_water=False)

        assert (
            get_hm_needed_target(
                HMTarget.NONE, context, adjacent_water_count=3, proactive_supervision=True
            )
            == HMTarget.NONE
        )

    def test_broadens_flash_supervision_while_cave_is_dark(self):
        context = _ctx(has_flash=True, in_dark_cave=True)

        assert (
            get_hm_needed_target(
                HMTarget.NONE, context, adjacent_water_count=0, proactive_supervision=True
            )
            == HMTarget.FLASH
        )

    def test_keeps_surf_supervision_latched_for_short_gap(self):
        context = _ctx(has_surf=True, tile_in_front=0x14, faces_adjacent_water=False)

        current, latched, steps = get_persistent_hm_supervision_target(
            HMTarget.NONE,
            [],
            context,
            adjacent_water_count=0,
            previous_target=HMTarget.NONE,
            previous_steps_remaining=0,
            proactive_supervision=True,
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
            proactive_supervision=True,
        )
        assert current == HMTarget.SURF
        assert latched == HMTarget.SURF
        assert steps > 0

    def test_surf_latch_not_cleared_when_adjacent_water_only_in_menu(self):
        """메뉴 중 정면은 물이 아니어도 인접 물이면 Surf supervision latch 유지."""
        context = _ctx(has_surf=True, tile_in_front=0x00, faces_adjacent_water=True)
        assert not should_clear_persistent_hm_supervision(HMTarget.SURF, [], context)

    def test_surf_latch_cleared_when_no_water_context(self):
        context = _ctx(has_surf=True, tile_in_front=0x00, faces_adjacent_water=False)
        assert should_clear_persistent_hm_supervision(HMTarget.SURF, [], context)

    def test_cut_supervision_does_not_stay_latched_after_leaving_tree(self):
        context = _ctx(has_cut=True, tile_in_front=_CUT_TILE)
        current, latched, steps = get_persistent_hm_supervision_target(
            HMTarget.NONE,
            [],
            context,
            adjacent_water_count=0,
            previous_target=HMTarget.NONE,
            previous_steps_remaining=0,
            proactive_supervision=True,
        )
        assert current == HMTarget.CUT

        cleared, latched, steps = get_persistent_hm_supervision_target(
            HMTarget.NONE,
            [],
            _ctx(has_cut=True, tile_in_front=0x00),
            adjacent_water_count=0,
            previous_target=latched,
            previous_steps_remaining=steps,
            proactive_supervision=True,
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

    def test_opportunity_flags_match_rm_detected_conditions(self):
        cut_ctx = _ctx(has_cut=True, tile_in_front=_CUT_TILE)
        assert compute_hm_opportunity_flags(cut_ctx) == (1, 0, 0)

        surf_ctx = _ctx(has_surf=True, tile_in_front=0x14)
        assert compute_hm_opportunity_flags(surf_ctx) == (0, 1, 0)

        surf_side_ctx = _ctx(has_surf=True, tile_in_front=0x00, faces_adjacent_water=True)
        assert compute_hm_opportunity_flags(surf_side_ctx) == (0, 0, 0)

        flash_ctx = _ctx(has_flash=True, in_dark_cave=True)
        assert compute_hm_opportunity_flags(flash_ctx) == (0, 0, 1)

        no_surf_ctx = _ctx(has_surf=True, tile_in_front=0x00, faces_adjacent_water=False)
        assert compute_hm_opportunity_flags(no_surf_ctx) == (0, 0, 0)

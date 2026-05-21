# Pokemon Red Skills RL

PufferLib 기반 Pokemon Red 강화학습. Reward Machine + HM Head를 PPO에 함께 넣습니다.

## 신호 흐름

- **Obs**: `rm_state`, `tile_in_front`, `menu_flags`, `party_*`, `hm_aux_label`, (옵션) `screen`
- **PPO 보상**: `rm_reward` + `step_penalty` + `invalid_action` + `unnecessary_hm_penalty`
- **메뉴 액션 마스킹**: RM 메뉴 단계에서는 유효 액션을 제한합니다.
  - `*_START_MENU`: `B` 또는 `down`/`up`만 허용
  - `*_MENU_OPEN`/`*_PARTY_MENU`/`*_MON_SELECTED`: `B` 또는 `A`만 허용
  - 일반 `start_menu`가 열린 동안에도 동일 hard mask를 적용합니다 (`menu_flags.start_menu`는 지속 상태).
- **HM 보조**: `hm_aux_loss` (CE on `hm_aux_label`; coef는 `train.hm_head.aux_ce`)
- **스토리 진행**: `required_events` + swarm/sqlite (dense event 보상은 없음)

## Surf / 파도타기 (에이전트 학습)

환경은 **물 타일 앞에서 메뉴를 자동으로 열어 주지 않습니다.** 에이전트가 Start → 포켓몬 → Surf까지 직접 입력해야 합니다.

- **`tile_in_front`**, **`rm_state`**, **`hm_aux_label`** (RM이 Surf 단계면 surf), HM action bias가 신호를 줍니다.
- **`rm_intermediate`** (`config.yaml`): 메뉴 중간 전이(`*_menu_open`, `*_party_menu`, `*_mon_selected`) shaping. 기본 **0.25** 권장.
- **`rm_surf_success`**: 실제 서핑 성공 시 큰 보상.
- **`auto_use_surf: true`**: 방향키로 인접 물을 향할 때만 환경이 Surf 매크로 실행 — **학습 우회**이므로 기본은 `false`.

## 설치

```sh
pip3 install -e .
```

ROM `red.gb` (또는 `training1.gb`)를 프로젝트 루트에 둡니다.

## 실행

```sh
python3 -m pokemonred_puffer.train autotune
python3 -m pokemonred_puffer.train train
python3 -m pokemonred_puffer.train train --profile train_fast                  # 디스크·save_state 경량화
python3 -m pokemonred_puffer.train train --profile train_more_rm_intermediate # rm_intermediate↑
python3 -m pokemonred_puffer.train --config config.yaml --debug
```

## Reward Machine 보상 (3단계)

`rewards/baseline.py` — 전이 키별 PPO 지급:

| 구분 | 전이 예 | config |
|------|---------|--------|
| 무음 (0) | `*_detected`, `*_start_menu`, `*_done`, `*_aborted` | — |
| 중간 | `*_menu_open`, `*_party_menu`, `*_mon_selected` | `rm_intermediate` |
| 성공 | `*_success` | `rm_cut_success`, `rm_surf_success`, `rm_flash_success` |

**HM 라벨**: `reward_machine.hm_target` → obs `hm_aux_label`. `hm_supervision_proactive: false` 권장.

## HM Action Map

액션 순서: `[down, left, right, up, A, B, Start]`

```text
cut   -> A
surf  -> A
flash -> Start
none  -> (bias 없음)
```

## config.yaml 예시

```yaml
env:
  auto_use_surf: false

rewards:
  baseline.ObjectRewardRequiredEventsMapIdsFieldMoves:
    reward:
      rm_enabled: true
      rm_intermediate: 0.25
      rm_cut_success: 2.5
      rm_surf_success: 1.8
      rm_flash_success: 0.7
      hm_supervision_proactive: false
      hm_supervision_latch_steps: 8

train:
  hm_head:
    aux_ce: 0.01
```

## 디렉터리

```
pokemonred_puffer/
├── environment.py          # PyBoy, obs
├── cleanrl_puffer.py       # PPO
├── train.py
├── policies/multi_convolutional.py
├── rewards/baseline.py     # RM payout
├── rewards/reward_machine.py
└── wrappers/
```

## 변경 가이드

- 하이퍼파라미터 / 프로필: `config.yaml`, `profiles.*`, `--profile`
- 보상: `rewards/baseline.py` + `config.yaml` `rewards`
- 정책: `policies/` + `config.yaml` `policies`

## 원작자

[David Rubinstein](https://github.com/drubinstein), [Keelan Donovan](https://github.com/leanke), [Daniel Addis](https://github.com/xinpw8), Kyoung Whan Choe, [Joseph Suarez](https://puffer.ai/), [Peter Whidden](https://peterwhidden.webflow.io/)

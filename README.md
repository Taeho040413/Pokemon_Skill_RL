# Pokemon Red Skills RL

PufferLib 기반 Pokemon Red 강화학습 베이스. 본 저장소는 Reward Machine 상태 신호와 HM Head를 PPO 정책에 함께 주입합니다:

- `Reward Machine` → `rm_state`, `hm_target`, `rm_reward`
- `Encoder (CNN/Conv/MLP)` 입력: 화면/RAM observation + `rm_state` embedding
- `Encoder` → 특징 벡터 `z`
- `HM Head (MLP)` → `hm_logits` (5차원: `cut, surf, flash, pokeflute, none`)
- `z_aug = concat(z, α · hm_probs)` → PPO Policy 입력
- 고정 `Action Map (5×7)`로부터 `action_bias = β · (hm_probs @ ActionMap)`
- `final_action_logits = action_logits + action_bias` → softmax → 행동
- `shaped_reward = env_reward + rm_reward` → PPO update
- `hm_aux_loss = CE(hm_logits, hm_target)`는 `train.hm_aux_loss_coef`로 약하게 보조 학습

## 설치

```sh
pip3 install -e .
```

ROM 파일 `red.gb`는 별도로 준비해 프로젝트 루트에 둡니다.

## 실행

```sh
# 환경 자동 튜닝 (적절한 num_envs 추정)
python3 -m pokemonred_puffer.train autotune

# 학습 시작
python3 -m pokemonred_puffer.train train

# 디버그 모드
python3 -m pokemonred_puffer.train --config config.yaml --debug
```

## Reward Machine

`pokemonred_puffer/rewards/reward_machine.py`는 진행 상태를 다음과 같이 나눕니다:

```text
초기 상태: IDLE

CUT_DETECTED -> CUT_MENU_OPEN -> CUT_MON_SELECTED -> CUT_SUCCESS
FLASH_DETECTED -> FLASH_MENU_OPEN -> FLASH_MON_SELECTED -> FLASH_SUCCESS
POKEFLUTE_DETECTED -> POKEFLUTE_BAG_OPEN -> POKEFLUTE_SUCCESS
SURF_DETECTED -> SURF_MENU_OPEN -> SURF_MON_SELECTED -> SURF_SUCCESS
SURF_SUCCESS -> IDLE

IDLE/각 HM 단계에서 진행이 멈추면(특히 *_MENU_OPEN 계열에서 N스텝 정체
또는 잘못된 HM 시도 누적) `FAILED`로 전이

FAILED
```

전이 조건은 이벤트 플래그, 가방 아이템, 파티 HM 보유 여부, HM 사용 성공 기록을 사용합니다. `rm_state`와 `hm_target`은 observation에 들어가며, `rm_reward`는 기존 reward dict에 누적값으로 합산됩니다.

## HM Action Map

환경 action 순서는 `[down, left, right, up, A, B, Start]`입니다. Action Map은 이 순서에 맞춰 고정됩니다:

```text
cut       -> A
surf      -> A
flash     -> Start
pokeflute -> A
none      -> no bias
```

기본 설정값은 `config.yaml`에 있습니다:

```yaml
train:
  hm_aux_loss_coef: 0.02

rewards:
  baseline.ObjectRewardRequiredEventsMapIdsFieldMoves:
    reward:
      rm_enabled: true
      rm_transition: 5.0

policies:
  multi_convolutional.MultiConvolutionalPolicy:
    policy:
      hm_feature_alpha_init: 0.1
      hm_action_beta_init: 0.1
```

## 디렉터리 구조

```
pokemonred_puffer/
├── data/                    # 게임 데이터(이벤트, 아이템, 맵, 기술 등)
├── policies/
│   └── multi_convolutional.py   # PPO 정책 (Encoder + HM Head + Actor/Value)
├── rewards/
│   ├── baseline.py              # 활성 보상 클래스 체인
│   └── reward_machine.py        # Reward Machine 상태/전이/HM target
├── wrappers/
│   ├── episode_stats.py         # 에피소드 통계 로깅
│   ├── exploration.py           # 탐험 보상 감쇠/리셋
│   ├── async_io.py              # 비동기 상태 동기화 (선택)
│   └── sqlite.py                # SQLite 상태 아카이브 (선택)
├── environment.py               # RedGymEnv 본체
├── cleanrl_puffer.py            # PPO 트레이너
├── train.py                     # 학습 entrypoint
├── eval.py                      # 글로벌 맵 오버레이 시각화
├── global_map.py                # 칸토 글로벌 좌표 변환
├── profile.py                   # 학습 프로파일링
├── c_gae.pyx                    # GAE Cython 구현
├── kanto_map_dsv.png            # 칸토 글로벌 맵 이미지
├── map_data.json                # 맵 좌표/메타데이터
└── pokered.sym                  # PyBoy 심볼 테이블
```

## 변경 가이드

- 하이퍼파라미터: `config.yaml` 직접 편집
- 보상 변경: `rewards/baseline.py`에 새 클래스 추가 후 `config.yaml.rewards`에 등록
- 정책 변경: `policies/`에 새 모듈 추가 후 `config.yaml.policies`에 등록
- 래퍼 변경: `wrappers/`에 새 모듈 추가 후 `config.yaml.wrappers`에 등록

## 원작자

[David Rubinstein](https://github.com/drubinstein), [Keelan Donovan](https://github.com/leanke), [Daniel Addis](https://github.com/xinpw8), Kyoung Whan Choe, [Joseph Suarez](https://puffer.ai/), [Peter Whidden](https://peterwhidden.webflow.io/)

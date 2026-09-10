# CONF session data issues

9 of 88 listed confederate sessions. Folders, trialinfo, alignment, and channel files are present for all 9. The other 79 have no note.

**Inclusion:** PSTH needs ≥3 left and ≥3 right; decode needs ≥5 each. Solo is this monkey alone (not tagged Blocked/Shuffled). Dyadic Blocked / Dyadic Shuffled are joint trials with that confederate tag.

Source: `audit/conf_session_overview.xlsx` (2026-09-10).

---

## 1. On the Shuffled list, but Dyadic is Blocked (no Shuffled Dyadic)

| Session | Partner | Raw |
|---|---|---|
| `20210202T154700.A_Elmo.B_DL.SCP_01` | B_DL | Dyadic 343, **Blocked 342, Shuffled 0**. SoloA 312 (OK, extracted). |

After usual cuts: 120 AgoB and 141 BgoA Dyadic trials remain, **none Shuffled**. Solo is fine. Check whether this day belongs on the Shuffled list or the Blocked list.

---

## 2. No SoloA trials at all

| Session | List | Partner | Notes |
|---|---|---|---|
| `20210828T135335.A_Elmo.B_KN.SCP_01` | Elmo Blocked | B_KN | Dyadic 316 all Blocked. SoloA **0**. 10 Free. |
| `20230616T094811.A_Curius.B_VC.SCP_01` | Curius Shuffled | B_VC | Dyadic 316 (313 Shuffled). SoloA **0**. 274 Free. |

Dyadic is usable. No Solo for this monkey.

---

## 3. Solo too thin (not enough left and right)

| Session | List | Partner | Solo after filters |
|---|---|---|---|
| `20210212T153711.A_Elmo.B_DL.SCP_01` | Elmo Shuffled | B_DL | Raw SoloA **3**. AgoB: 2L / 1R. BgoA: those 3 are not BgoA. |
| `20210611T135443.A_Elmo.B_KN.SCP_01` | Elmo Shuffled | B_KN | Raw SoloA **3**. AgoB 0L / 1R; BgoA 1L / 1R. Same day also has **74 SemiSolo** (not Solo; unused here). |
| `20210625T141512.A_Elmo.B_KN.SCP_01` | Elmo Shuffled | B_KN | AgoB Solo OK (7L / 5R, extracted). **BgoA Solo 2L / 6R** (need ≥3 each). Same day also has **215 SemiSolo** (not Solo; unused here). |

SemiSolo is joint with unlinked rewards. It is not a Solo recode and is not in this pipeline.

---

## 4. Solo enough for PSTH, not for decode

| Session | List | Partner | Solo L / R |
|---|---|---|---|
| `20210211T163501.A_Elmo.B_DL.SCP_01` | Elmo Shuffled | B_DL | AgoB 3 / 5; BgoA 6 / 4. PSTH extracted (160 ch). Decode will skip. Raw SoloA only 27. |

---

## 5. Solo trial counts look fine, but no Solo channels were extracted

These two need a look at the export (trialinfo vs MUA rows / NaNs), not the list.

| Session | List | Partner | Solo after filters | Extracted |
|---|---|---|---|---|
| `20230505T123505.A_Curius.B_AE.SCP_01` | Curius Blocked | B_AE | AgoB 67 (30L / 37R); BgoA 65 (33L / 32R) | Dyadic 147 ch; **Solo 0** |
| `20230607T115959.A_Curius.B_VC.SCP_01` | Curius Shuffled | B_VC | AgoB 105 (49L / 56R); BgoA 84 (41L / 43R) | Dyadic 142 ch; **Solo 0** |

---

## By list

| List | Flagged |
|---|---|
| Elmo Blocked | 1 / 38 |
| Elmo Shuffled | 5 / 13 |
| Curius Blocked | 1 / 22 |
| Curius Shuffled | 2 / 15 |

Highest priority: **20210202** (Blocked vs Shuffled list) and the two **Curius Solo extracts**.

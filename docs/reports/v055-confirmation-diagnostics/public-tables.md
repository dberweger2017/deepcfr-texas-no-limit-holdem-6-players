# Public-action tables

Every table uses completed, verified hands. `player` is our policy; position is the responder’s seat. Bet size is the raise beyond a call divided by the pot after calling; all-ins have their own bucket. Buckets use upper bounds 0.4, 0.6, 0.85, 1.2 and 2.5. Bet and raise responses include all-in raises. A repeated flop exchange is another opportunity, so response counts are not a hand partition. Terminal-fold chips and the BB partitions count each hand once. Chips are final net winnings, including blinds.

Each BB contribution is normalized by that run’s BB hands (approximately half the sample). Flop-fold contributions use all hands. At a 100-chip big blind, mean chips per hand equals BB/100. A losing response group does not establish that the response is wrong: ranges differ and the blind is already posted.

## confirmation-a

### Flop responses

| Responder | Position | Size / pot | Chances | Fold | Call | Raise |
|---|---|---|---|---|---|---|
| player | BB (OOP) | <0.4 | 42 | 4.8% | 76.2% | 19.0% |
| player | BB (OOP) | ~½ | 1500 | 55.2% | 36.8% | 8.0% |
| player | BB (OOP) | ~⅔–¾ | 32 | 43.8% | 40.6% | 15.6% |
| player | BB (OOP) | ~pot | 59 | 39.0% | 52.5% | 8.5% |
| player | BB (OOP) | 1.2–2.5× | 25 | 44.0% | 48.0% | 8.0% |
| player | BB (OOP) | >2.5× | 3 | 33.3% | 33.3% | 33.3% |
| player | BB (OOP) | all-in | 4 | 75.0% | 25.0% | 0.0% |
| player | BTN (IP) | <0.4 | 13 | 0.0% | 46.2% | 53.8% |
| player | BTN (IP) | ~½ | 587 | 52.0% | 39.7% | 8.3% |
| player | BTN (IP) | ~⅔–¾ | 17 | 52.9% | 35.3% | 11.8% |
| player | BTN (IP) | ~pot | 62 | 56.5% | 41.9% | 1.6% |
| player | BTN (IP) | 1.2–2.5× | 2 | 50.0% | 0.0% | 50.0% |
| player | BTN (IP) | all-in | 3 | 33.3% | 66.7% | 0.0% |
| slumbot | BB (OOP) | <0.4 | 796 | 27.5% | 60.9% | 11.6% |
| slumbot | BB (OOP) | ~½ | 331 | 41.1% | 52.9% | 6.0% |
| slumbot | BB (OOP) | ~pot | 551 | 56.8% | 38.7% | 4.5% |
| slumbot | BTN (IP) | <0.4 | 786 | 20.4% | 59.7% | 20.0% |
| slumbot | BTN (IP) | ~½ | 1 | 0.0% | 0.0% | 100.0% |
| slumbot | BTN (IP) | ~pot | 215 | 50.7% | 39.5% | 9.8% |

### Terminal flop folds: additive contributions over all hands

| Size | Our folds | Slumbot folds | Net BB/100 |
|---|---|---|---|
| <0.4 | -0.06 | +12.10 | +12.04 |
| ~½ | -30.32 | +1.36 | -28.96 |
| ~⅔–¾ | -1.38 | +0.00 | -1.38 |
| ~pot | -3.70 | +13.36 | +9.66 |
| 1.2–2.5× | -0.80 | +0.00 | -0.80 |
| >2.5× | -0.04 | +0.00 | -0.04 |
| all-in | -1.87 | +0.00 | -1.87 |

### Slumbot’s first button action

| Action | Hands | Share of BB hands |
|---|---|---|
| 2 BB | 4508 | 90.16% |
| fold | 492 | 9.84% |

### Our BB response and final-hand outcomes

| Open | Response | Hands | Response share | Net chips | Chips / selected hand | Adds BB/100 (BB hands) |
|---|---|---|---|---|---|---|
| 2 BB | call | 2669 | 59.2% | -219700 | -82.32 | -43.94 |
| 2 BB | fold | 795 | 17.6% | -79500 | -100.00 | -15.90 |
| 2 BB | raise | 1044 | 23.2% | 130000 | +124.52 | +26.00 |
| fold | no response | 492 | 100.0% | 24600 | +50.00 | +4.92 |

### BB hands by terminal ending

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| later: player folded | 1533 | -603300 | -120.66 |
| later: showdown | 1017 | -30650 | -6.13 |
| later: slumbot folded | 807 | 491950 | +98.39 |
| preflop: player folded | 828 | -96600 | -19.32 |
| preflop: slumbot folded | 815 | 94000 | +18.80 |

### BB called a 2 BB open: later endings

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| player folded flop | 809 | -168900 | -33.78 |
| player folded river | 218 | -120500 | -24.10 |
| player folded turn | 331 | -107600 | -21.52 |
| showdown | 763 | -56350 | -11.27 |
| slumbot folded flop | 154 | 43000 | +8.60 |
| slumbot folded river | 218 | 125000 | +25.00 |
| slumbot folded turn | 176 | 65650 | +13.13 |

## confirmation-b

### Flop responses

| Responder | Position | Size / pot | Chances | Fold | Call | Raise |
|---|---|---|---|---|---|---|
| player | BB (OOP) | <0.4 | 36 | 11.1% | 69.4% | 19.4% |
| player | BB (OOP) | ~½ | 1436 | 57.9% | 34.5% | 7.7% |
| player | BB (OOP) | ~⅔–¾ | 37 | 56.8% | 32.4% | 10.8% |
| player | BB (OOP) | ~pot | 60 | 51.7% | 33.3% | 15.0% |
| player | BB (OOP) | 1.2–2.5× | 21 | 52.4% | 38.1% | 9.5% |
| player | BB (OOP) | >2.5× | 1 | 100.0% | 0.0% | 0.0% |
| player | BB (OOP) | all-in | 4 | 0.0% | 100.0% | 0.0% |
| player | BTN (IP) | <0.4 | 11 | 0.0% | 63.6% | 36.4% |
| player | BTN (IP) | ~½ | 575 | 48.7% | 42.3% | 9.0% |
| player | BTN (IP) | ~⅔–¾ | 21 | 19.0% | 66.7% | 14.3% |
| player | BTN (IP) | ~pot | 87 | 50.6% | 42.5% | 6.9% |
| player | BTN (IP) | 1.2–2.5× | 2 | 0.0% | 50.0% | 50.0% |
| player | BTN (IP) | all-in | 3 | 33.3% | 66.7% | 0.0% |
| slumbot | BB (OOP) | <0.4 | 741 | 26.2% | 62.9% | 10.9% |
| slumbot | BB (OOP) | ~½ | 307 | 40.1% | 53.1% | 6.8% |
| slumbot | BB (OOP) | ~pot | 636 | 57.1% | 39.5% | 3.5% |
| slumbot | BTN (IP) | <0.4 | 797 | 20.8% | 59.3% | 19.8% |
| slumbot | BTN (IP) | ~pot | 229 | 51.5% | 38.0% | 10.5% |

### Terminal flop folds: additive contributions over all hands

| Size | Our folds | Slumbot folds | Net BB/100 |
|---|---|---|---|
| <0.4 | -0.12 | +11.67 | +11.55 |
| ~½ | -29.11 | +1.23 | -27.88 |
| ~⅔–¾ | -1.67 | +0.00 | -1.67 |
| ~pot | -5.16 | +13.98 | +8.82 |
| 1.2–2.5× | -0.44 | +0.00 | -0.44 |
| >2.5× | -0.04 | +0.00 | -0.04 |
| all-in | -0.81 | +0.00 | -0.81 |

### Slumbot’s first button action

| Action | Hands | Share of BB hands |
|---|---|---|
| 2 BB | 4514 | 90.28% |
| fold | 486 | 9.72% |

### Our BB response and final-hand outcomes

| Open | Response | Hands | Response share | Net chips | Chips / selected hand | Adds BB/100 (BB hands) |
|---|---|---|---|---|---|---|
| 2 BB | call | 2625 | 58.2% | -195800 | -74.59 | -39.16 |
| 2 BB | fold | 819 | 18.1% | -81900 | -100.00 | -16.38 |
| 2 BB | raise | 1070 | 23.7% | 39050 | +36.50 | +7.81 |
| fold | no response | 486 | 100.0% | 24300 | +50.00 | +4.86 |

### BB hands by terminal ending

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| later: player folded | 1567 | -667150 | -133.43 |
| later: showdown | 922 | -58750 | -11.75 |
| later: slumbot folded | 831 | 496350 | +99.27 |
| preflop: player folded | 842 | -93900 | -18.78 |
| preflop: slumbot folded | 838 | 109100 | +21.82 |

### BB called a 2 BB open: later endings

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| player folded flop | 809 | -168600 | -33.72 |
| player folded river | 217 | -150350 | -30.07 |
| player folded turn | 314 | -105900 | -21.18 |
| showdown | 714 | -14250 | -2.85 |
| slumbot folded flop | 168 | 45600 | +9.12 |
| slumbot folded river | 199 | 120200 | +24.04 |
| slumbot folded turn | 204 | 77500 | +15.50 |

## confirmation-c

### Flop responses

| Responder | Position | Size / pot | Chances | Fold | Call | Raise |
|---|---|---|---|---|---|---|
| player | BB (OOP) | <0.4 | 42 | 7.1% | 61.9% | 31.0% |
| player | BB (OOP) | ~½ | 1491 | 59.4% | 34.1% | 6.5% |
| player | BB (OOP) | ~⅔–¾ | 28 | 46.4% | 46.4% | 7.1% |
| player | BB (OOP) | ~pot | 57 | 47.4% | 29.8% | 22.8% |
| player | BB (OOP) | 1.2–2.5× | 30 | 60.0% | 26.7% | 13.3% |
| player | BB (OOP) | >2.5× | 2 | 0.0% | 100.0% | 0.0% |
| player | BB (OOP) | all-in | 5 | 40.0% | 60.0% | 0.0% |
| player | BTN (IP) | <0.4 | 16 | 6.2% | 50.0% | 43.8% |
| player | BTN (IP) | ~½ | 621 | 49.1% | 40.3% | 10.6% |
| player | BTN (IP) | ~⅔–¾ | 18 | 66.7% | 33.3% | 0.0% |
| player | BTN (IP) | ~pot | 95 | 47.4% | 42.1% | 10.5% |
| player | BTN (IP) | 1.2–2.5× | 1 | 0.0% | 100.0% | 0.0% |
| player | BTN (IP) | >2.5× | 1 | 100.0% | 0.0% | 0.0% |
| player | BTN (IP) | all-in | 3 | 66.7% | 33.3% | 0.0% |
| slumbot | BB (OOP) | <0.4 | 773 | 23.3% | 64.3% | 12.4% |
| slumbot | BB (OOP) | ~½ | 320 | 40.0% | 52.2% | 7.8% |
| slumbot | BB (OOP) | ~pot | 610 | 53.0% | 41.5% | 5.6% |
| slumbot | BTN (IP) | <0.4 | 752 | 19.8% | 58.4% | 21.8% |
| slumbot | BTN (IP) | ~½ | 2 | 50.0% | 50.0% | 0.0% |
| slumbot | BTN (IP) | ~pot | 212 | 53.3% | 38.2% | 8.5% |

### Terminal flop folds: additive contributions over all hands

| Size | Our folds | Slumbot folds | Net BB/100 |
|---|---|---|---|
| <0.4 | -0.14 | +9.93 | +9.79 |
| ~½ | -31.46 | +1.46 | -30.00 |
| ~⅔–¾ | -1.40 | +0.00 | -1.40 |
| ~pot | -4.45 | +12.23 | +7.78 |
| 1.2–2.5× | -0.82 | +0.00 | -0.82 |
| >2.5× | -0.11 | +0.00 | -0.11 |
| all-in | -2.20 | +0.00 | -2.20 |

### Slumbot’s first button action

| Action | Hands | Share of BB hands |
|---|---|---|
| 2 BB | 4513 | 90.26% |
| fold | 487 | 9.74% |

### Our BB response and final-hand outcomes

| Open | Response | Hands | Response share | Net chips | Chips / selected hand | Adds BB/100 (BB hands) |
|---|---|---|---|---|---|---|
| 2 BB | call | 2697 | 59.8% | -85800 | -31.81 | -17.16 |
| 2 BB | fold | 803 | 17.8% | -80300 | -100.00 | -16.06 |
| 2 BB | raise | 1013 | 22.4% | 8800 | +8.69 | +1.76 |
| fold | no response | 487 | 100.0% | 24350 | +50.00 | +4.87 |

### BB hands by terminal ending

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| later: player folded | 1574 | -609400 | -121.88 |
| later: showdown | 969 | 22600 | +4.52 |
| later: slumbot folded | 805 | 464600 | +92.92 |
| preflop: player folded | 848 | -114500 | -22.90 |
| preflop: slumbot folded | 804 | 103750 | +20.75 |

### BB called a 2 BB open: later endings

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| player folded flop | 866 | -179200 | -35.84 |
| player folded river | 226 | -134000 | -26.80 |
| player folded turn | 296 | -97600 | -19.52 |
| showdown | 740 | 97100 | +19.42 |
| slumbot folded flop | 165 | 46300 | +9.26 |
| slumbot folded river | 202 | 109100 | +21.82 |
| slumbot folded turn | 202 | 72500 | +14.50 |

## confirmation-pooled

### Flop responses

| Responder | Position | Size / pot | Chances | Fold | Call | Raise |
|---|---|---|---|---|---|---|
| player | BB (OOP) | <0.4 | 120 | 7.5% | 69.2% | 23.3% |
| player | BB (OOP) | ~½ | 4427 | 57.5% | 35.1% | 7.4% |
| player | BB (OOP) | ~⅔–¾ | 97 | 49.5% | 39.2% | 11.3% |
| player | BB (OOP) | ~pot | 176 | 46.0% | 38.6% | 15.3% |
| player | BB (OOP) | 1.2–2.5× | 76 | 52.6% | 36.8% | 10.5% |
| player | BB (OOP) | >2.5× | 6 | 33.3% | 50.0% | 16.7% |
| player | BB (OOP) | all-in | 13 | 38.5% | 61.5% | 0.0% |
| player | BTN (IP) | <0.4 | 40 | 2.5% | 52.5% | 45.0% |
| player | BTN (IP) | ~½ | 1783 | 49.9% | 40.7% | 9.4% |
| player | BTN (IP) | ~⅔–¾ | 56 | 44.6% | 46.4% | 8.9% |
| player | BTN (IP) | ~pot | 244 | 50.8% | 42.2% | 7.0% |
| player | BTN (IP) | 1.2–2.5× | 5 | 20.0% | 40.0% | 40.0% |
| player | BTN (IP) | >2.5× | 1 | 100.0% | 0.0% | 0.0% |
| player | BTN (IP) | all-in | 9 | 44.4% | 55.6% | 0.0% |
| slumbot | BB (OOP) | <0.4 | 2310 | 25.7% | 62.7% | 11.6% |
| slumbot | BB (OOP) | ~½ | 958 | 40.4% | 52.7% | 6.9% |
| slumbot | BB (OOP) | ~pot | 1797 | 55.6% | 39.9% | 4.5% |
| slumbot | BTN (IP) | <0.4 | 2335 | 20.3% | 59.1% | 20.5% |
| slumbot | BTN (IP) | ~½ | 3 | 33.3% | 33.3% | 33.3% |
| slumbot | BTN (IP) | ~pot | 656 | 51.8% | 38.6% | 9.6% |

### Terminal flop folds: additive contributions over all hands

| Size | Our folds | Slumbot folds | Net BB/100 |
|---|---|---|---|
| <0.4 | -0.11 | +11.23 | +11.13 |
| ~½ | -30.30 | +1.35 | -28.95 |
| ~⅔–¾ | -1.48 | +0.00 | -1.48 |
| ~pot | -4.44 | +13.19 | +8.75 |
| 1.2–2.5× | -0.69 | +0.00 | -0.69 |
| >2.5× | -0.06 | +0.00 | -0.06 |
| all-in | -1.63 | +0.00 | -1.63 |

### Slumbot’s first button action

| Action | Hands | Share of BB hands |
|---|---|---|
| 2 BB | 13535 | 90.23% |
| fold | 1465 | 9.77% |

### Our BB response and final-hand outcomes

| Open | Response | Hands | Response share | Net chips | Chips / selected hand | Adds BB/100 (BB hands) |
|---|---|---|---|---|---|---|
| 2 BB | call | 7991 | 59.0% | -501300 | -62.73 | -33.42 |
| 2 BB | fold | 2417 | 17.9% | -241700 | -100.00 | -16.11 |
| 2 BB | raise | 3127 | 23.1% | 177850 | +56.88 | +11.86 |
| fold | no response | 1465 | 100.0% | 73250 | +50.00 | +4.88 |

### BB hands by terminal ending

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| later: player folded | 4674 | -1879850 | -125.32 |
| later: showdown | 2908 | -66800 | -4.45 |
| later: slumbot folded | 2443 | 1452900 | +96.86 |
| preflop: player folded | 2518 | -305000 | -20.33 |
| preflop: slumbot folded | 2457 | 306850 | +20.46 |

### BB called a 2 BB open: later endings

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| player folded flop | 2484 | -516700 | -34.45 |
| player folded river | 661 | -404850 | -26.99 |
| player folded turn | 941 | -311100 | -20.74 |
| showdown | 2217 | 26500 | +1.77 |
| slumbot folded flop | 487 | 134900 | +8.99 |
| slumbot folded river | 619 | 354300 | +23.62 |
| slumbot folded turn | 582 | 215650 | +14.38 |

## 2026101302

### Flop responses

| Responder | Position | Size / pot | Chances | Fold | Call | Raise |
|---|---|---|---|---|---|---|
| player | BB (OOP) | <0.4 | 38 | 7.9% | 63.2% | 28.9% |
| player | BB (OOP) | ~½ | 1363 | 56.5% | 35.4% | 8.1% |
| player | BB (OOP) | ~⅔–¾ | 27 | 51.9% | 37.0% | 11.1% |
| player | BB (OOP) | ~pot | 61 | 49.2% | 31.1% | 19.7% |
| player | BB (OOP) | 1.2–2.5× | 21 | 76.2% | 23.8% | 0.0% |
| player | BB (OOP) | >2.5× | 2 | 0.0% | 100.0% | 0.0% |
| player | BTN (IP) | <0.4 | 11 | 0.0% | 36.4% | 63.6% |
| player | BTN (IP) | ~½ | 608 | 53.6% | 37.0% | 9.4% |
| player | BTN (IP) | ~⅔–¾ | 20 | 50.0% | 40.0% | 10.0% |
| player | BTN (IP) | ~pot | 76 | 48.7% | 42.1% | 9.2% |
| player | BTN (IP) | 1.2–2.5× | 4 | 25.0% | 75.0% | 0.0% |
| player | BTN (IP) | all-in | 6 | 33.3% | 66.7% | 0.0% |
| slumbot | BB (OOP) | <0.4 | 767 | 26.6% | 60.6% | 12.8% |
| slumbot | BB (OOP) | ~½ | 325 | 43.4% | 44.9% | 11.7% |
| slumbot | BB (OOP) | ~pot | 642 | 55.5% | 38.6% | 5.9% |
| slumbot | BB (OOP) | all-in | 1 | 0.0% | 100.0% | 0.0% |
| slumbot | BTN (IP) | <0.4 | 744 | 19.5% | 58.7% | 21.8% |
| slumbot | BTN (IP) | ~pot | 249 | 53.4% | 38.2% | 8.4% |

### Terminal flop folds: additive contributions over all hands

| Size | Our folds | Slumbot folds | Net BB/100 |
|---|---|---|---|
| <0.4 | -0.09 | +10.56 | +10.47 |
| ~½ | -29.37 | +1.41 | -27.96 |
| ~⅔–¾ | -1.41 | +0.00 | -1.41 |
| ~pot | -4.00 | +14.67 | +10.67 |
| 1.2–2.5× | -0.84 | +0.00 | -0.84 |
| >2.5× | +0.00 | +0.00 | +0.00 |
| all-in | -1.11 | +0.00 | -1.11 |

### Slumbot’s first button action

| Action | Hands | Share of BB hands |
|---|---|---|
| 2 BB | 4503 | 90.06% |
| fold | 497 | 9.94% |

### Our BB response and final-hand outcomes

| Open | Response | Hands | Response share | Net chips | Chips / selected hand | Adds BB/100 (BB hands) |
|---|---|---|---|---|---|---|
| 2 BB | call | 2556 | 56.8% | -129100 | -50.51 | -25.82 |
| 2 BB | fold | 887 | 19.7% | -88700 | -100.00 | -17.74 |
| 2 BB | raise | 1060 | 23.5% | -15900 | -15.00 | -3.18 |
| fold | no response | 497 | 100.0% | 24850 | +50.00 | +4.97 |

### BB hands by terminal ending

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| later: player folded | 1487 | -594100 | -118.82 |
| later: showdown | 917 | -133300 | -26.66 |
| later: slumbot folded | 818 | 525400 | +105.08 |
| preflop: player folded | 930 | -111500 | -22.30 |
| preflop: slumbot folded | 848 | 104650 | +20.93 |

### BB called a 2 BB open: later endings

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| player folded flop | 751 | -157100 | -31.42 |
| player folded river | 229 | -114900 | -22.98 |
| player folded turn | 324 | -111800 | -22.36 |
| showdown | 703 | 6300 | +1.26 |
| slumbot folded flop | 154 | 43000 | +8.60 |
| slumbot folded river | 212 | 141600 | +28.32 |
| slumbot folded turn | 183 | 63800 | +12.76 |

## 2026101421

### Flop responses

| Responder | Position | Size / pot | Chances | Fold | Call | Raise |
|---|---|---|---|---|---|---|
| player | BB (OOP) | <0.4 | 49 | 12.2% | 53.1% | 34.7% |
| player | BB (OOP) | ~½ | 1504 | 58.8% | 34.2% | 7.0% |
| player | BB (OOP) | ~⅔–¾ | 20 | 65.0% | 30.0% | 5.0% |
| player | BB (OOP) | ~pot | 71 | 50.7% | 36.6% | 12.7% |
| player | BB (OOP) | 1.2–2.5× | 27 | 33.3% | 44.4% | 22.2% |
| player | BB (OOP) | all-in | 7 | 42.9% | 57.1% | 0.0% |
| player | BTN (IP) | <0.4 | 13 | 0.0% | 46.2% | 53.8% |
| player | BTN (IP) | ~½ | 621 | 53.6% | 38.3% | 8.1% |
| player | BTN (IP) | ~⅔–¾ | 13 | 53.8% | 23.1% | 23.1% |
| player | BTN (IP) | ~pot | 73 | 47.9% | 45.2% | 6.8% |
| player | BTN (IP) | 1.2–2.5× | 4 | 0.0% | 75.0% | 25.0% |
| player | BTN (IP) | all-in | 4 | 25.0% | 75.0% | 0.0% |
| slumbot | BB (OOP) | <0.4 | 718 | 29.4% | 58.5% | 12.1% |
| slumbot | BB (OOP) | ~½ | 298 | 43.3% | 47.0% | 9.7% |
| slumbot | BB (OOP) | ~pot | 641 | 56.6% | 38.7% | 4.7% |
| slumbot | BTN (IP) | <0.4 | 782 | 19.7% | 58.2% | 22.1% |
| slumbot | BTN (IP) | ~½ | 1 | 0.0% | 100.0% | 0.0% |
| slumbot | BTN (IP) | ~pot | 217 | 53.0% | 36.9% | 10.1% |
| slumbot | BTN (IP) | all-in | 1 | 100.0% | 0.0% | 0.0% |

### Terminal flop folds: additive contributions over all hands

| Size | Our folds | Slumbot folds | Net BB/100 |
|---|---|---|---|
| <0.4 | -0.18 | +13.42 | +13.24 |
| ~½ | -32.05 | +1.29 | -30.76 |
| ~⅔–¾ | -1.19 | +0.00 | -1.19 |
| ~pot | -4.06 | +16.34 | +12.28 |
| 1.2–2.5× | -0.46 | +0.00 | -0.46 |
| >2.5× | +0.00 | +0.00 | +0.00 |
| all-in | -2.45 | +0.54 | -1.91 |

### Slumbot’s first button action

| Action | Hands | Share of BB hands |
|---|---|---|
| 2 BB | 4513 | 90.28% |
| fold | 486 | 9.72% |

### Our BB response and final-hand outcomes

| Open | Response | Hands | Response share | Net chips | Chips / selected hand | Adds BB/100 (BB hands) |
|---|---|---|---|---|---|---|
| 2 BB | call | 2767 | 61.3% | -114700 | -41.45 | -22.94 |
| 2 BB | fold | 702 | 15.6% | -70200 | -100.00 | -14.04 |
| 2 BB | raise | 1044 | 23.1% | 113000 | +108.24 | +22.60 |
| fold | no response | 486 | 100.0% | 24300 | +50.00 | +4.86 |

### BB hands by terminal ending

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| later: player folded | 1647 | -629400 | -125.91 |
| later: showdown | 1005 | 117250 | +23.45 |
| later: slumbot folded | 809 | 467750 | +93.57 |
| preflop: player folded | 751 | -95700 | -19.14 |
| preflop: slumbot folded | 787 | 92500 | +18.50 |

### BB called a 2 BB open: later endings

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| player folded flop | 874 | -181900 | -36.39 |
| player folded river | 250 | -135400 | -27.09 |
| player folded turn | 336 | -114500 | -22.90 |
| showdown | 766 | 87750 | +17.55 |
| slumbot folded flop | 144 | 42500 | +8.50 |
| slumbot folded river | 197 | 107650 | +21.53 |
| slumbot folded turn | 200 | 79200 | +15.84 |

## 2026101422

### Flop responses

| Responder | Position | Size / pot | Chances | Fold | Call | Raise |
|---|---|---|---|---|---|---|
| player | BB (OOP) | <0.4 | 38 | 10.5% | 65.8% | 23.7% |
| player | BB (OOP) | ~½ | 1472 | 56.4% | 34.9% | 8.7% |
| player | BB (OOP) | ~⅔–¾ | 22 | 50.0% | 36.4% | 13.6% |
| player | BB (OOP) | ~pot | 60 | 38.3% | 50.0% | 11.7% |
| player | BB (OOP) | 1.2–2.5× | 38 | 65.8% | 23.7% | 10.5% |
| player | BB (OOP) | >2.5× | 3 | 33.3% | 33.3% | 33.3% |
| player | BB (OOP) | all-in | 5 | 20.0% | 80.0% | 0.0% |
| player | BTN (IP) | <0.4 | 15 | 0.0% | 53.3% | 46.7% |
| player | BTN (IP) | ~½ | 602 | 52.5% | 38.5% | 9.0% |
| player | BTN (IP) | ~⅔–¾ | 16 | 68.8% | 25.0% | 6.2% |
| player | BTN (IP) | ~pot | 64 | 56.2% | 39.1% | 4.7% |
| player | BTN (IP) | 1.2–2.5× | 5 | 40.0% | 20.0% | 40.0% |
| player | BTN (IP) | >2.5× | 1 | 0.0% | 0.0% | 100.0% |
| player | BTN (IP) | all-in | 4 | 75.0% | 25.0% | 0.0% |
| slumbot | BB (OOP) | <0.4 | 782 | 27.4% | 60.0% | 12.7% |
| slumbot | BB (OOP) | ~½ | 283 | 41.3% | 52.3% | 6.4% |
| slumbot | BB (OOP) | ~pot | 604 | 58.9% | 37.4% | 3.6% |
| slumbot | BB (OOP) | all-in | 1 | 100.0% | 0.0% | 0.0% |
| slumbot | BTN (IP) | <0.4 | 796 | 21.0% | 56.2% | 22.9% |
| slumbot | BTN (IP) | ~½ | 2 | 50.0% | 50.0% | 0.0% |
| slumbot | BTN (IP) | ~pot | 233 | 54.1% | 34.3% | 11.6% |
| slumbot | BTN (IP) | all-in | 1 | 100.0% | 0.0% | 0.0% |

### Terminal flop folds: additive contributions over all hands

| Size | Our folds | Slumbot folds | Net BB/100 |
|---|---|---|---|
| <0.4 | -0.12 | +11.94 | +11.82 |
| ~½ | -29.71 | +1.35 | -28.36 |
| ~⅔–¾ | -1.22 | +0.00 | -1.22 |
| ~pot | -3.65 | +15.23 | +11.58 |
| 1.2–2.5× | -1.27 | +0.00 | -1.27 |
| >2.5× | -0.04 | +0.00 | -0.04 |
| all-in | -2.92 | +1.76 | -1.16 |

### Slumbot’s first button action

| Action | Hands | Share of BB hands |
|---|---|---|
| 2 BB | 4513 | 90.33% |
| fold | 483 | 9.67% |

### Our BB response and final-hand outcomes

| Open | Response | Hands | Response share | Net chips | Chips / selected hand | Adds BB/100 (BB hands) |
|---|---|---|---|---|---|---|
| 2 BB | call | 2660 | 58.9% | -72000 | -27.07 | -14.41 |
| 2 BB | fold | 818 | 18.1% | -81800 | -100.00 | -16.37 |
| 2 BB | raise | 1035 | 22.9% | 17800 | +17.20 | +3.56 |
| fold | no response | 483 | 100.0% | 24150 | +50.00 | +4.83 |

### BB hands by terminal ending

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| later: player folded | 1528 | -613500 | -122.80 |
| later: showdown | 998 | -31600 | -6.33 |
| later: slumbot folded | 845 | 555900 | +111.27 |
| preflop: player folded | 854 | -107600 | -21.54 |
| preflop: slumbot folded | 771 | 84950 | +17.00 |

### BB called a 2 BB open: later endings

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| player folded flop | 813 | -169200 | -33.87 |
| player folded river | 220 | -132700 | -26.56 |
| player folded turn | 298 | -106400 | -21.30 |
| showdown | 756 | 71000 | +14.21 |
| slumbot folded flop | 175 | 47300 | +9.47 |
| slumbot folded river | 209 | 151800 | +30.38 |
| slumbot folded turn | 189 | 66200 | +13.25 |

## exploratory

### Flop responses

| Responder | Position | Size / pot | Chances | Fold | Call | Raise |
|---|---|---|---|---|---|---|
| player | BB (OOP) | <0.4 | 40 | 15.0% | 62.5% | 22.5% |
| player | BB (OOP) | ~½ | 1491 | 53.9% | 37.5% | 8.6% |
| player | BB (OOP) | ~⅔–¾ | 20 | 40.0% | 40.0% | 20.0% |
| player | BB (OOP) | ~pot | 60 | 60.0% | 30.0% | 10.0% |
| player | BB (OOP) | 1.2–2.5× | 27 | 44.4% | 40.7% | 14.8% |
| player | BB (OOP) | >2.5× | 5 | 40.0% | 40.0% | 20.0% |
| player | BB (OOP) | all-in | 3 | 66.7% | 33.3% | 0.0% |
| player | BTN (IP) | <0.4 | 14 | 0.0% | 50.0% | 50.0% |
| player | BTN (IP) | ~½ | 573 | 47.3% | 43.8% | 8.9% |
| player | BTN (IP) | ~⅔–¾ | 25 | 44.0% | 36.0% | 20.0% |
| player | BTN (IP) | ~pot | 88 | 52.3% | 39.8% | 8.0% |
| player | BTN (IP) | 1.2–2.5× | 8 | 50.0% | 25.0% | 25.0% |
| player | BTN (IP) | all-in | 10 | 40.0% | 60.0% | 0.0% |
| slumbot | BB (OOP) | <0.4 | 747 | 28.9% | 56.0% | 15.1% |
| slumbot | BB (OOP) | ~½ | 316 | 43.7% | 49.1% | 7.3% |
| slumbot | BB (OOP) | ~pot | 623 | 52.6% | 43.0% | 4.3% |
| slumbot | BB (OOP) | all-in | 4 | 75.0% | 25.0% | 0.0% |
| slumbot | BTN (IP) | <0.4 | 760 | 17.1% | 60.9% | 22.0% |
| slumbot | BTN (IP) | ~½ | 1 | 0.0% | 100.0% | 0.0% |
| slumbot | BTN (IP) | ~pot | 225 | 46.7% | 40.9% | 12.4% |
| slumbot | BTN (IP) | all-in | 3 | 33.3% | 66.7% | 0.0% |

### Terminal flop folds: additive contributions over all hands

| Size | Our folds | Slumbot folds | Net BB/100 |
|---|---|---|---|
| <0.4 | -0.24 | +10.60 | +10.36 |
| ~½ | -27.80 | +1.38 | -26.42 |
| ~⅔–¾ | -1.01 | +0.00 | -1.01 |
| ~pot | -4.90 | +13.38 | +8.48 |
| 1.2–2.5× | -0.94 | +0.00 | -0.94 |
| >2.5× | -0.08 | +0.00 | -0.08 |
| all-in | -3.83 | +2.43 | -1.40 |

### Slumbot’s first button action

| Action | Hands | Share of BB hands |
|---|---|---|
| 2 BB | 4510 | 90.20% |
| fold | 490 | 9.80% |

### Our BB response and final-hand outcomes

| Open | Response | Hands | Response share | Net chips | Chips / selected hand | Adds BB/100 (BB hands) |
|---|---|---|---|---|---|---|
| 2 BB | call | 2664 | 59.1% | -88550 | -33.24 | -17.71 |
| 2 BB | fold | 796 | 17.6% | -79600 | -100.00 | -15.92 |
| 2 BB | raise | 1050 | 23.3% | 78750 | +75.00 | +15.75 |
| fold | no response | 490 | 100.0% | 24500 | +50.00 | +4.90 |

### BB hands by terminal ending

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| later: player folded | 1519 | -647500 | -129.50 |
| later: showdown | 1046 | 77650 | +15.53 |
| later: slumbot folded | 782 | 504250 | +100.85 |
| preflop: player folded | 829 | -101800 | -20.36 |
| preflop: slumbot folded | 824 | 102500 | +20.50 |

### BB called a 2 BB open: later endings

| Ending | Hands | Net chips | Adds BB/100 (BB hands) |
|---|---|---|---|
| player folded flop | 789 | -164500 | -32.90 |
| player folded river | 224 | -142300 | -28.46 |
| player folded turn | 316 | -111600 | -22.32 |
| showdown | 798 | 80900 | +16.18 |
| slumbot folded flop | 140 | 43200 | +8.64 |
| slumbot folded river | 192 | 115800 | +23.16 |
| slumbot folded turn | 205 | 89950 | +17.99 |

## Consistency of the largest specific flop-fold paths

All losses below use all hands as the denominator. These are gross losses from our own folds; Slumbot’s folds elsewhere offset them. Different rows are disjoint terminal outcomes. They overlap the BB first-response partition and must not be added to it.

| Run | All flop folds net | BB 2 BB call / check-fold: n; loss | BTN limp-call / fold: n; loss | BTN 3bet+ / half-pot: folds/chances; loss | BB call group (BB denominator) |
|---|---|---|---|---|---|
| confirmation-a | -11.35 | 768; -15.36 | 117; -3.51 | 69/110; -4.80 | -43.94 |
| confirmation-b | -10.47 | 753; -15.06 | 121; -3.63 | 52/114; -4.17 | -39.16 |
| confirmation-c | -16.95 | 812; -16.24 | 133; -3.99 | 67/111; -4.83 | -17.16 |
| confirmation-pooled | -12.93 | 2333; -15.55 | 371; -3.71 | 188/335; -4.60 | -33.42 |
| 2026101302 | -10.18 | 688; -13.76 | 127; -3.81 | 57/112; -4.20 | -25.82 |
| 2026101421 | -8.79 | 812; -16.24 | 120; -3.60 | 77/133; -5.25 | -22.94 |
| 2026101422 | -8.64 | 759; -15.18 | 129; -3.87 | 51/101; -3.66 | -14.41 |
| exploratory | -11.01 | 740; -14.80 | 116; -3.48 | 44/97; -2.91 | -17.71 |

The first three flop patterns occur with similar frequency in every run. The BB call group loses in every run, but BB raise outcomes and showdown results vary in sign; there is no consistent claim that raising more would improve the policy.

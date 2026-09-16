# Dataset Documentation — PhysioNet MIT-BIH Arrhythmia Database

## 1. Dataset Overview & Source
* **Dataset Name:** PhysioNet MIT-BIH Arrhythmia Database
* **Primary Source:** PhysioNet (PhysioBank Archive)
* **Official URL:** [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
* **Local Directory:** `data/raw/mitdb/`

## 2. Discovered Dataset Statistics
* **Total Valid WFDB Records Found:** `48` records
* **Record List:** `100`, `101`, `102`, `103`, `104`, `105`, `106`, `107`, `108`, `109`, `111`, `112`, `113`, `114`, `115`, `116`, `117`, `118`, `119`, `121`, `122`, `123`, `124`, `200`, `201`, `202`, `203`, `205`, `207`, `208`, `209`, `210`, `212`, `213`, `214`, `215`, `217`, `219`, `220`, `221`, `222`, `223`, `228`, `230`, `231`, `232`, `233`, `234`
* **Sampling Frequency ($f_s$):** `360 Hz` across all 48 records
* **ECG Lead Channels:**
  * Primary Lead: `MLII` (Modified Lead II, present in 46 of 48 records)
  * Secondary Leads: `V1`, `V2`, `V4`, `V5`
* **Total Raw Annotations Inspected:** `112,647` annotations

## 3. Discovered Annotation Symbols & Frequencies
| Annotation Symbol | Description | Frequency Count |
| :--- | :--- | :--- |
| `N` | Normal beat | 75,052 |
| `L` | Left bundle branch block beat | 8,075 |
| `R` | Right bundle branch block beat | 7,259 |
| `V` | Premature ventricular contraction (PVC) | 7,130 |
| `/` | Paced beat | 7,028 |
| `A` | Atrial premature beat (APB) | 2,546 |
| `+` | Rhythm change marker (non-beat) | 1,291 |
| `f` | Fusion of paced and normal beat | 982 |
| `F` | Fusion of ventricular and normal beat | 803 |
| `~` | Signal quality change (non-beat) | 616 |
| `!` | Ventricular flutter wave (non-beat) | 472 |
| `"` | Comment marker (non-beat) | 437 |
| `j` | Nodal / junctional escape beat | 229 |
| `x` | Non-conducted P-wave (non-beat) | 193 |
| `a` | Aberrant atrial premature beat | 150 |
| `|` | Isolated QRS comment (non-beat) | 132 |
| `E` | Ventricular escape beat | 106 |
| `J` | Nodal / junctional premature beat | 83 |
| `Q` | Unclassifiable beat | 33 |
| `e` | Atrial escape beat | 16 |
| `S` | Premature supraventricular beat | 2 |

## 4. AAMI EC57 5-Class Categorization
To align with standard biomedical machine learning protocols (AAMI EC57 standard), the 15 beat symbols are mapped into 5 major clinical arrhythmia categories:

1. **N (Normal / Bundle Branch Block):** `N`, `L`, `R`, `e`, `j`
2. **S (Supraventricular Ectopic Beats - SVEB):** `A`, `a`, `J`, `S`
3. **V (Ventricular Ectopic Beats - VEB):** `V`, `E`
4. **F (Fusion Beats):** `F`
5. **Q (Paced / Unclassifiable Beats):** `/`, `f`, `Q`

## 5. Signal Preprocessing & Segmentation Pipeline
1. **Lead Selection:** Automatically selects `MLII` lead when available, falling back to Channel 0.
2. **Bandpass Filtering:** 2nd order Butterworth bandpass filter ($0.5\text{ Hz} - 45.0\text{ Hz}$) to remove baseline wander and powerline interference.
3. **R-Peak Segmentation:** Extracts a $180$-sample window ($90$ samples before R-peak, $90$ samples after R-peak) around each valid beat annotation location.
4. **Amplitude Normalization:** Applies Min-Max normalization $[0, 1]$ per heartbeat segment.
5. **Data Leakage Protection:** Includes `record_id` with every extracted heartbeat row to ensure train-test splitting is grouped by patient/record ID.

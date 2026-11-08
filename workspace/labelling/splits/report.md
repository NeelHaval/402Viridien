# Train / test split report

- pages: **42** total — train **32**, test **10** (23.8%)
- groups: **25** total — train **17**, test **8**
- score total: **0.983** (lower is better)

## Run metadata

| key | value |
|---|---|
| seed | `42` |
| created | `2026-11-07T16:25:08+00:00` |
| method | `group-level random search (50000 candidates, seed 42) minimising size deviation + strata TV + numeric SMD + empty-stratum / dominant-group penalties; groups = API-10 well id with field-level merges; 0000000009_d-582 held out as test_bw, log01 excluded` |
| test_frac | `0.2` |

## Groups

| group_id | n_pages | split | strata | stems |
|---|---:|---|---|---|
| 6080840014 | 1 | **test** | colour/masterlog | 60808400140000_hadrian 2_masterlog_2in_10000-21962 |
| 6081140512 | 1 | **test** | colour/mudlog | 60811405120000_1inch tvd mudlog |
| 6081140535 | 1 | **test** | colour/mudlog | 60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400 |
| 6081140758 | 1 | **test** | colour/mudlog | 608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD |
| 6081240048 | 1 | **test** | colour/masterlog | 60812400480000_ocs-g 26419 001 st00bp00_masterlog_7860-27400_5in_md_2 |
| 6081240102 | 1 | **test** | colour/mudlog | 60812401020000_xom_ocs-g_20351_ju103_st00bp00_mudlog_10200-30900ft_5in_tvd |
| Callaway 3-33 mudlog to TD.PDF | 1 | **test** | colour/mudlog | Callaway 3-33 mudlog to TD.PDF |
| field_GC743 | 3 | **test** | colour/mudlog | 60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev<br>60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872<br>60811407570000_bp_gc743_15607_dc802_st00bp00_mudlog_5md_6900-17643 |
| 6080740286 | 1 | train | colour/mudlog | 60807402860000_gb506#2_md_mudlog |
| 6081140580 | 4 | train | colour/mudlog | 60811405800000_608114058000_g16786_001_00_00_md5_mud_show_2<br>60811405800100_mdn-1_st00bp01_5 inch tvdmudlog_4560ft-22998ft<br>60811405800300_mdn-1_st01bp01_5 inch tvdmudlog_4560ft to 20251ft<br>60811405800500_608114058005_g16786_001_03_00_tvd5_mud |
| 6081140605 | 2 | train | colour/masterlog | 60811406050000_608114060500_g31765 001 st00 bp00__master log_5640-28740_5 in_md<br>60811406050100_ardennes_ocs-g31765 001 st00 bp01__master log_5640-36552_5 in_md |
| 6081240081 | 1 | train | colour/mudlog | 60812400810000_ocs-g 16987 003 st00bp00_1inmd_mudlog_8921_15721 |
| 6081741014 | 2 | train | colour/mudlog | 60817410140000_mc822#10 1-1200 mud log<br>60817410140100_mc822#10bp01 1-240 tvd mud log |
| 6081741140 | 3 | train | colour/mudlog | 60817411400100_mc761_1_st1_mudlog_1md_17133_22886<br>60817411400100_mc761_1_st1_mudlog_1tvd_17129_20968<br>60817411400200_mc761_1_st1_bp01_mudlog_1md_17133_25775 |
| 6081741142 | 1 | train | colour/masterlog | 60817411420000_mc_540_#1_st00_bp00_masterlog_3400-24195_5in_md |
| 6081840060 | 1 | train | colour/mudlog | 60818400600100_608184006001_g08037_002_01_00_5 inch mudlog_13452ft-19912ft |
| 6081840065 | 1 | train | colour/mudlog | 60818400650000_at0574_4_neptune_1inch_tvdmudlog_6300-17378 |
| MC765_P011 | 4 | train | colour/mudlog | MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft<br>MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft<br>MC765_P011_ST00BP00_5inchTVD_Depth_MudLog_3686-16495ft<br>MC765_P011_ST00BP01_5inchMD_Depth_MudLog_18230-21150ft |
| Mudlog-Trail Mtn1Palisades | 1 | train | colour/mudlog | Mudlog-Trail Mtn1Palisades |
| Mudlog_MD | 1 | train | colour/mudlog | Mudlog_MD_FREEMAN 3 24 ST |
| Shell_08852 | 1 | train | colour/mudlog | Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft |
| Shell_AC0336 | 1 | train | colour/mudlog | Shell_AC0336_OCS-G-36102_001_ST00BP01_5inchMD_MudLog_9100 ft -27770 ft |
| field_CritterCreek_Hereford | 2 | train | colour/mudlog | Critter Creek 201-1807H_Mud_PDF_Vertical.pdf<br>Critter Creek 202-1807H_Mud_Vertical_PDF.pdf |
| field_MC776 | 2 | train | colour/mudlog | 60817411700000_mc776-5_5in_md_combo_mud_log_20415_mms<br>60817412840000_mc_776_6_ml_mudlog_5757_23721_5in_md_final |
| field_Rio_Endeavor_Spraberry | 4 | train | colour/mudlog | Rio_Lavaca_1748_Unit_1_241_MUD<br>Rio_Nueces_3744_Unit_1_212_MUD_5IN<br>Rio_Nueces_3744_Unit_1_222_5IN_MUD<br>Rio_San_Gabriel_2829_Unit_2_272_MUD |

## Strata (group-majority `(is_colour, log_type)`)

| stratum | train pages | test pages | test page % | train groups | test groups |
|---|---:|---:|---:|---:|---:|
| colour/masterlog | 3 | 2 | 40% | 2 | 2 |
| colour/mudlog | 29 | 8 | 22% | 15 | 6 |

## Feature means

SMD = (test − train) / pooled SD, NaNs ignored. Features marked * are used in the score.

| feature | train mean | test mean | SMD | train n | test n |
|---|---:|---:|---:|---:|---:|
| blank_frac* | 0.117 | 0.144 | 0.19 | 32 | 10 |
| texture_diversity* | 4.562 | 4.000 | -0.24 | 32 | 10 |
| legend_aspect* | 3.900 | 3.290 | -0.48 | 32 | 10 |
| column_height_px* | 96,825 | 76,168 | -0.26 | 32 | 10 |
| mean_sat | 0.569 | 0.522 | -0.17 | 32 | 10 |
| px_per_ft | 6.778 | 5.500 | -0.33 | 18 | 8 |
| legend_area_frac | 0.014 | 0.008 | -0.22 | 32 | 10 |
| depth_range_ft | 13,360 | 14,577 | 0.17 | 18 | 8 |
| is_colour (fraction) | 1.000 | 1.000 | — | 32 | 10 |

## Score breakdown

| term | raw value | weight | weighted |
|---|---|---:|---:|
| size | \|10 − 10\| = 0 pages | 1.00 | 0.000 |
| strata_tv | TV = 0.220 | 4.00 | 0.878 |
| smd | mean \|SMD\| = 0.053 (blank_frac 0.11, texture_diversity 0.04, legend_aspect 0.02, column_height_px 0.04) | 2.00 | 0.105 |
| empty_stratum | 0 empty multi-group strata | 3.00 | 0.000 |
| dominant_group | max group share = 0.300 (field_GC743; limit 0.4) | 5.00 | 0.000 |
| **total** | | | **0.983** |

## Estimated labelling load (test set)

- test lithology column length: **761,680 px** across 10 pages
- depth covered: **125,181 ft** = 116,619 ft known + 8,562 ft estimated
- pages with unknown depth_range_ft: **2** — estimated as column_height_px / median px_per_ft (7.56 px/ft, median over 26 pages with known px_per_ft)

## Feature warnings

None.

## Grouping decisions

Group id = first 10 digits of the US API number; otherwise prefix up to the 2nd `_`. Field-level merges applied on top (same block/pad/lease shares legend + geology):

- `6081140410` -> `field_GC743`
- `6081140730` -> `field_GC743`
- `6081140757` -> `field_GC743`
- `6081741170` -> `field_MC776`
- `6081741284` -> `field_MC776`
- `Critter Creek 201-1807H_Mud` -> `field_CritterCreek_Hereford`
- `Critter Creek 202-1807H_Mud` -> `field_CritterCreek_Hereford`
- `Rio_Lavaca` -> `field_Rio_Endeavor_Spraberry`
- `Rio_Nueces` -> `field_Rio_Endeavor_Spraberry`
- `Rio_San` -> `field_Rio_Endeavor_Spraberry`

API numbers read from headers of pages without one in the filename (none matched an API-named group, so no cross-merges):

- `0000000009_d-582`: 608114024601 (Conoco OCS-G-11075 GC562 #1 ST#1)
- `MC765_P011`: 608174143300
- `Shell_08852`: 608174139501 (MC764 KE001)
- `Shell_AC0336`: 608054009701
- `Callaway 3-33 mudlog to TD.PDF`: 05039066800000
- `Critter Creek 201-1807H_Mud`: 05-123-44933
- `Critter Creek 202-1807H_Mud`: 051234493500
- `Mudlog-Trail Mtn1Palisades`: 27-011-05309
- `Mudlog_MD`: 05-055-06309 (Freeman 3-24 ST)
- `Rio_Lavaca`: 42-329-42079
- `Rio_Nueces`: 42-329-41851 (#212; #222 not checked)
- `Rio_San`: 42-329-42511

Automatic grouping heuristics flagged (before merges):

- non-API group (fallback rule): '0000000009_d-582' <- ['0000000009_d-582_0000020435-0001_log01_fe_1in_md', '0000000009_d-582_0000020435-0001_log02_fe_5in_md', '0000000009_d-582_0000020435-0001_log03_chromatograph_ratio', '0000000009_d-582_0000020435-0001_log04_fe_5in_md', '0000000009_d-582_0000020435-0001_log05_chromatograph_ratio', '0000000009_d-582_0000020435-0001_log06_fe_5in_md', '0000000009_d-582_0000020435-0001_log07_chromatograph_ratio', '0000000009_d-582_0000020435-0001_log08_fe_5in_md', '0000000009_d-582_0000020435-0001_log09_chromatograph_ratio']
- non-API group (fallback rule): 'Callaway 3-33 mudlog to TD.PDF' <- ['Callaway 3-33 mudlog to TD.PDF']
- non-API group (fallback rule): 'Critter Creek 201-1807H_Mud' <- ['Critter Creek 201-1807H_Mud_PDF_Vertical.pdf']
- non-API group (fallback rule): 'Critter Creek 202-1807H_Mud' <- ['Critter Creek 202-1807H_Mud_Vertical_PDF.pdf']
- non-API group (fallback rule): 'MC765_P011' <- ['MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft', 'MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft', 'MC765_P011_ST00BP00_5inchTVD_Depth_MudLog_3686-16495ft', 'MC765_P011_ST00BP01_5inchMD_Depth_MudLog_18230-21150ft']
- non-API group (fallback rule): 'Mudlog-Trail Mtn1Palisades' <- ['Mudlog-Trail Mtn1Palisades']
- non-API group (fallback rule): 'Mudlog_MD' <- ['Mudlog_MD_FREEMAN 3 24 ST']
- non-API group (fallback rule): 'Rio_Lavaca' <- ['Rio_Lavaca_1748_Unit_1_241_MUD']
- non-API group (fallback rule): 'Rio_Nueces' <- ['Rio_Nueces_3744_Unit_1_212_MUD_5IN', 'Rio_Nueces_3744_Unit_1_222_5IN_MUD']
- non-API group (fallback rule): 'Rio_San' <- ['Rio_San_Gabriel_2829_Unit_2_272_MUD']
- non-API group (fallback rule): 'Shell_08852' <- ['Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft']
- non-API group (fallback rule): 'Shell_AC0336' <- ['Shell_AC0336_OCS-G-36102_001_ST00BP01_5inchMD_MudLog_9100 ft -27770 ft']
- token 'gc743' shared by groups ['6081140410', '6081140730', '6081140757'] (same block/lease, different wells?)
- token 'mc776' shared by groups ['6081741170', '6081741284'] (same block/lease, different wells?)
- non-API groups share leading word 'critter': ['Critter Creek 201-1807H_Mud', 'Critter Creek 202-1807H_Mud']
- non-API groups share leading word 'mudlog': ['Mudlog-Trail Mtn1Palisades', 'Mudlog_MD']
- non-API groups share leading word 'rio': ['Rio_Lavaca', 'Rio_Nueces', 'Rio_San']
- non-API groups share leading word 'shell': ['Shell_08852', 'Shell_AC0336']


## B&W holdout (`test_bw`)

Entire well held out of training; log01 excluded because it was used for baseline development. Tables above describe colour pages only. The original split-search score precedes removal of the held-out well from training.

- `0000000009_d-582_0000020435-0001_log02_fe_5in_md`
- `0000000009_d-582_0000020435-0001_log03_chromatograph_ratio`
- `0000000009_d-582_0000020435-0001_log04_fe_5in_md`
- `0000000009_d-582_0000020435-0001_log05_chromatograph_ratio`
- `0000000009_d-582_0000020435-0001_log06_fe_5in_md`
- `0000000009_d-582_0000020435-0001_log07_chromatograph_ratio`
- `0000000009_d-582_0000020435-0001_log08_fe_5in_md`
- `0000000009_d-582_0000020435-0001_log09_chromatograph_ratio`

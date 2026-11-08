# Legend parser repair

Implementation: `harness/legend_v2.py`; original `LegendEntry` is imported from `harness.legend`. The original parser is unchanged. All v2 JSON, swatch PNGs, OCR records and outlined debug crops go to `out/_cache/legend_v2/`. The dataset is read-only throughout.

## Discovery and cause

The audit called `io.list_logs()` and `legend.parse(log)` on all 51 logs. Eleven had zero lithology entries. Every failing source crop was saved to `notes/legfix_crops/<sheet number>.png` and opened for inspection.

The decisive failure is heading recognition: the original regular expression requires spaces and plural `symbols`, while OCR returns `LITHOLOGYSYMBOL` and `SHOW&DRILLINGSYMBOL`. Only `MODIFIERS` survives, and the nearest-heading rule assigns every swatch to it. Rectangle detection and right-hand label pairing already find the cells. Simply accepting the missing headings would still discard real rocks because the exporter has flattened the categories to five columns while retaining a three-title banner.

The requested stem `60811407300100_bp_gc743_15607_dc141_st00bp01_mudlog_1md_10700-22872` is absent from the 51 names returned here. The actual failing DC141 log is `60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev`. The similarly named DC222 log is present and passes the regression unchanged.

## Fix and visual review

Headings are normalized independent of OCR whitespace, punctuation and singular/plural `symbol`. True panels use continuous vertical rules, or heading midpoint boundaries when unruled; only the Lithology section can produce lithology entries. A general flattened-grid test requires a complete rock row spanning more columns than the heading count, a technical row, and no continuous panel dividers. This layout uses positive rock vocabulary instead of banner proximity. Gas_Good, Dry, fresh/salted water, core, casing, tests and survey symbols remain non-lithology. There are no log-name conditions or per-log settings.

For the recognized three-title layouts, batched recognition of tight OCR text crops recovers spaces without repeatedly running text detection. Other layouts retain the original detection, OCR and classification rules. Swatches remain RGB interior crops and boxes use full-image coordinates.

All eleven final contact sheets were opened and checked against the source crops. A short technical row in the two Ardennes legends initially prevented flattened-grid recognition; the rule was corrected to accept that shorter row. Their final sheets include Tar and Sylvite as well as the other rocks. Printed truncations such as `Calcareous clay-shal` and `Argillaceous limesto` are preserved. Some sources themselves print `Silstone` (confirmed in an enlarged source crop, `notes/legfix_crops/text_detail.png`); the returned label preserves that spelling. No per-log spelling substitutions were introduced.

## Per-log results

Counts are entries, while regression comparisons use sets of exact labels. One existing log has 16 entries but 15 distinct labels. Sheet numbers identify `notes/legfix_sheets/<n>.png`. Labels below are the returned label sets.

| Log | n_lith before | n_lith after | Labels after | Sheet |
|---|---:|---:|---|---:|
| 0000000009_d-582_0000020435-0001_log01_fe_1in_md | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 0000000009_d-582_0000020435-0001_log02_fe_5in_md | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 0000000009_d-582_0000020435-0001_log03_chromatograph_ratio | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 0000000009_d-582_0000020435-0001_log04_fe_5in_md | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 0000000009_d-582_0000020435-0001_log05_chromatograph_ratio | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 0000000009_d-582_0000020435-0001_log06_fe_5in_md | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 0000000009_d-582_0000020435-0001_log07_chromatograph_ratio | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 0000000009_d-582_0000020435-0001_log08_fe_5in_md | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 0000000009_d-582_0000020435-0001_log09_chromatograph_ratio | 10 | 10 | Anhydrlte; Clay; Coal; Dolomite; Halite; LImestone; SIllty Sandstone; Sand and Sandstone; Shale; Siltstone | — |
| 60807402860000_gb506#2_md_mudlog | 12 | 12 | Anhydrite; Chert; Clay/Shale; Diatomite; Dolomite; Gypsum; Halite; Limestone; Plastic Clay; Sandstone; Shale; Siltstone | — |
| 60808400140000_hadrian 2_masterlog_2in_10000-21962 | 18 | 18 | Anhydrite; Bitumen; Dolomite; Dolomitic clay; Gypsum; Halite; I!S; Limestone; MDT; Marl; Sand; Sandstone; Shale brown; Shale green; Shale grey; Siltstone; Sylvite; Tuff | — |
| 60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev | 0 | 15 | Argill and calc sand; Argillaceous limesto; Bituminous limestone; Calcareous clay-shal; Clay-shale; Dolomite; Fine sand; Fine sandstone; Halite; Limestone; Marl; Medium sand; Oolitic limestone; Silstone; Silt | 1 |
| 60811405120000_1inch tvd mudlog | 10 | 10 | Clay/Shale; Conglomerate; Dolomite; Evaporites; Gypsum; Halite; Limestone; Sandstone; Sandy Clay/Shale; Siltstone | — |
| 60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400 | 0 | 15 | Anhydrite; Chalk; Chert; Clay; Dolomite; Dolomitic limestone; Granite; Gypsum; Halite; Limestone; Marl; Sandstone; Shale; Siltstone; Volcanics | 2 |
| 60811405800000_608114058000_g16786_001_00_00_md5_mud_show_2 | 0 | 15 | Anhydrite; Chalk; Chert; Claystone; Dolomite; Dolomitic limestone; Fine sandstone; Granite; Gypsum; Halite; Limestone; Marl; Shale; Silstone; Volcanics | 3 |
| 60811405800100_mdn-1_st00bp01_5 inch tvdmudlog_4560ft-22998ft | 0 | 15 | Anhydrite; Chalk; Chert; Claystone; Dolomite; Dolomitic limestone; Fine sandstone; Granite; Gypsum; Halite; Limestone; Marl; Shale; Silstone; Volcanics | 4 |
| 60811405800300_mdn-1_st01bp01_5 inch tvdmudlog_4560ft to 20251ft | 0 | 15 | Anhydrite; Chalk; Chert; Claystone; Dolomite; Dolomitic limestone; Fine sandstone; Granite; Gypsum; Halite; Limestone; Marl; Shale; Silstone; Volcanics | 5 |
| 60811405800500_608114058005_g16786_001_03_00_tvd5_mud | 0 | 15 | Anhydrite; Chalk; Chert; Claystone; Dolomite; Dolomitic limestone; Fine sandstone; Granite; Gypsum; Halite; Limestone; Marl; Shale; Silstone; Volcanics | 6 |
| 60811406050000_608114060500_g31765 001 st00 bp00__master log_5640-28740_5 in_md | 0 | 16 | Chalk; Chert; Clay-shale; Dolomite; Dolomitic limestone; Granite; Gypsum; Halite; Limestone; Marl; Sandstone; Sandy Clay-Shale; Shale; Siltstone; Sylvite; Tar | 7 |
| 60811406050100_ardennes_ocs-g31765 001 st00 bp01__master log_5640-36552_5 in_md | 0 | 16 | Chalk; Chert; Clay-shale; Dolomite; Dolomitic limestone; Granite; Gypsum; Halite; Limestone; Marl; Sandstone; Sandy Clay-Shale; Shale; Siltstone; Sylvite; Tar | 8 |
| 60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872 | 30 | 30 | Anhydrite; Argil and sandy dolomite; Argillaceous limestone; Basalt; Bituminous limestone; Calcareous clay-shale; Calcareous sandstone; Chert; Clay Shale; Coal-lignite; Coarsesandstone; Dolomite; Dolomitic sandstone; Fine Dolomite; Fine sand; Fine sandstone; Granite; Gypsum; Halite; Limestone; Marl; Medium sand; Medium sandstone; Oolic limestone; Sandy Dolomite; Sandy and calcareous dolomite; Silt; Siltstone; Silty clay-shale; Sylvite | — |
| 60811407570000_bp_gc743_15607_dc802_st00bp00_mudlog_5md_6900-17643 | 30 | 30 | Anhydrite; Argil and sandy dolomite; Argillaceous limestone; Basalt; Bituminous limestone; Calcareous clay-shale; Calcareous sandstone; Chert; Clay Shale; Coal-lignite; Coarse sandstone; Dolomite; Dolomitic sandstone; Fine Dolomite; Fine sand; Fine sandstone; Granite; Gypsum; Halite; Limestone; Marl; Medium sandstone; Mediumsand; Oolic limestone; Sandy Dolomite; Sandy and calcareous dolomite; Silt; Siltstone; Silty clay-shale; Sylvite | — |
| 608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD | 30 | 30 | Arg. Limestone; Boundstone; Breccia; Calc. Dolomite; Chalk; Chert; Claystone; Claystone Sililicif.; Coal - Lignite; Conglomerate; Dol. Limestone; Dolomite; Extrusive Rocks; Grainstone; Gypsum, Anhyd.; Intrusive Rocks; Limestone; Marl; Methamorphics; Mudstone; Packstone; Pebbles; Salt (K-Na-Mg); Sand; Sandstone; Sils. Limestone; Silt; Silty Clay; Volcanics; Wackstone | — |
| 60812400480000_ocs-g 26419 001 st00bp00_masterlog_7860-27400_5in_md_2 | 13 | 13 | Anhydrite; Claystone; Dolomite; Gas_Good; Gypsum; Halite; Limestone; Marl; Oil_Good; Salted water; Sandstone; Shale; Siltstone | — |
| 60812400810000_ocs-g 16987 003 st00bp00_1inmd_mudlog_8921_15721 | 16 | 16 | Calcareous clay-shal; Calcareous sandstone; Clay-shale; Dry; Fine sand; Fine sandstone; Gas_Good; Halite; Limestone; Marl; Medium sand; Oil_Good; Oil_Poor; S; Volcanic cinder, tuf | — |
| 60812401020000_xom_ocs-g_20351_ju103_st00bp00_mudlog_10200-30900ft_5in_tvd | 0 | 15 | Anhydrite; Bitumen; Calcareous clay-shal; Chalk; Clay-shale; Coarse sand; Dolomite; Dolomitic limestone; Fine sand; Gypsum; Halite; Limestone; Silty clay-shale; Slaty shale; Volcanic rock | 9 |
| 60817410140000_mc822#10 1-1200 mud log | 21 | 21 | Anhydrite; Chalk; Clay-shale; Coal-lignite; Coarse sandstone; Conglomerate; Dolomite; Dolomitic limestone; Fine sandstone; Gypsum; Halite; Limestone; Marl; Medium sandstone; Oil_Good; Oil_Moderate; Sandy clay-shale; Silstone; Silty clay-shale; Slaty shale; Volcanic rock | — |
| 60817410140100_mc822#10bp01 1-240 tvd mud log | 21 | 21 | Anhydrite; Chalk; Clay-shale; Coal-lignite; Coarse sandstone; Conglomerate; Dolomite; Dolomitic limestone; Fine sandstone; Gypsum; Halite; Limestone; Marl; Medium sandstone; Oil_Good; Oil_Moderate; Sandy clay-shale; Silstone; Silty clay-shale; Slaty shale; Volcanic rock | — |
| 60817411400100_mc761_1_st1_mudlog_1md_17133_22886 | 11 | 11 | Anhydrite; Breccia; Claystone; Conglomerate; Dolomite; Limestone; Marlstone; Salt; Sandstone; Shale; Siltstone | — |
| 60817411400100_mc761_1_st1_mudlog_1tvd_17129_20968 | 11 | 11 | Anhydrite; Breccia; Claystone; Conglomerate; Dolomite; Limestone; Marlstone; Salt; Sandstone; Shale; Siltstone | — |
| 60817411400200_mc761_1_st1_bp01_mudlog_1md_17133_25775 | 11 | 11 | Anhydrite; Breccia; Claystone; Conglomerate; Dolomite; Limestone; Marlstone; Salt; Sandstone; Shale; Siltstone | — |
| 60817411420000_mc_540_#1_st00_bp00_masterlog_3400-24195_5in_md | 13 | 13 | Anhydrite; Claystone; Dolomite; Gas_Good; Gypsum; Halite; Limestone; Marl; Oil_Good; Salted water; Sandstone; Shale; Silstone | — |
| 60817411700000_mc776-5_5in_md_combo_mud_log_20415_mms | 21 | 21 | Anhydrite; Chalk; Chert; Clay-shale; Coal-lignite; Coarse sandstone; Dolomite; Dolomitic limestone; Fine sandstone; Gypsum; Halite; Limestone; Marl; Medium sandstone; Oil_Good; Oil_Moderate; Sandy clay-shale; Silstone; Silty clay-shale; Slaty shale; Volcanics | — |
| 60817412840000_mc_776_6_ml_mudlog_5757_23721_5in_md_final | 0 | 15 | Argill and calc sand; Argill limestone; Bituminous limestone; Calc clay-shale; Clay-shale; Dolomite; Fine sand; Fine sandstone; Halite; Limestone; Marl; Medium sand; Oolitic limestone; Silt; Siltstone | 10 |
| 60818400600100_608184006001_g08037_002_01_00_5 inch mudlog_13452ft-19912ft | 0 | 15 | Anhydrite; Chalk; Chert; Claystone; Dolomite; Dolomitic limestone; Granite; Gypsum; Halite; Limestone; Marl; Sandstone; Shale; Siltstone; Volcanics | 11 |
| 60818400650000_at0574_4_neptune_1inch_tvdmudlog_6300-17378 | 6 | 6 | Anhydrite; Clay/Shale; Dolomite; Sandstone; Siltstone; Tuff | — |
| Callaway 3-33 mudlog to TD.PDF | 5 | 5 | CEMENT CHALK; CHERT IGNEOUS LIMESTONE CLAYCHOKESAND CLAYSTONE SIDERITE or LIMONITE; SALT-PEPPERSAND GRANITE SANDSTONE GYPSUM; SHALE SHALE COLORED SHALE GRAY; TILL TUFF WELDED TUFF | — |
| Critter Creek 201-1807H_Mud_PDF_Vertical.pdf | 23 | 23 | ANHYDRITE; BENTONITE; BRECCIA; CEMENT; CHALK; CHERT; CLAYSTONE; COAL; CONGLOMERATE; DOLOMITE; GYPSUM; IGNEOUS; LIMESTONE; MARLSTONE; METAMORPHIC; SALT; SANDSTONE; SHALE; SHALE COLORED; SHALEGRAY; SILTSTONE; TILL; TUFF | — |
| Critter Creek 202-1807H_Mud_Vertical_PDF.pdf | 23 | 23 | ANHYDRITE; BENTONITE; BRECCIA; CEMENT; CHALK; CHERT; CLAYSTONE; COAL; CONGLOMERATE; DOLOMITE; GYPSUM; IGNEOUS; LIMESTONE; MARLSTONE; METAMORPHIC; SALT; SANDSTONE; SHALE; SHALE COLORED; SHALEGRAY; SILTSTONE; TILL; TUFF | — |
| MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft | 23 | 23 | Clay-CCS; Clay-M; Clay-S; Clay-SACS; Clay-SCS; Dolomites-AD; Dolomites-CD; Dolomites-D; Limestone-AL; Limestone-L; Limestone-SL; Others-A; Others-CL; Others-E; Others-GY; Others-H; Sand-CS; Sand-FS; Sand-FSS; Sand-MS; Sand-MSS; Sand-SIS; Sand-SS | — |
| MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft | 23 | 23 | Clay-CCS; Clay-M; Clay-S; Clay-SACS; Clay-SCS; Dolomites-AD; Dolomites-CD; Dolomites-D; Limestone-AL; Limestone-L; Limestone-SL; Others-A; Others-CL; Others-E; Others-GY; Others-H; Sand-CS; Sand-FS; Sand-FSS; Sand-MS; Sand-MSS; Sand-SIS; Sand-SS | — |
| MC765_P011_ST00BP00_5inchTVD_Depth_MudLog_3686-16495ft | 23 | 23 | Clay-CCS; Clay-M; Clay-S; Clay-SACS; Clay-SCS; Dolomites-AD; Dolomites-CD; Dolomites-D; Limestone-AL; Limestone-L; Limestone-SL; Others-A; Others-CL; Others-E; Others-GY; Others-H; Sand-CS; Sand-FS; Sand-FSS; Sand-MS; Sand-MSS; Sand-SIS; Sand-SS | — |
| MC765_P011_ST00BP01_5inchMD_Depth_MudLog_18230-21150ft | 23 | 23 | Clay-CCS; Clay-M; Clay-S; Clay-SACS; Clay-SCS; Dolomites-AD; Dolomites-CD; Dolomites-D; Limestone-AL; Limestone-L; Limestone-SL; Others-A; Others-CL; Others-E; Others-GY; Others-H; Sand-CS; Sand-FS; Sand-FSS; Sand-MS; Sand-MSS; Sand-SIS; Sand-SS | — |
| Mudlog-Trail Mtn1Palisades | 16 | 16 | ANHYDRITE; BENTONITE; CHERT; COAL; CONGLOMERATE; DOLOMITE; GRANITE; IGNEOUS; LIMESTONE; METAMORPHIC; SALT; SANDSTONE; SHALE; SILTSTONE; TUFF; VOLCANICS | — |
| Mudlog_MD_FREEMAN 3 24 ST | 47 | 47 | ANHYDRITE; BENTONITE; BIOTITIZATION; BRECCIA; CALCARENITE; CALCAREOUSTUFF; CALCILUTITE; CARBONACEOUSMAT; CARBONATES; CEMENTCONTAM.; CHERT-ARGILL; CHERT-GLASSY; CHERT-PORCEL; COAL; CONGL.SAND; CONGL.SANDSTONE; CONGLOMERATE; COQUINA; CRYSTALLINE TUFF; DIATOMITE; DOLOSTONE; FELSIC SILIC DIKE; GABBRO; GLASSYTUFF; GRANITE; GRANITE WASH; GYPSUM; HALITE; IGNEOUS (ACIDIC); IGNEOUS (BASIC); INTRUSIVES; LIMESTONE; LITHIC TUFF; MARL-CALC; MARL-DOLO; METAMORPHICS; MUDSTONE; OBSIDIAN; PALEOSOL; PORCELANEOUSCLYS; QUARTZMONZONITE; RECRYSTALLIZEDCALC; RHYOLITE; SALT; SAND; SANDSTONE; SANDSTONE-TUFFACE | — |
| Rio_Lavaca_1748_Unit_1_241_MUD | 23 | 23 | ANHYDRITE; BENTONITE; BRECCIA; CEMENT; CHERT; CLAYSTONE; COAL; CONGLOMERATE; DOLOMITE; GYPSUM; IGNEOUS; LIMESTONE; MARLSTONE; METAMORPHIC; SALT; SANDSTONE; SHALE; SHALE COLORED; SHALE GRAY; SIDERITEorLIMONITE; SILTSTONE; TILL; TUFF | — |
| Rio_Nueces_3744_Unit_1_212_MUD_5IN | 23 | 23 | ANHYDRITE; BENTONITE; BRECCIA; CEMENT; CHERT; CLAYSTONE; COAL; CONGLOMERATE; DOLOMITE; GYPSUM; IGNEOUS; LIMESTONE; MARLSTONE; METAMORPHIC; SALT; SANDSTONE; SHALE; SHALE COLORED; SHALE GRAY; SIDERITEorLIMONITE; SILTSTONE; TILL; TUFF | — |
| Rio_Nueces_3744_Unit_1_222_5IN_MUD | 23 | 23 | ANHYDRITE; BENTONITE; BRECCIA; CEMENT; CHERT; CLAYSTONE; COAL; CONGLOMERATE; DOLOMITE; GYPSUM; IGNEOUS; LIMESTONE; MARLSTONE; METAMORPHIC; SALT; SANDSTONE; SHALE; SHALE COLORED; SHALE GRAY; SIDERITEorLIMONITE; SILTSTONE; TILL; TUFF | — |
| Rio_San_Gabriel_2829_Unit_2_272_MUD | 14 | 14 | :D HBHHHEHHHRH; :LSTONE; ANHYC; CLAYS; COAL; DOLO; E; ESTONE; GYPSL; SALT; SAND:; SHALE; SIDER; SILTS | — |
| Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft | 2 | 2 | Clay-shale; Fine sandstone | — |
| Shell_AC0336_OCS-G-36102_001_ST00BP01_5inchMD_MudLog_9100 ft -27770 ft | 7 | 7 | Clay-S; Limestone-L; Others-H; Others-S; Sand-FSS; Sand-MS; Sand-MSS | — |

## Regression diff (all 40 previously working logs)

40/40 identical lithology label sets; 0 differences.

| Log | Added labels | Removed labels |
|---|---|---|
| 0000000009_d-582_0000020435-0001_log01_fe_1in_md | — | — |
| 0000000009_d-582_0000020435-0001_log02_fe_5in_md | — | — |
| 0000000009_d-582_0000020435-0001_log03_chromatograph_ratio | — | — |
| 0000000009_d-582_0000020435-0001_log04_fe_5in_md | — | — |
| 0000000009_d-582_0000020435-0001_log05_chromatograph_ratio | — | — |
| 0000000009_d-582_0000020435-0001_log06_fe_5in_md | — | — |
| 0000000009_d-582_0000020435-0001_log07_chromatograph_ratio | — | — |
| 0000000009_d-582_0000020435-0001_log08_fe_5in_md | — | — |
| 0000000009_d-582_0000020435-0001_log09_chromatograph_ratio | — | — |
| 60807402860000_gb506#2_md_mudlog | — | — |
| 60808400140000_hadrian 2_masterlog_2in_10000-21962 | — | — |
| 60811405120000_1inch tvd mudlog | — | — |
| 60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872 | — | — |
| 60811407570000_bp_gc743_15607_dc802_st00bp00_mudlog_5md_6900-17643 | — | — |
| 608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD | — | — |
| 60812400480000_ocs-g 26419 001 st00bp00_masterlog_7860-27400_5in_md_2 | — | — |
| 60812400810000_ocs-g 16987 003 st00bp00_1inmd_mudlog_8921_15721 | — | — |
| 60817410140000_mc822#10 1-1200 mud log | — | — |
| 60817410140100_mc822#10bp01 1-240 tvd mud log | — | — |
| 60817411400100_mc761_1_st1_mudlog_1md_17133_22886 | — | — |
| 60817411400100_mc761_1_st1_mudlog_1tvd_17129_20968 | — | — |
| 60817411400200_mc761_1_st1_bp01_mudlog_1md_17133_25775 | — | — |
| 60817411420000_mc_540_#1_st00_bp00_masterlog_3400-24195_5in_md | — | — |
| 60817411700000_mc776-5_5in_md_combo_mud_log_20415_mms | — | — |
| 60818400650000_at0574_4_neptune_1inch_tvdmudlog_6300-17378 | — | — |
| Callaway 3-33 mudlog to TD.PDF | — | — |
| Critter Creek 201-1807H_Mud_PDF_Vertical.pdf | — | — |
| Critter Creek 202-1807H_Mud_Vertical_PDF.pdf | — | — |
| MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft | — | — |
| MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft | — | — |
| MC765_P011_ST00BP00_5inchTVD_Depth_MudLog_3686-16495ft | — | — |
| MC765_P011_ST00BP01_5inchMD_Depth_MudLog_18230-21150ft | — | — |
| Mudlog-Trail Mtn1Palisades | — | — |
| Mudlog_MD_FREEMAN 3 24 ST | — | — |
| Rio_Lavaca_1748_Unit_1_241_MUD | — | — |
| Rio_Nueces_3744_Unit_1_212_MUD_5IN | — | — |
| Rio_Nueces_3744_Unit_1_222_5IN_MUD | — | — |
| Rio_San_Gabriel_2829_Unit_2_272_MUD | — | — |
| Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft | — | — |
| Shell_AC0336_OCS-G-36102_001_ST00BP01_5inchMD_MudLog_9100 ft -27770 ft | — | — |

## Timing (seconds per legend)

Measured with `perf_counter`. Uncached means the v2 JSON was absent: it includes reading the TIFF crop, OCR, detection and writing cache/debug PNGs. The model is initialized once per Python process. Cached is the median of three calls, including JSON and swatch PNG reads; no in-memory result cache. Original-cache OCR is never reused by v2.

| Group | Uncached mean | Uncached median | Uncached range | Cached mean | Cached median | Cached range |
|---|---:|---:|---|---:|---:|---|
| All 51 | 6.702 | 6.122 | 2.410–18.696 | 0.0183 | 0.0186 | 0.0033–0.0361 |
| 11 repaired | 8.328 | 7.895 | 5.620–11.173 | 0.0192 | 0.0193 | 0.0156–0.0229 |
| 40 regression | 6.254 | 6.046 | 2.410–18.696 | 0.0181 | 0.0180 | 0.0033–0.0361 |

| Log | Uncached seconds | Cached seconds |
|---|---:|---:|
| 0000000009_d-582_0000020435-0001_log01_fe_1in_md | 8.352 | 0.0128 |
| 0000000009_d-582_0000020435-0001_log02_fe_5in_md | 6.038 | 0.0153 |
| 0000000009_d-582_0000020435-0001_log03_chromatograph_ratio | 6.100 | 0.0146 |
| 0000000009_d-582_0000020435-0001_log04_fe_5in_md | 5.741 | 0.0147 |
| 0000000009_d-582_0000020435-0001_log05_chromatograph_ratio | 5.969 | 0.0148 |
| 0000000009_d-582_0000020435-0001_log06_fe_5in_md | 6.284 | 0.0143 |
| 0000000009_d-582_0000020435-0001_log07_chromatograph_ratio | 6.143 | 0.0159 |
| 0000000009_d-582_0000020435-0001_log08_fe_5in_md | 6.117 | 0.0149 |
| 0000000009_d-582_0000020435-0001_log09_chromatograph_ratio | 6.204 | 0.0154 |
| 60807402860000_gb506#2_md_mudlog | 4.966 | 0.0190 |
| 60808400140000_hadrian 2_masterlog_2in_10000-21962 | 4.608 | 0.0194 |
| 60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev | 6.704 | 0.0186 |
| 60811405120000_1inch tvd mudlog | 3.657 | 0.0156 |
| 60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400 | 7.895 | 0.0193 |
| 60811405800000_608114058000_g16786_001_00_00_md5_mud_show_2 | 5.954 | 0.0195 |
| 60811405800100_mdn-1_st00bp01_5 inch tvdmudlog_4560ft-22998ft | 5.620 | 0.0186 |
| 60811405800300_mdn-1_st01bp01_5 inch tvdmudlog_4560ft to 20251ft | 7.212 | 0.0181 |
| 60811405800500_608114058005_g16786_001_03_00_tvd5_mud | 8.476 | 0.0205 |
| 60811406050000_608114060500_g31765 001 st00 bp00__master log_5640-28740_5 in_md | 11.142 | 0.0156 |
| 60811406050100_ardennes_ocs-g31765 001 st00 bp01__master log_5640-36552_5 in_md | 11.173 | 0.0196 |
| 60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872 | 11.048 | 0.0293 |
| 60811407570000_bp_gc743_15607_dc802_st00bp00_mudlog_5md_6900-17643 | 8.604 | 0.0273 |
| 608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD | 9.695 | 0.0242 |
| 60812400480000_ocs-g 26419 001 st00bp00_masterlog_7860-27400_5in_md_2 | 6.954 | 0.0190 |
| 60812400810000_ocs-g 16987 003 st00bp00_1inmd_mudlog_8921_15721 | 4.200 | 0.0174 |
| 60812401020000_xom_ocs-g_20351_ju103_st00bp00_mudlog_10200-30900ft_5in_tvd | 10.622 | 0.0202 |
| 60817410140000_mc822#10 1-1200 mud log | 3.926 | 0.0173 |
| 60817410140100_mc822#10bp01 1-240 tvd mud log | 3.644 | 0.0204 |
| 60817411400100_mc761_1_st1_mudlog_1md_17133_22886 | 5.487 | 0.0183 |
| 60817411400100_mc761_1_st1_mudlog_1tvd_17129_20968 | 5.232 | 0.0176 |
| 60817411400200_mc761_1_st1_bp01_mudlog_1md_17133_25775 | 4.589 | 0.0213 |
| 60817411420000_mc_540_#1_st00_bp00_masterlog_3400-24195_5in_md | 5.762 | 0.0210 |
| 60817411700000_mc776-5_5in_md_combo_mud_log_20415_mms | 3.864 | 0.0168 |
| 60817412840000_mc_776_6_ml_mudlog_5757_23721_5in_md_final | 10.686 | 0.0229 |
| 60818400600100_608184006001_g08037_002_01_00_5 inch mudlog_13452ft-19912ft | 6.122 | 0.0184 |
| 60818400650000_at0574_4_neptune_1inch_tvdmudlog_6300-17378 | 7.495 | 0.0189 |
| Callaway 3-33 mudlog to TD.PDF | 7.420 | 0.0072 |
| Critter Creek 201-1807H_Mud_PDF_Vertical.pdf | 8.351 | 0.0191 |
| Critter Creek 202-1807H_Mud_Vertical_PDF.pdf | 8.288 | 0.0186 |
| MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft | 3.508 | 0.0228 |
| MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft | 3.688 | 0.0228 |
| MC765_P011_ST00BP00_5inchTVD_Depth_MudLog_3686-16495ft | 3.445 | 0.0239 |
| MC765_P011_ST00BP01_5inchMD_Depth_MudLog_18230-21150ft | 3.551 | 0.0204 |
| Mudlog-Trail Mtn1Palisades | 6.053 | 0.0142 |
| Mudlog_MD_FREEMAN 3 24 ST | 18.696 | 0.0361 |
| Rio_Lavaca_1748_Unit_1_241_MUD | 9.801 | 0.0166 |
| Rio_Nueces_3744_Unit_1_212_MUD_5IN | 7.944 | 0.0150 |
| Rio_Nueces_3744_Unit_1_222_5IN_MUD | 6.815 | 0.0189 |
| Rio_San_Gabriel_2829_Unit_2_272_MUD | 6.518 | 0.0212 |
| Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft | 3.007 | 0.0033 |
| Shell_AC0336_OCS-G-36102_001_ST00BP01_5inchMD_MudLog_9100 ft -27770 ft | 2.410 | 0.0065 |

## Reproduction and checks

Run from the bake-off root:

```text
python -X utf8 -m notes.legfix_audit
python -X utf8 -m notes.legfix_audit --compare
python -X utf8 -m notes.legfix_audit --report
python -X utf8 -m notes.legfix_checks
```

Five focused checks cover compact/singular headings, full and shortened flattened rows, real ruled panel exclusion (even for rock labels), original LegendEntry type, full-image coordinates and exact cached swatch round trips. The full audit checks every entry stays inside the annotated legend box, RGB shape/dtype and cached label/flag/box equality. Machine-readable results: `notes/legfix_baseline.json` and `notes/legfix_results.json`.
